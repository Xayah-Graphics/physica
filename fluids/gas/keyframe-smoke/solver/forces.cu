#include "../domain/device.cuh"
#include "kernels.h"
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    namespace {
        constexpr float smooth_epsilon = 1.0e-6F;

        __device__ float centered_load(const float* values, int x, int y, int z, const Grid grid) {
            x = max(0, min(grid.nx - 1, x));
            y = max(0, min(grid.ny - 1, y));
            z = max(0, min(grid.nz - 1, z));
            return values[index3(x, y, z, grid.nx, grid.ny)];
        }

        __device__ void centered_scatter(double* values, int x, int y, int z, const Grid grid, const double value) {
            x = max(0, min(grid.nx - 1, x));
            y = max(0, min(grid.ny - 1, y));
            z = max(0, min(grid.nz - 1, z));
            atomicAdd(values + index3(x, y, z, grid.nx, grid.ny), value);
        }

        __global__ void initialize_force_kernel(const Grid grid, const float density_buoyancy, const float* density, const ConstCenteredVectorView control, const CenteredVectorView physical, const CenteredVectorView control_copy, const CenteredVectorView total) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            physical.x[index] = 0.0F;
            physical.y[index] = density_buoyancy * density[index];
            physical.z[index] = 0.0F;
            control_copy.x[index] = control.x[index];
            control_copy.y[index] = control.y[index];
            control_copy.z[index] = control.z[index];
            total.x[index] = control.x[index];
            total.y[index] = physical.y[index] + control.y[index];
            total.z[index] = control.z[index];
        }

        __global__ void initialize_force_tangent_kernel(const Grid grid, const float density_buoyancy, const float* density_tangent, const ConstCenteredVectorView control_tangent, const CenteredVectorView physical_tangent, const CenteredVectorView total_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            physical_tangent.x[index] = 0.0F;
            physical_tangent.y[index] = density_buoyancy * density_tangent[index];
            physical_tangent.z[index] = 0.0F;
            total_tangent.x[index] = control_tangent.x[index];
            total_tangent.y[index] = physical_tangent.y[index] + control_tangent.y[index];
            total_tangent.z[index] = control_tangent.z[index];
        }

        __global__ void centered_velocity_kernel(const Grid grid, const ConstStaggeredVectorView velocity, const CenteredVectorView centered) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            centered.x[index] = 0.5F * (velocity.x[index3(x, y, z, grid.nx + 1, grid.ny)] + velocity.x[index3(x + 1, y, z, grid.nx + 1, grid.ny)]);
            centered.y[index] = 0.5F * (velocity.y[index3(x, y, z, grid.nx, grid.ny + 1)] + velocity.y[index3(x, y + 1, z, grid.nx, grid.ny + 1)]);
            centered.z[index] = 0.5F * (velocity.z[index3(x, y, z, grid.nx, grid.ny)] + velocity.z[index3(x, y, z + 1, grid.nx, grid.ny)]);
        }

        __global__ void curl_magnitude_kernel(const Grid grid, const ConstCenteredVectorView centered, const CenteredVectorView vorticity, float* magnitude) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const float scale = 0.5F / grid.cell_size;
            const float wx = scale * (centered_load(centered.z, x, y + 1, z, grid) - centered_load(centered.z, x, y - 1, z, grid) - centered_load(centered.y, x, y, z + 1, grid) + centered_load(centered.y, x, y, z - 1, grid));
            const float wy = scale * (centered_load(centered.x, x, y, z + 1, grid) - centered_load(centered.x, x, y, z - 1, grid) - centered_load(centered.z, x + 1, y, z, grid) + centered_load(centered.z, x - 1, y, z, grid));
            const float wz = scale * (centered_load(centered.y, x + 1, y, z, grid) - centered_load(centered.y, x - 1, y, z, grid) - centered_load(centered.x, x, y + 1, z, grid) + centered_load(centered.x, x, y - 1, z, grid));
            vorticity.x[index] = wx;
            vorticity.y[index] = wy;
            vorticity.z[index] = wz;
            magnitude[index] = sqrtf(wx * wx + wy * wy + wz * wz + smooth_epsilon * smooth_epsilon);
        }

        __global__ void normal_kernel(const Grid grid, const float* magnitude, const CenteredVectorView normal, float* normalizer) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const float scale = 0.5F / grid.cell_size;
            const float gx = scale * (centered_load(magnitude, x + 1, y, z, grid) - centered_load(magnitude, x - 1, y, z, grid));
            const float gy = scale * (centered_load(magnitude, x, y + 1, z, grid) - centered_load(magnitude, x, y - 1, z, grid));
            const float gz = scale * (centered_load(magnitude, x, y, z + 1, grid) - centered_load(magnitude, x, y, z - 1, grid));
            const float length = sqrtf(gx * gx + gy * gy + gz * gz + smooth_epsilon * smooth_epsilon);
            normal.x[index] = gx / length;
            normal.y[index] = gy / length;
            normal.z[index] = gz / length;
            normalizer[index] = length;
        }

        __global__ void add_vorticity_force_kernel(const Grid grid, const float confinement, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView normal, const CenteredVectorView physical, const CenteredVectorView total) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            const float scale = confinement * grid.cell_size;
            const Vector cross{
                normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index],
                normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index],
                normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index],
            };
            physical.x[index] += scale * cross.x;
            physical.y[index] += scale * cross.y;
            physical.z[index] += scale * cross.z;
            total.x[index] += scale * cross.x;
            total.y[index] += scale * cross.y;
            total.z[index] += scale * cross.z;
        }

        __global__ void magnitude_tangent_kernel(const Grid grid, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView vorticity_tangent, const float* magnitude, float* magnitude_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            magnitude_tangent[index] = (vorticity.x[index] * vorticity_tangent.x[index] + vorticity.y[index] * vorticity_tangent.y[index] + vorticity.z[index] * vorticity_tangent.z[index]) / magnitude[index];
        }

        __global__ void normal_tangent_kernel(const Grid grid, const float* magnitude_tangent, const ConstCenteredVectorView normal, const float* normalizer, const CenteredVectorView normal_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const float scale = 0.5F / grid.cell_size;
            const Vector gradient_tangent{
                scale * (centered_load(magnitude_tangent, x + 1, y, z, grid) - centered_load(magnitude_tangent, x - 1, y, z, grid)),
                scale * (centered_load(magnitude_tangent, x, y + 1, z, grid) - centered_load(magnitude_tangent, x, y - 1, z, grid)),
                scale * (centered_load(magnitude_tangent, x, y, z + 1, grid) - centered_load(magnitude_tangent, x, y, z - 1, grid)),
            };
            const float projection = normal.x[index] * gradient_tangent.x + normal.y[index] * gradient_tangent.y + normal.z[index] * gradient_tangent.z;
            normal_tangent.x[index] = (gradient_tangent.x - normal.x[index] * projection) / normalizer[index];
            normal_tangent.y[index] = (gradient_tangent.y - normal.y[index] * projection) / normalizer[index];
            normal_tangent.z[index] = (gradient_tangent.z - normal.z[index] * projection) / normalizer[index];
        }

        __global__ void add_vorticity_force_tangent_kernel(const Grid grid, const float confinement, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView vorticity_tangent, const ConstCenteredVectorView normal, const ConstCenteredVectorView normal_tangent, const CenteredVectorView physical_tangent, const CenteredVectorView total_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            const float scale = confinement * grid.cell_size;
            const Vector cross_tangent{
                normal_tangent.y[index] * vorticity.z[index] + normal.y[index] * vorticity_tangent.z[index] - normal_tangent.z[index] * vorticity.y[index] - normal.z[index] * vorticity_tangent.y[index],
                normal_tangent.z[index] * vorticity.x[index] + normal.z[index] * vorticity_tangent.x[index] - normal_tangent.x[index] * vorticity.z[index] - normal.x[index] * vorticity_tangent.z[index],
                normal_tangent.x[index] * vorticity.y[index] + normal.x[index] * vorticity_tangent.y[index] - normal_tangent.y[index] * vorticity.x[index] - normal.y[index] * vorticity_tangent.x[index],
            };
            physical_tangent.x[index] += scale * cross_tangent.x;
            physical_tangent.y[index] += scale * cross_tangent.y;
            physical_tangent.z[index] += scale * cross_tangent.z;
            total_tangent.x[index] += scale * cross_tangent.x;
            total_tangent.y[index] += scale * cross_tangent.y;
            total_tangent.z[index] += scale * cross_tangent.z;
        }

        __global__ void split_force_adjoint_kernel(const Grid grid, const float density_buoyancy, const ConstCenteredVectorAdjointView total_adjoint, double* density_adjoint, const CenteredVectorAdjointView physical_adjoint, const CenteredVectorAdjointView control_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            physical_adjoint.x[index] = total_adjoint.x[index];
            physical_adjoint.y[index] = total_adjoint.y[index];
            physical_adjoint.z[index] = total_adjoint.z[index];
            control_adjoint.x[index] += total_adjoint.x[index];
            control_adjoint.y[index] += total_adjoint.y[index];
            control_adjoint.z[index] += total_adjoint.z[index];
            density_adjoint[index] += density_buoyancy * total_adjoint.y[index];
        }

        __global__ void vorticity_force_reverse_kernel(const Grid grid, const float confinement, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView normal, const ConstCenteredVectorAdjointView force_adjoint, const CenteredVectorAdjointView vorticity_adjoint, const CenteredVectorAdjointView normal_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            const double scale = static_cast<double>(confinement) * grid.cell_size;
            normal_adjoint.x[index] += scale * (vorticity.y[index] * force_adjoint.z[index] - vorticity.z[index] * force_adjoint.y[index]);
            normal_adjoint.y[index] += scale * (vorticity.z[index] * force_adjoint.x[index] - vorticity.x[index] * force_adjoint.z[index]);
            normal_adjoint.z[index] += scale * (vorticity.x[index] * force_adjoint.y[index] - vorticity.y[index] * force_adjoint.x[index]);
            vorticity_adjoint.x[index] += scale * (force_adjoint.y[index] * normal.z[index] - force_adjoint.z[index] * normal.y[index]);
            vorticity_adjoint.y[index] += scale * (force_adjoint.z[index] * normal.x[index] - force_adjoint.x[index] * normal.z[index]);
            vorticity_adjoint.z[index] += scale * (force_adjoint.x[index] * normal.y[index] - force_adjoint.y[index] * normal.x[index]);
        }

        __global__ void normal_reverse_kernel(const Grid grid, const ConstCenteredVectorView normal, const float* normalizer, const ConstCenteredVectorAdjointView normal_adjoint, double* magnitude_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double projection = normal.x[index] * normal_adjoint.x[index] + normal.y[index] * normal_adjoint.y[index] + normal.z[index] * normal_adjoint.z[index];
            const double gx = (normal_adjoint.x[index] - normal.x[index] * projection) / normalizer[index];
            const double gy = (normal_adjoint.y[index] - normal.y[index] * projection) / normalizer[index];
            const double gz = (normal_adjoint.z[index] - normal.z[index] * projection) / normalizer[index];
            const double scale = 0.5 / grid.cell_size;
            centered_scatter(magnitude_adjoint, x + 1, y, z, grid, scale * gx);
            centered_scatter(magnitude_adjoint, x - 1, y, z, grid, -scale * gx);
            centered_scatter(magnitude_adjoint, x, y + 1, z, grid, scale * gy);
            centered_scatter(magnitude_adjoint, x, y - 1, z, grid, -scale * gy);
            centered_scatter(magnitude_adjoint, x, y, z + 1, grid, scale * gz);
            centered_scatter(magnitude_adjoint, x, y, z - 1, grid, -scale * gz);
        }

        __global__ void magnitude_reverse_kernel(const Grid grid, const ConstCenteredVectorView vorticity, const float* magnitude, const double* magnitude_adjoint, const CenteredVectorAdjointView vorticity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            const double scale = magnitude_adjoint[index] / magnitude[index];
            vorticity_adjoint.x[index] += scale * vorticity.x[index];
            vorticity_adjoint.y[index] += scale * vorticity.y[index];
            vorticity_adjoint.z[index] += scale * vorticity.z[index];
        }

        __global__ void curl_reverse_kernel(const Grid grid, const ConstCenteredVectorAdjointView vorticity_adjoint, const CenteredVectorAdjointView centered_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double scale = 0.5 / grid.cell_size;
            const double ax = vorticity_adjoint.x[index] * scale;
            const double ay = vorticity_adjoint.y[index] * scale;
            const double az = vorticity_adjoint.z[index] * scale;
            centered_scatter(centered_adjoint.z, x, y + 1, z, grid, ax);
            centered_scatter(centered_adjoint.z, x, y - 1, z, grid, -ax);
            centered_scatter(centered_adjoint.y, x, y, z + 1, grid, -ax);
            centered_scatter(centered_adjoint.y, x, y, z - 1, grid, ax);
            centered_scatter(centered_adjoint.x, x, y, z + 1, grid, ay);
            centered_scatter(centered_adjoint.x, x, y, z - 1, grid, -ay);
            centered_scatter(centered_adjoint.z, x + 1, y, z, grid, -ay);
            centered_scatter(centered_adjoint.z, x - 1, y, z, grid, ay);
            centered_scatter(centered_adjoint.y, x + 1, y, z, grid, az);
            centered_scatter(centered_adjoint.y, x - 1, y, z, grid, -az);
            centered_scatter(centered_adjoint.x, x, y + 1, z, grid, -az);
            centered_scatter(centered_adjoint.x, x, y - 1, z, grid, az);
        }

        __global__ void centered_velocity_reverse_kernel(const Grid grid, const ConstCenteredVectorAdjointView centered_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            atomicAdd(velocity_adjoint.x + index3(x, y, z, grid.nx + 1, grid.ny), 0.5 * centered_adjoint.x[index]);
            atomicAdd(velocity_adjoint.x + index3(x + 1, y, z, grid.nx + 1, grid.ny), 0.5 * centered_adjoint.x[index]);
            atomicAdd(velocity_adjoint.y + index3(x, y, z, grid.nx, grid.ny + 1), 0.5 * centered_adjoint.y[index]);
            atomicAdd(velocity_adjoint.y + index3(x, y + 1, z, grid.nx, grid.ny + 1), 0.5 * centered_adjoint.y[index]);
            atomicAdd(velocity_adjoint.z + index3(x, y, z, grid.nx, grid.ny), 0.5 * centered_adjoint.z[index]);
            atomicAdd(velocity_adjoint.z + index3(x, y, z + 1, grid.nx, grid.ny), 0.5 * centered_adjoint.z[index]);
        }
    } // namespace

    void force_forward(const ::cuda::stream_ref stream, const Grid grid, const float density_buoyancy, const float vorticity_confinement, const ConstScalarView density, const ConstStaggeredVectorView velocity, const ConstCenteredVectorView control, const VorticityView vorticity, const CenteredVectorView physical_force, const CenteredVectorView control_force, const CenteredVectorView total_force) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), initialize_force_kernel, grid, density_buoyancy, density.values, control, physical_force, control_force, total_force);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), centered_velocity_kernel, grid, velocity, vorticity.centered_velocity);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), curl_magnitude_kernel, grid, ConstCenteredVectorView{vorticity.centered_velocity.x, vorticity.centered_velocity.y, vorticity.centered_velocity.z}, vorticity.vorticity, vorticity.magnitude.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), normal_kernel, grid, vorticity.magnitude.values, vorticity.normal, vorticity.normalizer.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), add_vorticity_force_kernel, grid, vorticity_confinement, ConstCenteredVectorView{vorticity.vorticity.x, vorticity.vorticity.y, vorticity.vorticity.z}, ConstCenteredVectorView{vorticity.normal.x, vorticity.normal.y, vorticity.normal.z}, physical_force, total_force);
    }

    void force_jvp(const ::cuda::stream_ref stream, const Grid grid, const float density_buoyancy, const float vorticity_confinement, const ConstScalarView density_tangent, const ConstStaggeredVectorView velocity_tangent, const ConstCenteredVectorView control_tangent, const ConstVorticityView cache, const VorticityTangentView vorticity_tangent, const CenteredVectorView physical_force_tangent, const CenteredVectorView total_force_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), initialize_force_tangent_kernel, grid, density_buoyancy, density_tangent.values, control_tangent, physical_force_tangent, total_force_tangent);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), centered_velocity_kernel, grid, velocity_tangent, vorticity_tangent.centered_velocity);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), curl_magnitude_kernel, grid, ConstCenteredVectorView{vorticity_tangent.centered_velocity.x, vorticity_tangent.centered_velocity.y, vorticity_tangent.centered_velocity.z}, vorticity_tangent.vorticity, vorticity_tangent.magnitude.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), magnitude_tangent_kernel, grid, cache.vorticity, ConstCenteredVectorView{vorticity_tangent.vorticity.x, vorticity_tangent.vorticity.y, vorticity_tangent.vorticity.z}, cache.magnitude.values, vorticity_tangent.magnitude.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), normal_tangent_kernel, grid, vorticity_tangent.magnitude.values, cache.normal, cache.normalizer.values, vorticity_tangent.normal);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), add_vorticity_force_tangent_kernel, grid, vorticity_confinement, cache.vorticity, ConstCenteredVectorView{vorticity_tangent.vorticity.x, vorticity_tangent.vorticity.y, vorticity_tangent.vorticity.z}, cache.normal, ConstCenteredVectorView{vorticity_tangent.normal.x, vorticity_tangent.normal.y, vorticity_tangent.normal.z}, physical_force_tangent, total_force_tangent);
    }

    void force_vjp(const ::cuda::stream_ref stream, const Grid grid, const float density_buoyancy, const float vorticity_confinement, const ConstVorticityView cache, const ConstCenteredVectorAdjointView total_force_adjoint, const ScalarAdjointView density_adjoint, const StaggeredVectorAdjointView velocity_adjoint, const CenteredVectorAdjointView physical_force_adjoint, const CenteredVectorAdjointView control_adjoint, const VorticityAdjointView scratch) {
        const std::size_t count = static_cast<std::size_t>(cell_count(grid));
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.centered_velocity.x, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.centered_velocity.y, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.centered_velocity.z, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.vorticity.x, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.vorticity.y, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.vorticity.z, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.magnitude.values, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.normal.x, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.normal.y, count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{scratch.normal.z, count}, 0u);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), split_force_adjoint_kernel, grid, density_buoyancy, total_force_adjoint, density_adjoint.values, physical_force_adjoint, control_adjoint);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), vorticity_force_reverse_kernel, grid, vorticity_confinement, cache.vorticity, cache.normal, ConstCenteredVectorAdjointView{physical_force_adjoint.x, physical_force_adjoint.y, physical_force_adjoint.z}, scratch.vorticity, scratch.normal);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), normal_reverse_kernel, grid, cache.normal, cache.normalizer.values, ConstCenteredVectorAdjointView{scratch.normal.x, scratch.normal.y, scratch.normal.z}, scratch.magnitude.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), magnitude_reverse_kernel, grid, cache.vorticity, cache.magnitude.values, scratch.magnitude.values, scratch.vorticity);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), curl_reverse_kernel, grid, ConstCenteredVectorAdjointView{scratch.vorticity.x, scratch.vorticity.y, scratch.vorticity.z}, scratch.centered_velocity);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), centered_velocity_reverse_kernel, grid, ConstCenteredVectorAdjointView{scratch.centered_velocity.x, scratch.centered_velocity.y, scratch.centered_velocity.z}, velocity_adjoint);
    }
} // namespace physica::fluids::gas::keyframe_smoke::cuda_detail
