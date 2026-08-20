#include "../domain/device.cuh"
#include "kernels.h"
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>

namespace physica::fluids::gas::smoke::cuda_detail {
    namespace {
        constexpr float smooth_epsilon = 1.0e-6F;

        __global__ void buoyancy_forward_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* density, const float* temperature, const ConstCenteredVectorView external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const CenteredVectorView force) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (cell_mask[index] != 0u) {
                force.x[index] = 0.0F;
                force.y[index] = 0.0F;
                force.z[index] = 0.0F;
                return;
            }
            force.x[index] = external_acceleration.x[index];
            force.y[index] = external_acceleration.y[index] + density_buoyancy[0] * density[index] + temperature_buoyancy[0] * (temperature[index] - ambient_temperature[0]);
            force.z[index] = external_acceleration.z[index];
        }

        __global__ void buoyancy_jvp_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* density, const float* temperature, const float* density_tangent, const float* temperature_tangent, const ConstCenteredVectorView external_tangent, const float* ambient, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, const CenteredVectorView output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (cell_mask[index] != 0u) {
                output.x[index] = 0.0F;
                output.y[index] = 0.0F;
                output.z[index] = 0.0F;
                return;
            }
            output.x[index] = external_tangent.x[index];
            output.y[index] = external_tangent.y[index] + density_buoyancy[0] * density_tangent[index] + density_buoyancy_tangent[0] * density[index] + temperature_buoyancy[0] * (temperature_tangent[index] - ambient_tangent[0]) + temperature_buoyancy_tangent[0] * (temperature[index] - ambient[0]);
            output.z[index] = external_tangent.z[index];
        }

        __global__ void buoyancy_vjp_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* density, const float* temperature, const float* ambient, const float* density_buoyancy, const float* temperature_buoyancy, const ConstCenteredVectorAdjointView force_adjoint, double* density_adjoint, double* temperature_adjoint, const CenteredVectorAdjointView external_adjoint, double* ambient_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            external_adjoint.x[index] += force_adjoint.x[index];
            external_adjoint.y[index] += force_adjoint.y[index];
            external_adjoint.z[index] += force_adjoint.z[index];
            density_adjoint[index] += density_buoyancy[0] * force_adjoint.y[index];
            temperature_adjoint[index] += temperature_buoyancy[0] * force_adjoint.y[index];
            atomicAdd(ambient_adjoint, -temperature_buoyancy[0] * force_adjoint.y[index]);
            atomicAdd(density_buoyancy_adjoint, density[index] * force_adjoint.y[index]);
            atomicAdd(temperature_buoyancy_adjoint, (temperature[index] - ambient[0]) * force_adjoint.y[index]);
        }
        __device__ float centered_load(const float* values, int x, int y, int z, const Grid grid) {
            x = max(0, min(grid.nx - 1, x));
            y = max(0, min(grid.ny - 1, y));
            z = max(0, min(grid.nz - 1, z));
            return values[index3(x, y, z, grid.nx, grid.ny)];
        }

        __device__ void centered_scatter(double* values, int x, int y, int z, const Grid grid, const double adjoint) {
            x = max(0, min(grid.nx - 1, x));
            y = max(0, min(grid.ny - 1, y));
            z = max(0, min(grid.nz - 1, z));
            atomicAdd(values + index3(x, y, z, grid.nx, grid.ny), adjoint);
        }

        __global__ void centered_velocity_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const CenteredVectorView centered) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (cell_mask[index] != 0u) {
                centered.x[index] = 0.0F;
                centered.y[index] = 0.0F;
                centered.z[index] = 0.0F;
                return;
            }
            centered.x[index] = 0.5F * (velocity.x[index3(x, y, z, grid.nx + 1, grid.ny)] + velocity.x[index3(x + 1, y, z, grid.nx + 1, grid.ny)]);
            centered.y[index] = 0.5F * (velocity.y[index3(x, y, z, grid.nx, grid.ny + 1)] + velocity.y[index3(x, y + 1, z, grid.nx, grid.ny + 1)]);
            centered.z[index] = 0.5F * (velocity.z[index3(x, y, z, grid.nx, grid.ny)] + velocity.z[index3(x, y, z + 1, grid.nx, grid.ny)]);
        }

        __global__ void curl_magnitude_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView centered, const CenteredVectorView vorticity, float* magnitude) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (cell_mask[index] != 0u) {
                vorticity.x[index] = 0.0F;
                vorticity.y[index] = 0.0F;
                vorticity.z[index] = 0.0F;
                magnitude[index]   = smooth_epsilon;
                return;
            }
            const float inverse_spacing = 0.5F / grid.cell_size;
            const float wx              = (centered_load(centered.z, x, y + 1, z, grid) - centered_load(centered.z, x, y - 1, z, grid) - centered_load(centered.y, x, y, z + 1, grid) + centered_load(centered.y, x, y, z - 1, grid)) * inverse_spacing;
            const float wy              = (centered_load(centered.x, x, y, z + 1, grid) - centered_load(centered.x, x, y, z - 1, grid) - centered_load(centered.z, x + 1, y, z, grid) + centered_load(centered.z, x - 1, y, z, grid)) * inverse_spacing;
            const float wz              = (centered_load(centered.y, x + 1, y, z, grid) - centered_load(centered.y, x - 1, y, z, grid) - centered_load(centered.x, x, y + 1, z, grid) + centered_load(centered.x, x, y - 1, z, grid)) * inverse_spacing;
            vorticity.x[index]          = wx;
            vorticity.y[index]          = wy;
            vorticity.z[index]          = wz;
            magnitude[index]            = sqrtf(wx * wx + wy * wy + wz * wz + smooth_epsilon * smooth_epsilon);
        }

        __global__ void vorticity_normal_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* magnitude, const CenteredVectorView normal, float* normalizer) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (cell_mask[index] != 0u) {
                normal.x[index]   = 0.0F;
                normal.y[index]   = 0.0F;
                normal.z[index]   = 0.0F;
                normalizer[index] = smooth_epsilon;
                return;
            }
            const float inverse_spacing = 0.5F / grid.cell_size;
            const Vector gradient{
                (centered_load(magnitude, x + 1, y, z, grid) - centered_load(magnitude, x - 1, y, z, grid)) * inverse_spacing,
                (centered_load(magnitude, x, y + 1, z, grid) - centered_load(magnitude, x, y - 1, z, grid)) * inverse_spacing,
                (centered_load(magnitude, x, y, z + 1, grid) - centered_load(magnitude, x, y, z - 1, grid)) * inverse_spacing,
            };
            const float length = sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z + smooth_epsilon * smooth_epsilon);
            normal.x[index]    = gradient.x / length;
            normal.y[index]    = gradient.y / length;
            normal.z[index]    = gradient.z / length;
            normalizer[index]  = length;
        }

        __global__ void add_vorticity_force_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView normal, const float* confinement, const CenteredVectorView force) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            const float scale = confinement[0] * grid.cell_size;
            force.x[index] += scale * (normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index]);
            force.y[index] += scale * (normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index]);
            force.z[index] += scale * (normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index]);
        }

        __global__ void magnitude_tangent_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView vorticity_tangent, const float* magnitude, float* magnitude_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            magnitude_tangent[index] = cell_mask[index] == 0u ? (vorticity.x[index] * vorticity_tangent.x[index] + vorticity.y[index] * vorticity_tangent.y[index] + vorticity.z[index] * vorticity_tangent.z[index]) / magnitude[index] : 0.0F;
        }

        __global__ void normal_tangent_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* magnitude_tangent, const ConstCenteredVectorView normal, const float* normalizer, const CenteredVectorView normal_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (cell_mask[index] != 0u) {
                normal_tangent.x[index] = 0.0F;
                normal_tangent.y[index] = 0.0F;
                normal_tangent.z[index] = 0.0F;
                return;
            }
            const float inverse_spacing = 0.5F / grid.cell_size;
            const Vector gradient_tangent{
                (centered_load(magnitude_tangent, x + 1, y, z, grid) - centered_load(magnitude_tangent, x - 1, y, z, grid)) * inverse_spacing,
                (centered_load(magnitude_tangent, x, y + 1, z, grid) - centered_load(magnitude_tangent, x, y - 1, z, grid)) * inverse_spacing,
                (centered_load(magnitude_tangent, x, y, z + 1, grid) - centered_load(magnitude_tangent, x, y, z - 1, grid)) * inverse_spacing,
            };
            const float projection  = normal.x[index] * gradient_tangent.x + normal.y[index] * gradient_tangent.y + normal.z[index] * gradient_tangent.z;
            normal_tangent.x[index] = (gradient_tangent.x - normal.x[index] * projection) / normalizer[index];
            normal_tangent.y[index] = (gradient_tangent.y - normal.y[index] * projection) / normalizer[index];
            normal_tangent.z[index] = (gradient_tangent.z - normal.z[index] * projection) / normalizer[index];
        }

        __global__ void vorticity_force_tangent_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView vorticity_tangent, const ConstCenteredVectorView normal, const ConstCenteredVectorView normal_tangent, const float* confinement, const float* confinement_tangent, const CenteredVectorView force_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            const Vector cross{
                normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index],
                normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index],
                normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index],
            };
            const Vector cross_tangent{
                normal_tangent.y[index] * vorticity.z[index] + normal.y[index] * vorticity_tangent.z[index] - normal_tangent.z[index] * vorticity.y[index] - normal.z[index] * vorticity_tangent.y[index],
                normal_tangent.z[index] * vorticity.x[index] + normal.z[index] * vorticity_tangent.x[index] - normal_tangent.x[index] * vorticity.z[index] - normal.x[index] * vorticity_tangent.z[index],
                normal_tangent.x[index] * vorticity.y[index] + normal.x[index] * vorticity_tangent.y[index] - normal_tangent.y[index] * vorticity.x[index] - normal.y[index] * vorticity_tangent.x[index],
            };
            force_tangent.x[index] += grid.cell_size * (confinement_tangent[0] * cross.x + confinement[0] * cross_tangent.x);
            force_tangent.y[index] += grid.cell_size * (confinement_tangent[0] * cross.y + confinement[0] * cross_tangent.y);
            force_tangent.z[index] += grid.cell_size * (confinement_tangent[0] * cross.z + confinement[0] * cross_tangent.z);
        }

        __global__ void vorticity_force_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView normal, const float* confinement, const ConstCenteredVectorAdjointView force_adjoint, const CenteredVectorAdjointView vorticity_adjoint, const CenteredVectorAdjointView normal_adjoint, double* confinement_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            const Vector cross{
                normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index],
                normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index],
                normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index],
            };
            const double scale = static_cast<double>(confinement[0]) * grid.cell_size;
            atomicAdd(confinement_adjoint, grid.cell_size * (cross.x * force_adjoint.x[index] + cross.y * force_adjoint.y[index] + cross.z * force_adjoint.z[index]));
            normal_adjoint.x[index] += scale * (vorticity.y[index] * force_adjoint.z[index] - vorticity.z[index] * force_adjoint.y[index]);
            normal_adjoint.y[index] += scale * (vorticity.z[index] * force_adjoint.x[index] - vorticity.x[index] * force_adjoint.z[index]);
            normal_adjoint.z[index] += scale * (vorticity.x[index] * force_adjoint.y[index] - vorticity.y[index] * force_adjoint.x[index]);
            vorticity_adjoint.x[index] += scale * (force_adjoint.y[index] * normal.z[index] - force_adjoint.z[index] * normal.y[index]);
            vorticity_adjoint.y[index] += scale * (force_adjoint.z[index] * normal.x[index] - force_adjoint.x[index] * normal.z[index]);
            vorticity_adjoint.z[index] += scale * (force_adjoint.x[index] * normal.y[index] - force_adjoint.y[index] * normal.x[index]);
        }

        __global__ void normal_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView normal, const float* normalizer, const ConstCenteredVectorAdjointView normal_adjoint, double* magnitude_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double projection         = normal.x[index] * normal_adjoint.x[index] + normal.y[index] * normal_adjoint.y[index] + normal.z[index] * normal_adjoint.z[index];
            const double gradient_adjoint_x = (normal_adjoint.x[index] - normal.x[index] * projection) / normalizer[index];
            const double gradient_adjoint_y = (normal_adjoint.y[index] - normal.y[index] * projection) / normalizer[index];
            const double gradient_adjoint_z = (normal_adjoint.z[index] - normal.z[index] * projection) / normalizer[index];
            const double scale              = 0.5 / grid.cell_size;
            centered_scatter(magnitude_adjoint, x + 1, y, z, grid, scale * gradient_adjoint_x);
            centered_scatter(magnitude_adjoint, x - 1, y, z, grid, -scale * gradient_adjoint_x);
            centered_scatter(magnitude_adjoint, x, y + 1, z, grid, scale * gradient_adjoint_y);
            centered_scatter(magnitude_adjoint, x, y - 1, z, grid, -scale * gradient_adjoint_y);
            centered_scatter(magnitude_adjoint, x, y, z + 1, grid, scale * gradient_adjoint_z);
            centered_scatter(magnitude_adjoint, x, y, z - 1, grid, -scale * gradient_adjoint_z);
        }

        __global__ void magnitude_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const float* magnitude, const double* magnitude_adjoint, const CenteredVectorAdjointView vorticity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            const double scale = magnitude_adjoint[index] / magnitude[index];
            vorticity_adjoint.x[index] += scale * vorticity.x[index];
            vorticity_adjoint.y[index] += scale * vorticity.y[index];
            vorticity_adjoint.z[index] += scale * vorticity.z[index];
        }

        __global__ void curl_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorAdjointView vorticity_adjoint, const CenteredVectorAdjointView centered_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double scale = 0.5 / grid.cell_size;
            const double ax    = vorticity_adjoint.x[index] * scale;
            const double ay    = vorticity_adjoint.y[index] * scale;
            const double az    = vorticity_adjoint.z[index] * scale;
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

        __global__ void centered_velocity_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorAdjointView centered_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            atomicAdd(velocity_adjoint.x + index3(x, y, z, grid.nx + 1, grid.ny), 0.5F * centered_adjoint.x[index]);
            atomicAdd(velocity_adjoint.x + index3(x + 1, y, z, grid.nx + 1, grid.ny), 0.5F * centered_adjoint.x[index]);
            atomicAdd(velocity_adjoint.y + index3(x, y, z, grid.nx, grid.ny + 1), 0.5F * centered_adjoint.y[index]);
            atomicAdd(velocity_adjoint.y + index3(x, y + 1, z, grid.nx, grid.ny + 1), 0.5F * centered_adjoint.y[index]);
            atomicAdd(velocity_adjoint.z + index3(x, y, z, grid.nx, grid.ny), 0.5F * centered_adjoint.z[index]);
            atomicAdd(velocity_adjoint.z + index3(x, y, z + 1, grid.nx, grid.ny), 0.5F * centered_adjoint.z[index]);
        }
    } // namespace

    void buoyancy_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView density, const ConstScalarView temperature, const ConstCenteredVectorView external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const CenteredVectorView force) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), buoyancy_forward_kernel, grid, cell_mask, density.values, temperature.values, external_acceleration, ambient_temperature, density_buoyancy, temperature_buoyancy, force);
    }

    void buoyancy_jvp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView density, const ConstScalarView temperature, const ConstScalarView density_tangent, const ConstScalarView temperature_tangent, const ConstCenteredVectorView external_acceleration_tangent, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_temperature_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, const CenteredVectorView force_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), buoyancy_jvp_kernel, grid, cell_mask, density.values, temperature.values, density_tangent.values, temperature_tangent.values, external_acceleration_tangent, ambient_temperature, density_buoyancy, temperature_buoyancy, ambient_temperature_tangent, density_buoyancy_tangent, temperature_buoyancy_tangent, force_tangent);
    }

    void buoyancy_vjp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView density, const ConstScalarView temperature, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const ConstCenteredVectorAdjointView force_adjoint, const ScalarAdjointView density_adjoint, const ScalarAdjointView temperature_adjoint, const CenteredVectorAdjointView external_acceleration_adjoint, double* ambient_temperature_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), buoyancy_vjp_kernel, grid, cell_mask, density.values, temperature.values, ambient_temperature, density_buoyancy, temperature_buoyancy, force_adjoint, density_adjoint.values, temperature_adjoint.values, external_acceleration_adjoint, ambient_temperature_adjoint, density_buoyancy_adjoint, temperature_buoyancy_adjoint);
    }
    void vorticity_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const float* confinement, const VorticityView cache, const CenteredVectorView force) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), centered_velocity_kernel, grid, cell_mask, velocity, cache.centered_velocity);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), curl_magnitude_kernel, grid, cell_mask, ConstCenteredVectorView{cache.centered_velocity.x, cache.centered_velocity.y, cache.centered_velocity.z}, cache.vorticity, cache.magnitude.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), vorticity_normal_kernel, grid, cell_mask, cache.magnitude.values, cache.normal, cache.normalizer.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), add_vorticity_force_kernel, grid, cell_mask, ConstCenteredVectorView{cache.vorticity.x, cache.vorticity.y, cache.vorticity.z}, ConstCenteredVectorView{cache.normal.x, cache.normal.y, cache.normal.z}, confinement, force);
    }

    void vorticity_jvp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity_tangent, const float* confinement, const float* confinement_tangent, const ConstVorticityView cache, const CenteredVectorView force_tangent, const VorticityTangentScratch tangent_scratch) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), centered_velocity_kernel, grid, cell_mask, velocity_tangent, tangent_scratch.centered_velocity);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), curl_magnitude_kernel, grid, cell_mask, ConstCenteredVectorView{tangent_scratch.centered_velocity.x, tangent_scratch.centered_velocity.y, tangent_scratch.centered_velocity.z}, tangent_scratch.vorticity, tangent_scratch.magnitude.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), magnitude_tangent_kernel, grid, cell_mask, cache.vorticity, ConstCenteredVectorView{tangent_scratch.vorticity.x, tangent_scratch.vorticity.y, tangent_scratch.vorticity.z}, cache.magnitude.values, tangent_scratch.magnitude.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), normal_tangent_kernel, grid, cell_mask, tangent_scratch.magnitude.values, cache.normal, cache.normalizer.values, tangent_scratch.normal);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), vorticity_force_tangent_kernel, grid, cell_mask, cache.vorticity, ConstCenteredVectorView{tangent_scratch.vorticity.x, tangent_scratch.vorticity.y, tangent_scratch.vorticity.z}, cache.normal, ConstCenteredVectorView{tangent_scratch.normal.x, tangent_scratch.normal.y, tangent_scratch.normal.z}, confinement, confinement_tangent, force_tangent);
    }

    void vorticity_vjp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t* cell_mask, const float* confinement, const ConstVorticityView cache, const ConstCenteredVectorAdjointView force_adjoint, const StaggeredVectorAdjointView velocity_adjoint, double* confinement_adjoint, const VorticityAdjointScratch scratch) {
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
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), vorticity_force_reverse_kernel, grid, cell_mask, cache.vorticity, cache.normal, confinement, force_adjoint, scratch.vorticity, scratch.normal, confinement_adjoint);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), normal_reverse_kernel, grid, cell_mask, cache.normal, cache.normalizer.values, ConstCenteredVectorAdjointView{scratch.normal.x, scratch.normal.y, scratch.normal.z}, scratch.magnitude.values);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), magnitude_reverse_kernel, grid, cell_mask, cache.vorticity, cache.magnitude.values, scratch.magnitude.values, scratch.vorticity);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), curl_reverse_kernel, grid, cell_mask, ConstCenteredVectorAdjointView{scratch.vorticity.x, scratch.vorticity.y, scratch.vorticity.z}, scratch.centered_velocity);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), centered_velocity_reverse_kernel, grid, cell_mask, ConstCenteredVectorAdjointView{scratch.centered_velocity.x, scratch.centered_velocity.y, scratch.centered_velocity.z}, velocity_adjoint);
    }

} // namespace physica::fluids::gas::smoke::cuda_detail
