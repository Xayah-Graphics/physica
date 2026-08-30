#include "force-kernels.h"
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/span>

namespace physica::fluids::gas::operators::kernels {
    namespace {
        constexpr float smooth_epsilon = 1.0e-6F;

        __global__ void combine_forward_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> physical, const simulation::VectorView<const float> control, const simulation::VectorView<float> total) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            const float enabled = collider_ids[index] == 0u ? 1.0F : 0.0F;
            total.x[index]      = enabled * (physical.x[index] + control.x[index]);
            total.y[index]      = enabled * (physical.y[index] + control.y[index]);
            total.z[index]      = enabled * (physical.z[index] + control.z[index]);
        }

        __global__ void combine_vjp_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const double> total_adjoint, const simulation::VectorView<double> physical_adjoint, const simulation::VectorView<double> control_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            const double enabled      = collider_ids[index] == 0u ? 1.0 : 0.0;
            physical_adjoint.x[index] = enabled * total_adjoint.x[index];
            physical_adjoint.y[index] = enabled * total_adjoint.y[index];
            physical_adjoint.z[index] = enabled * total_adjoint.z[index];
            control_adjoint.x[index] += physical_adjoint.x[index];
            control_adjoint.y[index] += physical_adjoint.y[index];
            control_adjoint.z[index] += physical_adjoint.z[index];
        }

        struct ConstantParameter final {
            float value;
        };

        struct DynamicParameter final {
            const float* value;
        };

        struct NoParameterAdjoint final {};

        struct DynamicParameterAdjoint final {
            double* value;
        };

        __device__ float parameter(const ConstantParameter value) {
            return value.value;
        }

        __device__ float parameter(const DynamicParameter value) {
            return value.value[0];
        }

        __device__ void accumulate_parameter(NoParameterAdjoint, const double) {}

        __device__ void accumulate_parameter(const DynamicParameterAdjoint parameter_adjoint, const double value) {
            atomicAdd(parameter_adjoint.value, value);
        }

        __global__ void density_buoyancy_forward_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float buoyancy, const float* density, const simulation::VectorView<float> force) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            force.x[index] = 0.0F;
            force.y[index] = collider_ids[index] == 0u ? buoyancy * density[index] : 0.0F;
            force.z[index] = 0.0F;
        }

        __global__ void density_buoyancy_vjp_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float buoyancy, const simulation::VectorView<const double> force_adjoint, double* density_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < fluids::grid::device::cell_count(grid.grid) && collider_ids[index] == 0u) density_adjoint[index] += buoyancy * force_adjoint.y[index];
        }

        __global__ void buoyancy_forward_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* density, const float* temperature, const simulation::VectorView<const float> external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const simulation::VectorView<float> force) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            if (collider_ids[index] != 0u) {
                force.x[index] = 0.0F;
                force.y[index] = 0.0F;
                force.z[index] = 0.0F;
                return;
            }
            force.x[index] = external_acceleration.x[index];
            force.y[index] = external_acceleration.y[index] + density_buoyancy[0] * density[index] + temperature_buoyancy[0] * (temperature[index] - ambient_temperature[0]);
            force.z[index] = external_acceleration.z[index];
        }

        __global__ void buoyancy_jvp_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* density, const float* temperature, const float* density_tangent, const float* temperature_tangent, const simulation::VectorView<const float> external_tangent, const float* ambient, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, const simulation::VectorView<float> output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            if (collider_ids[index] != 0u) {
                output.x[index] = 0.0F;
                output.y[index] = 0.0F;
                output.z[index] = 0.0F;
                return;
            }
            output.x[index] = external_tangent.x[index];
            output.y[index] = external_tangent.y[index] + density_buoyancy[0] * density_tangent[index] + density_buoyancy_tangent[0] * density[index] + temperature_buoyancy[0] * (temperature_tangent[index] - ambient_tangent[0]) + temperature_buoyancy_tangent[0] * (temperature[index] - ambient[0]);
            output.z[index] = external_tangent.z[index];
        }

        __global__ void buoyancy_vjp_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* density, const float* temperature, const float* ambient, const float* density_buoyancy, const float* temperature_buoyancy, const simulation::VectorView<const double> force_adjoint, double* density_adjoint, double* temperature_adjoint, const simulation::VectorView<double> external_adjoint, double* ambient_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            external_adjoint.x[index] += force_adjoint.x[index];
            external_adjoint.y[index] += force_adjoint.y[index];
            external_adjoint.z[index] += force_adjoint.z[index];
            density_adjoint[index] += density_buoyancy[0] * force_adjoint.y[index];
            temperature_adjoint[index] += temperature_buoyancy[0] * force_adjoint.y[index];
            atomicAdd(ambient_adjoint, -temperature_buoyancy[0] * force_adjoint.y[index]);
            atomicAdd(density_buoyancy_adjoint, density[index] * force_adjoint.y[index]);
            atomicAdd(temperature_buoyancy_adjoint, (temperature[index] - ambient[0]) * force_adjoint.y[index]);
        }
        __device__ float centered_load(const float* values, int x, int y, int z, const device::Discretization grid) {
            x = ::cuda::std::clamp(x, 0, static_cast<int>(grid.grid.nx) - 1);
            y = ::cuda::std::clamp(y, 0, static_cast<int>(grid.grid.ny) - 1);
            z = ::cuda::std::clamp(z, 0, static_cast<int>(grid.grid.nz) - 1);
            return values[fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny)];
        }

        __device__ void centered_scatter(double* values, int x, int y, int z, const device::Discretization grid, const double adjoint) {
            x = ::cuda::std::clamp(x, 0, static_cast<int>(grid.grid.nx) - 1);
            y = ::cuda::std::clamp(y, 0, static_cast<int>(grid.grid.ny) - 1);
            z = ::cuda::std::clamp(z, 0, static_cast<int>(grid.grid.nz) - 1);
            atomicAdd(values + fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny), adjoint);
        }

        __global__ void centered_velocity_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const simulation::VectorView<float> centered) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            if (collider_ids[index] != 0u) {
                centered.x[index] = 0.0F;
                centered.y[index] = 0.0F;
                centered.z[index] = 0.0F;
                return;
            }
            centered.x[index] = 0.5F * (velocity.x[fluids::grid::device::index3(x, y, z, grid.grid.nx + 1, grid.grid.ny)] + velocity.x[fluids::grid::device::index3(x + 1, y, z, grid.grid.nx + 1, grid.grid.ny)]);
            centered.y[index] = 0.5F * (velocity.y[fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny + 1)] + velocity.y[fluids::grid::device::index3(x, y + 1, z, grid.grid.nx, grid.grid.ny + 1)]);
            centered.z[index] = 0.5F * (velocity.z[fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny)] + velocity.z[fluids::grid::device::index3(x, y, z + 1, grid.grid.nx, grid.grid.ny)]);
        }

        __global__ void curl_magnitude_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> centered, const simulation::VectorView<float> vorticity, float* magnitude) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            if (collider_ids[index] != 0u) {
                vorticity.x[index] = 0.0F;
                vorticity.y[index] = 0.0F;
                vorticity.z[index] = 0.0F;
                magnitude[index]   = smooth_epsilon;
                return;
            }
            const float inverse_spacing = 0.5F / grid.grid.cell_size;
            const float wx              = (centered_load(centered.z, x, y + 1, z, grid) - centered_load(centered.z, x, y - 1, z, grid) - centered_load(centered.y, x, y, z + 1, grid) + centered_load(centered.y, x, y, z - 1, grid)) * inverse_spacing;
            const float wy              = (centered_load(centered.x, x, y, z + 1, grid) - centered_load(centered.x, x, y, z - 1, grid) - centered_load(centered.z, x + 1, y, z, grid) + centered_load(centered.z, x - 1, y, z, grid)) * inverse_spacing;
            const float wz              = (centered_load(centered.y, x + 1, y, z, grid) - centered_load(centered.y, x - 1, y, z, grid) - centered_load(centered.x, x, y + 1, z, grid) + centered_load(centered.x, x, y - 1, z, grid)) * inverse_spacing;
            vorticity.x[index]          = wx;
            vorticity.y[index]          = wy;
            vorticity.z[index]          = wz;
            magnitude[index]            = ::cuda::std::sqrt(wx * wx + wy * wy + wz * wz + smooth_epsilon * smooth_epsilon);
        }

        __global__ void vorticity_normal_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* magnitude, const simulation::VectorView<float> normal, float* normalizer) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            if (collider_ids[index] != 0u) {
                normal.x[index]   = 0.0F;
                normal.y[index]   = 0.0F;
                normal.z[index]   = 0.0F;
                normalizer[index] = smooth_epsilon;
                return;
            }
            const float inverse_spacing = 0.5F / grid.grid.cell_size;
            const Vector3<float> gradient{
                (centered_load(magnitude, x + 1, y, z, grid) - centered_load(magnitude, x - 1, y, z, grid)) * inverse_spacing,
                (centered_load(magnitude, x, y + 1, z, grid) - centered_load(magnitude, x, y - 1, z, grid)) * inverse_spacing,
                (centered_load(magnitude, x, y, z + 1, grid) - centered_load(magnitude, x, y, z - 1, grid)) * inverse_spacing,
            };
            const float length = ::cuda::std::sqrt(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z + smooth_epsilon * smooth_epsilon);
            normal.x[index]    = gradient.x / length;
            normal.y[index]    = gradient.y / length;
            normal.z[index]    = gradient.z / length;
            normalizer[index]  = length;
        }

        template <class Parameter>
        __global__ void add_vorticity_force_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> vorticity, const simulation::VectorView<const float> normal, const Parameter confinement, const simulation::VectorView<float> force) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            const float scale = parameter(confinement) * grid.grid.cell_size;
            force.x[index] += scale * (normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index]);
            force.y[index] += scale * (normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index]);
            force.z[index] += scale * (normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index]);
        }

        __global__ void magnitude_tangent_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> vorticity, const simulation::VectorView<const float> vorticity_tangent, const float* magnitude, float* magnitude_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            magnitude_tangent[index] = collider_ids[index] == 0u ? (vorticity.x[index] * vorticity_tangent.x[index] + vorticity.y[index] * vorticity_tangent.y[index] + vorticity.z[index] * vorticity_tangent.z[index]) / magnitude[index] : 0.0F;
        }

        __global__ void normal_tangent_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* magnitude_tangent, const simulation::VectorView<const float> normal, const float* normalizer, const simulation::VectorView<float> normal_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            if (collider_ids[index] != 0u) {
                normal_tangent.x[index] = 0.0F;
                normal_tangent.y[index] = 0.0F;
                normal_tangent.z[index] = 0.0F;
                return;
            }
            const float inverse_spacing = 0.5F / grid.grid.cell_size;
            const Vector3<float> gradient_tangent{
                (centered_load(magnitude_tangent, x + 1, y, z, grid) - centered_load(magnitude_tangent, x - 1, y, z, grid)) * inverse_spacing,
                (centered_load(magnitude_tangent, x, y + 1, z, grid) - centered_load(magnitude_tangent, x, y - 1, z, grid)) * inverse_spacing,
                (centered_load(magnitude_tangent, x, y, z + 1, grid) - centered_load(magnitude_tangent, x, y, z - 1, grid)) * inverse_spacing,
            };
            const float projection  = normal.x[index] * gradient_tangent.x + normal.y[index] * gradient_tangent.y + normal.z[index] * gradient_tangent.z;
            normal_tangent.x[index] = (gradient_tangent.x - normal.x[index] * projection) / normalizer[index];
            normal_tangent.y[index] = (gradient_tangent.y - normal.y[index] * projection) / normalizer[index];
            normal_tangent.z[index] = (gradient_tangent.z - normal.z[index] * projection) / normalizer[index];
        }

        template <class Parameter, class ParameterTangent>
        __global__ void vorticity_force_tangent_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> vorticity, const simulation::VectorView<const float> vorticity_tangent, const simulation::VectorView<const float> normal, const simulation::VectorView<const float> normal_tangent, const Parameter confinement, const ParameterTangent confinement_tangent, const simulation::VectorView<float> force_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            const Vector3<float> cross{
                normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index],
                normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index],
                normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index],
            };
            const Vector3<float> cross_tangent{
                normal_tangent.y[index] * vorticity.z[index] + normal.y[index] * vorticity_tangent.z[index] - normal_tangent.z[index] * vorticity.y[index] - normal.z[index] * vorticity_tangent.y[index],
                normal_tangent.z[index] * vorticity.x[index] + normal.z[index] * vorticity_tangent.x[index] - normal_tangent.x[index] * vorticity.z[index] - normal.x[index] * vorticity_tangent.z[index],
                normal_tangent.x[index] * vorticity.y[index] + normal.x[index] * vorticity_tangent.y[index] - normal_tangent.y[index] * vorticity.x[index] - normal.y[index] * vorticity_tangent.x[index],
            };
            force_tangent.x[index] += grid.grid.cell_size * (parameter(confinement_tangent) * cross.x + parameter(confinement) * cross_tangent.x);
            force_tangent.y[index] += grid.grid.cell_size * (parameter(confinement_tangent) * cross.y + parameter(confinement) * cross_tangent.y);
            force_tangent.z[index] += grid.grid.cell_size * (parameter(confinement_tangent) * cross.z + parameter(confinement) * cross_tangent.z);
        }

        template <class Parameter, class ParameterAdjoint>
        __global__ void vorticity_force_reverse_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> vorticity, const simulation::VectorView<const float> normal, const Parameter confinement, const simulation::VectorView<const double> force_adjoint, const simulation::VectorView<double> vorticity_adjoint, const simulation::VectorView<double> normal_adjoint, const ParameterAdjoint confinement_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            const Vector3<float> cross{
                normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index],
                normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index],
                normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index],
            };
            const double scale = static_cast<double>(parameter(confinement)) * grid.grid.cell_size;
            accumulate_parameter(confinement_adjoint, grid.grid.cell_size * (cross.x * force_adjoint.x[index] + cross.y * force_adjoint.y[index] + cross.z * force_adjoint.z[index]));
            normal_adjoint.x[index] += scale * (vorticity.y[index] * force_adjoint.z[index] - vorticity.z[index] * force_adjoint.y[index]);
            normal_adjoint.y[index] += scale * (vorticity.z[index] * force_adjoint.x[index] - vorticity.x[index] * force_adjoint.z[index]);
            normal_adjoint.z[index] += scale * (vorticity.x[index] * force_adjoint.y[index] - vorticity.y[index] * force_adjoint.x[index]);
            vorticity_adjoint.x[index] += scale * (force_adjoint.y[index] * normal.z[index] - force_adjoint.z[index] * normal.y[index]);
            vorticity_adjoint.y[index] += scale * (force_adjoint.z[index] * normal.x[index] - force_adjoint.x[index] * normal.z[index]);
            vorticity_adjoint.z[index] += scale * (force_adjoint.x[index] * normal.y[index] - force_adjoint.y[index] * normal.x[index]);
        }

        __global__ void normal_reverse_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> normal, const float* normalizer, const simulation::VectorView<const double> normal_adjoint, double* magnitude_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            const double projection         = normal.x[index] * normal_adjoint.x[index] + normal.y[index] * normal_adjoint.y[index] + normal.z[index] * normal_adjoint.z[index];
            const double gradient_adjoint_x = (normal_adjoint.x[index] - normal.x[index] * projection) / normalizer[index];
            const double gradient_adjoint_y = (normal_adjoint.y[index] - normal.y[index] * projection) / normalizer[index];
            const double gradient_adjoint_z = (normal_adjoint.z[index] - normal.z[index] * projection) / normalizer[index];
            const double scale              = 0.5 / grid.grid.cell_size;
            centered_scatter(magnitude_adjoint, x + 1, y, z, grid, scale * gradient_adjoint_x);
            centered_scatter(magnitude_adjoint, x - 1, y, z, grid, -scale * gradient_adjoint_x);
            centered_scatter(magnitude_adjoint, x, y + 1, z, grid, scale * gradient_adjoint_y);
            centered_scatter(magnitude_adjoint, x, y - 1, z, grid, -scale * gradient_adjoint_y);
            centered_scatter(magnitude_adjoint, x, y, z + 1, grid, scale * gradient_adjoint_z);
            centered_scatter(magnitude_adjoint, x, y, z - 1, grid, -scale * gradient_adjoint_z);
        }

        __global__ void magnitude_reverse_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> vorticity, const float* magnitude, const double* magnitude_adjoint, const simulation::VectorView<double> vorticity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            const double scale = magnitude_adjoint[index] / magnitude[index];
            vorticity_adjoint.x[index] += scale * vorticity.x[index];
            vorticity_adjoint.y[index] += scale * vorticity.y[index];
            vorticity_adjoint.z[index] += scale * vorticity.z[index];
        }

        __global__ void curl_reverse_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const double> vorticity_adjoint, const simulation::VectorView<double> centered_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            const double scale = 0.5 / grid.grid.cell_size;
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

        __global__ void centered_velocity_reverse_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const double> centered_adjoint, const simulation::VectorView<double> velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            atomicAdd(velocity_adjoint.x + fluids::grid::device::index3(x, y, z, grid.grid.nx + 1, grid.grid.ny), 0.5F * centered_adjoint.x[index]);
            atomicAdd(velocity_adjoint.x + fluids::grid::device::index3(x + 1, y, z, grid.grid.nx + 1, grid.grid.ny), 0.5F * centered_adjoint.x[index]);
            atomicAdd(velocity_adjoint.y + fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny + 1), 0.5F * centered_adjoint.y[index]);
            atomicAdd(velocity_adjoint.y + fluids::grid::device::index3(x, y + 1, z, grid.grid.nx, grid.grid.ny + 1), 0.5F * centered_adjoint.y[index]);
            atomicAdd(velocity_adjoint.z + fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny), 0.5F * centered_adjoint.z[index]);
            atomicAdd(velocity_adjoint.z + fluids::grid::device::index3(x, y, z + 1, grid.grid.nx, grid.grid.ny), 0.5F * centered_adjoint.z[index]);
        }

        void vorticity_forward_fields(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const VorticityView cache) {
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), centered_velocity_kernel, grid, collider_ids, velocity, cache.centered_velocity);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), curl_magnitude_kernel, grid, collider_ids, simulation::VectorView<const float>{cache.centered_velocity.x, cache.centered_velocity.y, cache.centered_velocity.z}, cache.vorticity, cache.magnitude.values);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), vorticity_normal_kernel, grid, collider_ids, cache.magnitude.values, cache.normal, cache.normalizer.values);
        }

        void vorticity_tangent_fields(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity_tangent, const ConstVorticityView cache, const VorticityTangentScratch tangent) {
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), centered_velocity_kernel, grid, collider_ids, velocity_tangent, tangent.centered_velocity);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), curl_magnitude_kernel, grid, collider_ids, simulation::VectorView<const float>{tangent.centered_velocity.x, tangent.centered_velocity.y, tangent.centered_velocity.z}, tangent.vorticity, tangent.magnitude.values);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), magnitude_tangent_kernel, grid, collider_ids, cache.vorticity, simulation::VectorView<const float>{tangent.vorticity.x, tangent.vorticity.y, tangent.vorticity.z}, cache.magnitude.values, tangent.magnitude.values);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), normal_tangent_kernel, grid, collider_ids, tangent.magnitude.values, cache.normal, cache.normalizer.values, tangent.normal);
        }

        void clear_vorticity_adjoint(const ::cuda::stream_ref stream, const device::Discretization grid, const VorticityAdjointScratch scratch) {
            const std::size_t count = static_cast<std::size_t>(fluids::grid::device::cell_count(grid.grid));
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
        }

        void vorticity_reverse_fields(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const ConstVorticityView cache, const VorticityAdjointScratch scratch, const simulation::VectorView<double> velocity_adjoint) {
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), normal_reverse_kernel, grid, collider_ids, cache.normal, cache.normalizer.values, simulation::VectorView<const double>{scratch.normal.x, scratch.normal.y, scratch.normal.z}, scratch.magnitude.values);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), magnitude_reverse_kernel, grid, collider_ids, cache.vorticity, cache.magnitude.values, scratch.magnitude.values, scratch.vorticity);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), curl_reverse_kernel, grid, collider_ids, simulation::VectorView<const double>{scratch.vorticity.x, scratch.vorticity.y, scratch.vorticity.z}, scratch.centered_velocity);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), centered_velocity_reverse_kernel, grid, collider_ids, simulation::VectorView<const double>{scratch.centered_velocity.x, scratch.centered_velocity.y, scratch.centered_velocity.z}, velocity_adjoint);
        }
    } // namespace

    void buoyancy_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::ScalarView<const float> density, const simulation::ScalarView<const float> temperature, const simulation::VectorView<const float> external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const simulation::VectorView<float> force) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), buoyancy_forward_kernel, grid, collider_ids, density.values, temperature.values, external_acceleration, ambient_temperature, density_buoyancy, temperature_buoyancy, force);
    }

    void density_buoyancy_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const float buoyancy, const simulation::ScalarView<const float> density, const simulation::VectorView<float> force) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), density_buoyancy_forward_kernel, grid, collider_ids, buoyancy, density.values, force);
    }

    void density_buoyancy_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const float buoyancy, const simulation::VectorView<const double> force_adjoint, const simulation::ScalarView<double> density_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), density_buoyancy_vjp_kernel, grid, collider_ids, buoyancy, force_adjoint, density_adjoint.values);
    }

    void buoyancy_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::ScalarView<const float> density, const simulation::ScalarView<const float> temperature, const simulation::ScalarView<const float> density_tangent, const simulation::ScalarView<const float> temperature_tangent, const simulation::VectorView<const float> external_acceleration_tangent, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_temperature_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, const simulation::VectorView<float> force_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), buoyancy_jvp_kernel, grid, collider_ids, density.values, temperature.values, density_tangent.values, temperature_tangent.values, external_acceleration_tangent, ambient_temperature, density_buoyancy, temperature_buoyancy, ambient_temperature_tangent, density_buoyancy_tangent, temperature_buoyancy_tangent, force_tangent);
    }

    void buoyancy_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::ScalarView<const float> density, const simulation::ScalarView<const float> temperature, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const simulation::VectorView<const double> force_adjoint, const simulation::ScalarView<double> density_adjoint, const simulation::ScalarView<double> temperature_adjoint, const simulation::VectorView<double> external_acceleration_adjoint, double* ambient_temperature_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), buoyancy_vjp_kernel, grid, collider_ids, density.values, temperature.values, ambient_temperature, density_buoyancy, temperature_buoyancy, force_adjoint, density_adjoint.values, temperature_adjoint.values, external_acceleration_adjoint, ambient_temperature_adjoint, density_buoyancy_adjoint, temperature_buoyancy_adjoint);
    }
    void vorticity_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const float* confinement, const VorticityView cache, const simulation::VectorView<float> force) {
        vorticity_forward_fields(stream, grid, collider_ids, velocity, cache);
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), add_vorticity_force_kernel<DynamicParameter>, grid, collider_ids, simulation::VectorView<const float>{cache.vorticity.x, cache.vorticity.y, cache.vorticity.z}, simulation::VectorView<const float>{cache.normal.x, cache.normal.y, cache.normal.z}, DynamicParameter{confinement}, force);
    }

    void vorticity_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const float confinement, const VorticityView cache, const simulation::VectorView<float> force) {
        vorticity_forward_fields(stream, grid, collider_ids, velocity, cache);
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), add_vorticity_force_kernel<ConstantParameter>, grid, collider_ids, simulation::VectorView<const float>{cache.vorticity.x, cache.vorticity.y, cache.vorticity.z}, simulation::VectorView<const float>{cache.normal.x, cache.normal.y, cache.normal.z}, ConstantParameter{confinement}, force);
    }

    void vorticity_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity_tangent, const float* confinement, const float* confinement_tangent, const ConstVorticityView cache, const simulation::VectorView<float> force_tangent, const VorticityTangentScratch tangent_scratch) {
        vorticity_tangent_fields(stream, grid, collider_ids, velocity_tangent, cache, tangent_scratch);
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), vorticity_force_tangent_kernel<DynamicParameter, DynamicParameter>, grid, collider_ids, cache.vorticity, simulation::VectorView<const float>{tangent_scratch.vorticity.x, tangent_scratch.vorticity.y, tangent_scratch.vorticity.z}, cache.normal, simulation::VectorView<const float>{tangent_scratch.normal.x, tangent_scratch.normal.y, tangent_scratch.normal.z}, DynamicParameter{confinement}, DynamicParameter{confinement_tangent}, force_tangent);
    }

    void vorticity_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity_tangent, const float confinement, const ConstVorticityView cache, const simulation::VectorView<float> force_tangent, const VorticityTangentScratch tangent_scratch) {
        vorticity_tangent_fields(stream, grid, collider_ids, velocity_tangent, cache, tangent_scratch);
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), vorticity_force_tangent_kernel<ConstantParameter, ConstantParameter>, grid, collider_ids, cache.vorticity, simulation::VectorView<const float>{tangent_scratch.vorticity.x, tangent_scratch.vorticity.y, tangent_scratch.vorticity.z}, cache.normal, simulation::VectorView<const float>{tangent_scratch.normal.x, tangent_scratch.normal.y, tangent_scratch.normal.z}, ConstantParameter{confinement}, ConstantParameter{0.0F}, force_tangent);
    }

    void vorticity_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const float* confinement, const ConstVorticityView cache, const simulation::VectorView<const double> force_adjoint, const simulation::VectorView<double> velocity_adjoint, double* confinement_adjoint, const VorticityAdjointScratch scratch) {
        clear_vorticity_adjoint(stream, grid, scratch);
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), vorticity_force_reverse_kernel<DynamicParameter, DynamicParameterAdjoint>, grid, collider_ids, cache.vorticity, cache.normal, DynamicParameter{confinement}, force_adjoint, scratch.vorticity, scratch.normal, DynamicParameterAdjoint{confinement_adjoint});
        vorticity_reverse_fields(stream, grid, collider_ids, cache, scratch, velocity_adjoint);
    }

    void vorticity_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const float confinement, const ConstVorticityView cache, const simulation::VectorView<const double> force_adjoint, const simulation::VectorView<double> velocity_adjoint, const VorticityAdjointScratch scratch) {
        clear_vorticity_adjoint(stream, grid, scratch);
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), vorticity_force_reverse_kernel<ConstantParameter, NoParameterAdjoint>, grid, collider_ids, cache.vorticity, cache.normal, ConstantParameter{confinement}, force_adjoint, scratch.vorticity, scratch.normal, NoParameterAdjoint{});
        vorticity_reverse_fields(stream, grid, collider_ids, cache, scratch, velocity_adjoint);
    }

    void combine_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> physical, const simulation::VectorView<const float> control, const simulation::VectorView<float> total) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), combine_forward_kernel, grid, collider_ids, physical, control, total);
    }

    void combine_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const double> total_adjoint, const simulation::VectorView<double> physical_adjoint, const simulation::VectorView<double> control_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), combine_vjp_kernel, grid, collider_ids, total_adjoint, physical_adjoint, control_adjoint);
    }

} // namespace physica::fluids::gas::operators::kernels
