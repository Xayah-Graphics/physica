#include "simulation_kernels.h"
#include <cuda/cmath>
#include <cuda/std/cmath>
#include <cuda/std/numbers>
#include <cuda_runtime.h>

namespace physica::examples::smoke::simulation_cuda {
    namespace {
        constexpr std::uint32_t block_size = 256u;
        constexpr float two_pi             = 2.0F * ::cuda::std::numbers::pi_v<float>;

        __device__ float gaussian(const float x, const float y, const float z, const Vector3<float> center, const float radius) {
            const float dx = x - center.x;
            const float dy = y - center.y;
            const float dz = z - center.z;
            return ::cuda::std::exp(-0.5F * (dx * dx + dy * dy + dz * dz) / (radius * radius));
        }

        __global__ void write_control_kernel(const Grid grid, const std::uint64_t step, const float pulse_period, const Vector3<float> left_center, const Vector3<float> right_center, const float source_radius, const float density_source_rate, const float temperature_source_rate, const Vector3<float> left_acceleration, const Vector3<float> right_acceleration, float* const density_source, float* const temperature_source, float* const acceleration_x, float* const acceleration_y, float* const acceleration_z) {
            const std::uint32_t cell       = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
            if (cell >= cell_count) return;
            const std::uint32_t x_index = cell % grid.nx;
            const std::uint32_t y_index = cell / grid.nx % grid.ny;
            const std::uint32_t z_index = cell / (grid.nx * grid.ny);
            const float x               = (static_cast<float>(x_index) + 0.5F) * grid.cell_size;
            const float y               = (static_cast<float>(y_index) + 0.5F) * grid.cell_size;
            const float z               = (static_cast<float>(z_index) + 0.5F) * grid.cell_size;
            const float phase           = two_pi * static_cast<float>(step) * grid.time_step / pulse_period;
            const float left_weight     = (0.75F + 0.25F * ::cuda::std::sin(phase)) * gaussian(x, y, z, left_center, source_radius);
            const float right_weight    = (0.75F + 0.25F * ::cuda::std::sin(phase + 0.5F * two_pi)) * gaussian(x, y, z, right_center, source_radius);
            const float source_weight   = left_weight + right_weight;
            density_source[cell]        = density_source_rate * source_weight;
            temperature_source[cell]    = temperature_source_rate * source_weight;
            acceleration_x[cell]        = left_weight * left_acceleration.x + right_weight * right_acceleration.x;
            acceleration_y[cell]        = left_weight * left_acceleration.y + right_weight * right_acceleration.y;
            acceleration_z[cell]        = left_weight * left_acceleration.z + right_weight * right_acceleration.z;
        }
    } // namespace

    void write_control(const ::cuda::stream_ref stream, const Grid grid, const std::uint64_t step, const float pulse_period, const Vector3<float> left_center, const Vector3<float> right_center, const float source_radius, const float density_source_rate, const float temperature_source_rate, const Vector3<float> left_acceleration, const Vector3<float> right_acceleration, float* const density_source, float* const temperature_source, float* const acceleration_x, float* const acceleration_y, float* const acceleration_z) {
        const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
        write_control_kernel<<<::cuda::ceil_div(cell_count, block_size), block_size, 0u, stream.get()>>>(grid, step, pulse_period, left_center, right_center, source_radius, density_source_rate, temperature_source_rate, left_acceleration, right_acceleration, density_source, temperature_source, acceleration_x, acceleration_y, acceleration_z);
    }
} // namespace physica::examples::smoke::simulation_cuda
