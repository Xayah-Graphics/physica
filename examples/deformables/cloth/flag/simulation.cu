#include "simulation_kernels.h"
#include <cuda/launch>
#include <cuda/std/cmath>
#include <cuda/std/numbers>
#include <cuda_runtime_api.h>

namespace physica::examples::cloth_flag::simulation_cuda {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void aerodynamic_forces_kernel(const std::uint32_t triangle_count, const std::uint64_t step, const float time_step, const Wind wind, const std::uint32_t* first, const std::uint32_t* second, const std::uint32_t* third, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, float* force_x, float* force_y, float* force_z) {
            const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
            if (triangle >= triangle_count) return;
            const std::uint32_t a = first[triangle];
            const std::uint32_t b = second[triangle];
            const std::uint32_t c = third[triangle];
            const float first_x   = position_x[b] - position_x[a];
            const float first_y   = position_y[b] - position_y[a];
            const float first_z   = position_z[b] - position_z[a];
            const float second_x  = position_x[c] - position_x[a];
            const float second_y  = position_y[c] - position_y[a];
            const float second_z  = position_z[c] - position_z[a];
            const float area_x    = 0.5F * (first_y * second_z - first_z * second_y);
            const float area_y    = 0.5F * (first_z * second_x - first_x * second_z);
            const float area_z    = 0.5F * (first_x * second_y - first_y * second_x);
            const float area      = ::cuda::std::sqrt(area_x * area_x + area_y * area_y + area_z * area_z);
            const float normal_x  = area_x / area;
            const float normal_y  = area_y / area;
            const float normal_z  = area_z / area;
            const float time      = static_cast<float>(step) * time_step;
            const float phase     = position_x[a] * 1.37F + position_y[a] * 0.71F + position_z[a] * 0.43F;
            const float gust      = 1.0F + wind.gust_strength * ::cuda::std::sin(2.0F * ::cuda::std::numbers::pi_v<float> * wind.gust_frequency * time + phase);
            const float relative_x = gust * wind.x - (velocity_x[a] + velocity_x[b] + velocity_x[c]) / 3.0F;
            const float relative_y = gust * wind.y - (velocity_y[a] + velocity_y[b] + velocity_y[c]) / 3.0F;
            const float relative_z = gust * wind.z - (velocity_z[a] + velocity_z[b] + velocity_z[c]) / 3.0F;
            const float normal_speed = relative_x * normal_x + relative_y * normal_y + relative_z * normal_z;
            const float pressure     = 0.5F * wind.air_density * wind.drag_coefficient * normal_speed * ::cuda::std::abs(normal_speed) * area / 3.0F;
            const float x            = pressure * normal_x;
            const float y            = pressure * normal_y;
            const float z            = pressure * normal_z;
            atomicAdd(force_x + a, x);
            atomicAdd(force_y + a, y);
            atomicAdd(force_z + a, z);
            atomicAdd(force_x + b, x);
            atomicAdd(force_y + b, y);
            atomicAdd(force_z + b, z);
            atomicAdd(force_x + c, x);
            atomicAdd(force_y + c, y);
            atomicAdd(force_z + c, z);
        }
    } // namespace

    void write_aerodynamic_forces(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t triangle_count, const std::uint64_t step, const float time_step, const Wind wind, const std::uint32_t* first, const std::uint32_t* second, const std::uint32_t* third, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, float* force_x, float* force_y, float* force_z) {
        cudaMemsetAsync(force_x, 0, static_cast<std::size_t>(particle_count) * sizeof(float), stream.get());
        cudaMemsetAsync(force_y, 0, static_cast<std::size_t>(particle_count) * sizeof(float), stream.get());
        cudaMemsetAsync(force_z, 0, static_cast<std::size_t>(particle_count) * sizeof(float), stream.get());
        ::cuda::launch(stream, ::cuda::distribute<block_size>(triangle_count), aerodynamic_forces_kernel, triangle_count, step, time_step, wind, first, second, third, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, force_x, force_y, force_z);
    }
} // namespace physica::examples::cloth_flag::simulation_cuda
