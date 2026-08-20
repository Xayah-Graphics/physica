#include "simulation_kernels.h"
#include <cuda/launch>
#include <math.h>

namespace physica::examples::cloth::simulation_cuda {
    namespace {
        constexpr std::uint32_t block_size = 256u;
        constexpr float two_pi = 6.28318530717958647692F;
        constexpr float one_third_pi = 1.04719755119659774615F;

        struct WindVelocity final {
            float x;
            float z;
        };

        __device__ WindVelocity wind_velocity(const float time, const float normalized_x, const float normalized_y, const Wind wind) {
            const float ramp_coordinate = fminf(time / wind.ramp_duration, 1.0F);
            const float ramp = ramp_coordinate * ramp_coordinate * (3.0F - 2.0F * ramp_coordinate);
            const float primary_gust = sinf(two_pi * (wind.gust_frequency * time - 0.85F * normalized_x + 0.12F * normalized_y));
            const float secondary_gust = sinf(two_pi * (1.73F * wind.gust_frequency * time - 1.70F * normalized_x - 0.28F * normalized_y) + one_third_pi);
            return {
                .x = ramp * wind.speed * (1.0F + 0.20F * wind.gust_strength * primary_gust),
                .z = ramp * wind.speed * wind.gust_strength * (0.72F * primary_gust + 0.28F * secondary_gust),
            };
        }

        __global__ void write_control_kernel(const Grid grid, const std::uint64_t step, const float time_step, const Wind wind, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, float* force_x, float* force_y, float* force_z) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= grid.rows * grid.columns) return;
            const std::uint32_t row = particle / grid.columns;
            const std::uint32_t column = particle % grid.columns;
            if (column == 0u) {
                force_x[particle] = 0.0F;
                force_y[particle] = 0.0F;
                force_z[particle] = 0.0F;
                return;
            }

            const std::uint32_t left = row * grid.columns + column - 1u;
            const std::uint32_t right = row * grid.columns + (column + 1u < grid.columns ? column + 1u : grid.columns - 1u);
            const std::uint32_t top = (row == 0u ? 0u : row - 1u) * grid.columns + column;
            const std::uint32_t bottom = (row + 1u < grid.rows ? row + 1u : grid.rows - 1u) * grid.columns + column;
            const float tangent_x_x = position_x[right] - position_x[left];
            const float tangent_x_y = position_y[right] - position_y[left];
            const float tangent_x_z = position_z[right] - position_z[left];
            const float tangent_y_x = position_x[bottom] - position_x[top];
            const float tangent_y_y = position_y[bottom] - position_y[top];
            const float tangent_y_z = position_z[bottom] - position_z[top];
            float normal_x = tangent_x_y * tangent_y_z - tangent_x_z * tangent_y_y;
            float normal_y = tangent_x_z * tangent_y_x - tangent_x_x * tangent_y_z;
            float normal_z = tangent_x_x * tangent_y_y - tangent_x_y * tangent_y_x;
            const float inverse_normal_length = rsqrtf(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
            normal_x *= inverse_normal_length;
            normal_y *= inverse_normal_length;
            normal_z *= inverse_normal_length;

            const float normalized_x = static_cast<float>(column) / static_cast<float>(grid.columns - 1u);
            const float normalized_y = static_cast<float>(row) / static_cast<float>(grid.rows - 1u);
            const WindVelocity local_wind = wind_velocity(static_cast<float>(step) * time_step, normalized_x, normalized_y, wind);
            const float relative_x = local_wind.x - velocity_x[particle];
            const float relative_y = -velocity_y[particle];
            const float relative_z = local_wind.z - velocity_z[particle];
            const float normal_speed = relative_x * normal_x + relative_y * normal_y + relative_z * normal_z;
            const float tangent_x = relative_x - normal_speed * normal_x;
            const float tangent_y = relative_y - normal_speed * normal_y;
            const float tangent_z = relative_z - normal_speed * normal_z;
            const float tangent_speed = sqrtf(tangent_x * tangent_x + tangent_y * tangent_y + tangent_z * tangent_z);
            const float edge_weight_x = column == grid.columns - 1u ? 0.5F : 1.0F;
            const float edge_weight_y = row == 0u || row == grid.rows - 1u ? 0.5F : 1.0F;
            const float particle_area = grid.width * grid.height / static_cast<float>((grid.columns - 1u) * (grid.rows - 1u));
            const float area_scale = 0.5F * wind.air_density * particle_area * edge_weight_x * edge_weight_y;
            const float normal_force = area_scale * wind.drag_coefficient * normal_speed * fabsf(normal_speed);
            const float tangent_force = area_scale * wind.skin_drag_coefficient * tangent_speed;
            force_x[particle] = normal_force * normal_x + tangent_force * tangent_x;
            force_y[particle] = normal_force * normal_y + tangent_force * tangent_y;
            force_z[particle] = normal_force * normal_z + tangent_force * tangent_z;
        }
    } // namespace

    void write_control(const ::cuda::stream_ref stream, const Grid grid, const std::uint64_t step, const float time_step, const Wind wind, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, float* force_x, float* force_y, float* force_z) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(grid.rows * grid.columns), write_control_kernel, grid, step, time_step, wind, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, force_x, force_y, force_z);
    }
} // namespace physica::examples::cloth::simulation_cuda
