#ifndef PHYSICA_EXAMPLES_DEFORMABLES_CLOTH_SIMULATION_KERNELS_H
#define PHYSICA_EXAMPLES_DEFORMABLES_CLOTH_SIMULATION_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::examples::cloth::simulation_cuda {
    struct Grid final {
        std::uint32_t rows;
        std::uint32_t columns;
        float width;
        float height;
    };

    struct Wind final {
        float speed;
        float gust_strength;
        float gust_frequency;
        float air_density;
        float drag_coefficient;
        float skin_drag_coefficient;
        float ramp_duration;
    };

    void write_control(::cuda::stream_ref stream, Grid grid, std::uint64_t step, float time_step, Wind wind, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, float* force_x, float* force_y, float* force_z);
} // namespace physica::examples::cloth::simulation_cuda

#endif
