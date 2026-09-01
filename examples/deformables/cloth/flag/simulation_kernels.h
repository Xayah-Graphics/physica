#ifndef PHYSICA_EXAMPLES_DEFORMABLES_CLOTH_FLAG_SIMULATION_KERNELS_H
#define PHYSICA_EXAMPLES_DEFORMABLES_CLOTH_FLAG_SIMULATION_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::examples::cloth_flag::simulation_cuda {
    struct Wind final {
        float x;
        float y;
        float z;
        float gust_strength;
        float gust_frequency;
        float air_density;
        float drag_coefficient;
    };

    void write_aerodynamic_forces(::cuda::stream_ref stream, std::uint32_t particle_count, std::uint32_t triangle_count, std::uint64_t step, float time_step, Wind wind, const std::uint32_t* first, const std::uint32_t* second, const std::uint32_t* third, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, float* force_x, float* force_y, float* force_z);
} // namespace physica::examples::cloth_flag::simulation_cuda

#endif
