#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_NEIGHBORHOOD_DEVICE_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_NEIGHBORHOOD_DEVICE_H

#include "../domain/device.h"
#include <cstdint>

namespace physica::fluids::liquid::particle::cuda_detail {
    struct NeighborhoodView final {
        const std::uint64_t* sorted_keys;
        const std::uint32_t* sorted_particle_indices;
        const std::uint32_t* cell_offsets;
        const std::uint64_t* sorted_boundary_keys;
        const std::uint32_t* sorted_boundary_indices;
        const std::uint32_t* boundary_cell_offsets;
        std::uint32_t particle_count;
        std::uint32_t boundary_count;
        std::uint32_t cells_x;
        std::uint32_t cells_y;
        std::uint32_t cells_z;
        float origin_x;
        float origin_y;
        float origin_z;
        float cell_size;
    };

    struct BoundaryView final {
        const float* position_x;
        const float* position_y;
        const float* position_z;
        const float* velocity_x;
        const float* velocity_y;
        const float* velocity_z;
        const float* volumes;
        std::uint32_t count;
        float time;
    };
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
