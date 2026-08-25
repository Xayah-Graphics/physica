#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_NEIGHBORHOOD_DEVICE_CUH
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_NEIGHBORHOOD_DEVICE_CUH

#include "../domain/device.cuh"
#include "device.h"

namespace physica::fluids::liquid::particle::cuda_detail {
    struct CellRange final {
        std::uint32_t first;
        std::uint32_t last;
        std::uint32_t boundary_first;
        std::uint32_t boundary_last;
        bool valid;
    };

    __device__ inline Float3 boundary_position(const BoundaryView boundary, const std::uint32_t index) {
        return {
            boundary.position_x[index] + boundary.time * boundary.velocity_x[index],
            boundary.position_y[index] + boundary.time * boundary.velocity_y[index],
            boundary.position_z[index] + boundary.time * boundary.velocity_z[index],
        };
    }

    __device__ inline Float3 boundary_velocity(const BoundaryView boundary, const std::uint32_t index) {
        return {boundary.velocity_x[index], boundary.velocity_y[index], boundary.velocity_z[index]};
    }

    __device__ inline std::uint32_t cell_index(const NeighborhoodView neighborhood, const int x, const int y, const int z) {
        return (static_cast<std::uint32_t>(z) * neighborhood.cells_y + static_cast<std::uint32_t>(y)) * neighborhood.cells_x + static_cast<std::uint32_t>(x);
    }

    __device__ inline CellRange cell_range(const NeighborhoodView neighborhood, const int x, const int y, const int z) {
        if (x < 0 || x >= static_cast<int>(neighborhood.cells_x) || y < 0 || y >= static_cast<int>(neighborhood.cells_y) || z < 0 || z >= static_cast<int>(neighborhood.cells_z)) return {};
        const std::uint32_t cell = cell_index(neighborhood, x, y, z);
        return {
            .first          = neighborhood.cell_offsets[cell],
            .last           = neighborhood.cell_offsets[cell + 1u],
            .boundary_first = neighborhood.boundary_cell_offsets[cell],
            .boundary_last  = neighborhood.boundary_cell_offsets[cell + 1u],
            .valid          = true,
        };
    }

    __device__ inline void particle_cell(const NeighborhoodView neighborhood, const Float3 position, int& x, int& y, int& z) {
        x = min(static_cast<int>(neighborhood.cells_x) - 1, max(0, __float2int_rd((position.x - neighborhood.origin_x) / neighborhood.cell_size)));
        y = min(static_cast<int>(neighborhood.cells_y) - 1, max(0, __float2int_rd((position.y - neighborhood.origin_y) / neighborhood.cell_size)));
        z = min(static_cast<int>(neighborhood.cells_z) - 1, max(0, __float2int_rd((position.z - neighborhood.origin_z) / neighborhood.cell_size)));
    }
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
