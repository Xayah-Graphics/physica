#ifndef PHYSICA_FLUIDS_GRID_INTEROP_H
#define PHYSICA_FLUIDS_GRID_INTEROP_H

#include <fluids/grid/device.cuh>

namespace physica::fluids::grid::device {
    template <class Configuration>
    Grid grid(const Configuration& configuration, const float time) {
        return {
            .nx         = configuration.resolution[0],
            .ny         = configuration.resolution[1],
            .nz         = configuration.resolution[2],
            .cell_size  = configuration.cell_size,
            .origin_x   = configuration.origin.x + configuration.velocity.x * time,
            .origin_y   = configuration.origin.y + configuration.velocity.y * time,
            .origin_z   = configuration.origin.z + configuration.velocity.z * time,
            .velocity_x = configuration.velocity.x,
            .velocity_y = configuration.velocity.y,
            .velocity_z = configuration.velocity.z,
        };
    }
} // namespace physica::fluids::grid::device

#endif
