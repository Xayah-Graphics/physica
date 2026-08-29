#ifndef PHYSICA_FLUIDS_GAS_INTEROP_H
#define PHYSICA_FLUIDS_GAS_INTEROP_H

#include <physica/fluids/gas/device.cuh>
#include <physica/fluids/grid/interop.h>
#include <array>
#include <cstddef>
#include <cstdint>

namespace physica::fluids::gas::device {
    template <class Configuration>
    Discretization discretization(const Configuration& configuration) {
        return {
            .grid       = grid::device::grid(configuration.grid, 0.0F),
            .dimensions = configuration.grid.resolution[2] == 1u ? 2u : 3u,
            .time_step  = configuration.time_step,
        };
    }

    template <class Boundary>
    ScalarBoundary scalar_boundary(const Boundary& boundary) {
        const std::array faces{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max};
        ScalarBoundary packed{};
        for (std::size_t face = 0u; face < faces.size(); ++face) packed.faces[face] = {.mode = static_cast<std::uint32_t>(faces[face].mode), .value = faces[face].value};
        return packed;
    }

    template <class Boundary>
    VelocityBoundary velocity_boundary(const Boundary& boundary) {
        const std::array faces{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max};
        VelocityBoundary packed{};
        for (std::size_t face = 0u; face < faces.size(); ++face) packed.faces[face] = {.mode = static_cast<std::uint32_t>(faces[face].mode), .value = faces[face].value};
        return packed;
    }
}

#endif
