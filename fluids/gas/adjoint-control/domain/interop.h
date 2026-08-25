#ifndef PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_INTEROP_H
#define PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_INTEROP_H

#include "device.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace physica::fluids::gas::adjoint_control::cuda_detail {
    template<class Configuration>
    Grid grid(const Configuration& configuration) {
        return {
            .nx = configuration.resolution[0],
            .ny = configuration.resolution[1],
            .nz = configuration.resolution[2],
            .dimensions = static_cast<std::uint32_t>(configuration.dimension),
            .cell_size = configuration.cell_size,
            .time_step = configuration.time_step,
        };
    }

    template<class Field>
    ConstScalarView scalar(const Field& field) { return {.values = field.values.data()}; }

    template<class Field>
    ScalarView scalar(Field& field) { return {.values = field.values.data()}; }

    template<class Field>
    ConstCenteredVectorView centered(const Field& field) { return {.x = field.x.values.data(), .y = field.y.values.data(), .z = field.z.values.data()}; }

    template<class Field>
    CenteredVectorView centered(Field& field) { return {.x = field.x.values.data(), .y = field.y.values.data(), .z = field.z.values.data()}; }

    template<class Field>
    ConstStaggeredVectorView staggered(const Field& field) { return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()}; }

    template<class Field>
    StaggeredVectorView staggered(Field& field) { return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()}; }

    template<class Field>
    ConstScalarAdjointView scalar_adjoint(const Field& field) { return {.values = field.values.data()}; }

    template<class Field>
    ScalarAdjointView scalar_adjoint(Field& field) { return {.values = field.values.data()}; }

    template<class Field>
    ConstCenteredVectorAdjointView centered_adjoint(const Field& field) { return {.x = field.x.values.data(), .y = field.y.values.data(), .z = field.z.values.data()}; }

    template<class Field>
    CenteredVectorAdjointView centered_adjoint(Field& field) { return {.x = field.x.values.data(), .y = field.y.values.data(), .z = field.z.values.data()}; }

    template<class Field>
    ConstStaggeredVectorAdjointView staggered_adjoint(const Field& field) { return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()}; }

    template<class Field>
    StaggeredVectorAdjointView staggered_adjoint(Field& field) { return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()}; }

    template<class Boundary>
    ScalarBoundaryData scalar_boundary(const Boundary& boundary) {
        const std::array faces{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max};
        ScalarBoundaryData packed{};
        for (std::size_t face = 0u; face < faces.size(); ++face) {
            packed.modes[face] = static_cast<std::uint32_t>(faces[face].mode);
            packed.values[face] = faces[face].value;
        }
        return packed;
    }

    template<class Boundary>
    VelocityBoundaryData velocity_boundary(const Boundary& boundary) {
        const std::array faces{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max};
        VelocityBoundaryData packed{};
        for (std::size_t face = 0u; face < faces.size(); ++face) {
            packed.modes[face] = static_cast<std::uint32_t>(faces[face].mode);
            packed.values[face * 3u] = faces[face].value.x;
            packed.values[face * 3u + 1u] = faces[face].value.y;
            packed.values[face * 3u + 2u] = faces[face].value.z;
        }
        return packed;
    }
} // namespace physica::fluids::gas::adjoint_control::cuda_detail

#endif
