#ifndef PHYSICA_FLUIDS_GAS_DETAIL_CUDA_INTEROP_H
#define PHYSICA_FLUIDS_GAS_DETAIL_CUDA_INTEROP_H

#include "types.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace physica::fluids::gas::detail::cuda {
    template <class Configuration>
    Grid grid(const Configuration& configuration) {
        return {
            .nx         = configuration.resolution[0],
            .ny         = configuration.resolution[1],
            .nz         = configuration.resolution[2],
            .dimensions = configuration.resolution[2] == 1u ? 2u : 3u,
            .cell_size  = configuration.cell_size,
            .time_step  = configuration.time_step,
        };
    }

    template <class Field>
    ScalarView<const float> scalar(const Field& field) {
        return ScalarView<const float>{field.values.data()};
    }

    template <class Field>
    ScalarView<float> scalar(Field& field) {
        return ScalarView<float>{field.values.data()};
    }

    template <class Field>
    CenteredVectorView<const float> centered(const Field& field) {
        return {field.x.values.data(), field.y.values.data(), field.z.values.data()};
    }

    template <class Field>
    CenteredVectorView<float> centered(Field& field) {
        return {field.x.values.data(), field.y.values.data(), field.z.values.data()};
    }

    template <class Field>
    StaggeredVectorView<const float> staggered(const Field& field) {
        return {field.x.data(), field.y.data(), field.z.data()};
    }

    template <class Field>
    StaggeredVectorView<float> staggered(Field& field) {
        return {field.x.data(), field.y.data(), field.z.data()};
    }

    template <class Field>
    ScalarView<const double> scalar_adjoint(const Field& field) {
        return ScalarView<const double>{field.values.data()};
    }

    template <class Field>
    ScalarView<double> scalar_adjoint(Field& field) {
        return ScalarView<double>{field.values.data()};
    }

    template <class Field>
    CenteredVectorView<const double> centered_adjoint(const Field& field) {
        return {field.x.values.data(), field.y.values.data(), field.z.values.data()};
    }

    template <class Field>
    CenteredVectorView<double> centered_adjoint(Field& field) {
        return {field.x.values.data(), field.y.values.data(), field.z.values.data()};
    }

    template <class Field>
    StaggeredVectorView<const double> staggered_adjoint(const Field& field) {
        return {field.x.data(), field.y.data(), field.z.data()};
    }

    template <class Field>
    StaggeredVectorView<double> staggered_adjoint(Field& field) {
        return {field.x.data(), field.y.data(), field.z.data()};
    }

    template <class Boundary>
    ScalarBoundaryData scalar_boundary(const Boundary& boundary) {
        const std::array faces{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max};
        ScalarBoundaryData packed{};
        for (std::size_t face = 0u; face < faces.size(); ++face) {
            packed.modes[face]  = static_cast<std::uint32_t>(faces[face].mode);
            packed.values[face] = faces[face].value;
        }
        return packed;
    }

    template <class Boundary>
    VelocityBoundaryData velocity_boundary(const Boundary& boundary) {
        const std::array faces{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max};
        VelocityBoundaryData packed{};
        for (std::size_t face = 0u; face < faces.size(); ++face) {
            packed.modes[face]            = static_cast<std::uint32_t>(faces[face].mode);
            packed.values[face * 3u]      = faces[face].value.x;
            packed.values[face * 3u + 1u] = faces[face].value.y;
            packed.values[face * 3u + 2u] = faces[face].value.z;
        }
        return packed;
    }
} // namespace physica::fluids::gas::detail::cuda

#endif
