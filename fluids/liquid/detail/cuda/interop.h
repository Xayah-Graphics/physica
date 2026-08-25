#ifndef PHYSICA_FLUIDS_LIQUID_DETAIL_CUDA_INTEROP_H
#define PHYSICA_FLUIDS_LIQUID_DETAIL_CUDA_INTEROP_H

#include "types.h"
#include <type_traits>
#include <utility>

namespace physica::fluids::liquid::cuda_detail {
    template <class Field>
    ConstVectorView<std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<const Field&>().x.data())>>> vector(const Field& field) {
        return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
    }

    template <class Field>
    VectorView<std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<Field&>().x.data())>>> vector(Field& field) {
        return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
    }

    template <class Parameters>
    ParticleParameterView parameters(const Parameters& value) {
        return {.masses = value.masses.data(), .rest_densities = value.rest_densities.data(), .viscosities = value.viscosities.data(), .surface_tensions = value.surface_tensions.data()};
    }

    template <class Parameters>
    ParticleParameterTangentView parameter_tangent(const Parameters& value) {
        return {.masses = value.masses.data(), .rest_densities = value.rest_densities.data(), .viscosities = value.viscosities.data(), .surface_tensions = value.surface_tensions.data()};
    }

    template <class Parameters>
    ParticleParameterAdjointView parameter_adjoint(Parameters& value) {
        return {.masses = value.masses.data(), .rest_densities = value.rest_densities.data(), .viscosities = value.viscosities.data(), .surface_tensions = value.surface_tensions.data()};
    }

    template <class Neighborhood>
    NeighborhoodView neighborhood(const Neighborhood& value) {
        return {
            .sorted_keys             = value.sorted_keys.data(),
            .sorted_particle_indices = value.sorted_particle_indices.data(),
            .cell_offsets            = value.cell_offsets.data(),
            .sorted_boundary_keys    = value.sorted_boundary_keys.data(),
            .sorted_boundary_indices = value.sorted_boundary_indices.data(),
            .boundary_cell_offsets   = value.boundary_cell_offsets.data(),
            .particle_count          = static_cast<std::uint32_t>(value.sorted_keys.size()),
            .boundary_count          = static_cast<std::uint32_t>(value.sorted_boundary_keys.size()),
            .cells_x                 = value.cell_resolution[0],
            .cells_y                 = value.cell_resolution[1],
            .cells_z                 = value.cell_resolution[2],
            .origin_x                = value.cell_origin.x,
            .origin_y                = value.cell_origin.y,
            .origin_z                = value.cell_origin.z,
            .cell_size               = value.cell_size,
        };
    }

    template <class Boundary, class Neighborhood>
    BoundaryView boundary(const Boundary& value, const Neighborhood& neighborhood) {
        return {
            .position_x = value.positions.x.data(),
            .position_y = value.positions.y.data(),
            .position_z = value.positions.z.data(),
            .velocity_x = value.velocities.x.data(),
            .velocity_y = value.velocities.y.data(),
            .velocity_z = value.velocities.z.data(),
            .volumes    = value.volumes.values.data(),
            .count      = static_cast<std::uint32_t>(value.volumes.values.size()),
            .time       = neighborhood.boundary_time,
        };
    }
} // namespace physica::fluids::liquid::cuda_detail

#endif
