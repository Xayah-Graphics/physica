#ifndef PHYSICA_FLUIDS_LIQUID_INTEROP_H
#define PHYSICA_FLUIDS_LIQUID_INTEROP_H

#include <physica/fluids/liquid/device.cuh>
#include <physica/field/device.cuh>
#include <cstdint>

namespace physica::fluids::liquid::device {
    template <class Configuration>
    CollisionBox collision_box(const Configuration& configuration, const std::uint64_t step_index) {
        const float time = static_cast<float>(step_index) * configuration.time_step;
        return {
            .bounds = {
                .minimum = {
                    configuration.boundary.bounds.minimum.x + time * configuration.boundary.velocity.x + configuration.particle_radius,
                    configuration.boundary.bounds.minimum.y + time * configuration.boundary.velocity.y + configuration.particle_radius,
                    configuration.boundary.bounds.minimum.z + time * configuration.boundary.velocity.z + configuration.particle_radius,
                },
                .maximum = {
                    configuration.boundary.bounds.maximum.x + time * configuration.boundary.velocity.x - configuration.particle_radius,
                    configuration.boundary.bounds.maximum.y + time * configuration.boundary.velocity.y - configuration.particle_radius,
                    configuration.boundary.bounds.maximum.z + time * configuration.boundary.velocity.z - configuration.particle_radius,
                },
            },
            .velocity = {
                configuration.boundary.velocity.x,
                configuration.boundary.velocity.y,
                configuration.boundary.velocity.z,
            },
            .no_slip = configuration.boundary.no_slip ? 1u : 0u,
        };
    }

    template <class Parameters>
    ParticleParameterView particle_parameters(const Parameters& value) {
        return {.masses = value.masses.data(), .rest_densities = value.rest_densities.data(), .viscosities = value.viscosities.data(), .surface_tensions = value.surface_tensions.data()};
    }

    template <class Parameters>
    ParticleParameterTangentView particle_parameter_tangent(const Parameters& value) {
        return {.masses = value.masses.data(), .rest_densities = value.rest_densities.data(), .viscosities = value.viscosities.data(), .surface_tensions = value.surface_tensions.data()};
    }

    template <class Parameters>
    ParticleParameterAdjointView particle_parameter_adjoint(Parameters& value) {
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
            .positions  = field::view(value.positions),
            .velocities = field::view(value.velocities),
            .volumes    = value.volumes.values.data(),
            .count      = static_cast<std::uint32_t>(value.volumes.values.size()),
            .time       = neighborhood.boundary_time,
        };
    }
}

#endif
