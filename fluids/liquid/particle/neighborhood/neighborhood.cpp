module;

#include "../domain/interop.h"
#include "kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.particle.neighborhood;

import std;

namespace physica::fluids::liquid::particle {
    NeighborhoodSearch::NeighborhoodSearch(const Domain& source_domain)
        : domain(source_domain), unsorted_keys(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), domain.configuration.particle_count, ::cuda::no_init), unsorted_particle_indices(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), domain.configuration.particle_count, ::cuda::no_init), unsorted_boundary_keys(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), domain.configuration.boundary_particles.size(), ::cuda::no_init), unsorted_boundary_indices(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), domain.configuration.boundary_particles.size(), ::cuda::no_init), sort_scratch(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), cuda_detail::radix_sort_storage_size(domain.configuration.particle_count), ::cuda::no_init),
          boundary_sort_scratch(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), cuda_detail::radix_sort_storage_size(static_cast<std::uint32_t>(domain.configuration.boundary_particles.size())), ::cuda::no_init) {}

    Neighborhood NeighborhoodSearch::allocate() const {
        const DomainConfiguration& configuration = domain.configuration;
        const std::array<std::uint32_t, 3u> resolution{
            static_cast<std::uint32_t>(std::ceil((configuration.boundary.maximum.x - configuration.boundary.minimum.x) / configuration.support_radius)),
            static_cast<std::uint32_t>(std::ceil((configuration.boundary.maximum.y - configuration.boundary.minimum.y) / configuration.support_radius)),
            static_cast<std::uint32_t>(std::ceil((configuration.boundary.maximum.z - configuration.boundary.minimum.z) / configuration.support_radius)),
        };
        const std::size_t cell_count = static_cast<std::size_t>(resolution[0]) * resolution[1] * resolution[2];
        return {
            .sorted_keys             = ::cuda::device_buffer<std::uint64_t>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), configuration.particle_count, ::cuda::no_init},
            .sorted_particle_indices = ::cuda::device_buffer<std::uint32_t>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), configuration.particle_count, ::cuda::no_init},
            .cell_offsets            = ::cuda::device_buffer<std::uint32_t>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), cell_count + 1uz, ::cuda::no_init},
            .sorted_boundary_keys    = ::cuda::device_buffer<std::uint64_t>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), configuration.boundary_particles.size(), ::cuda::no_init},
            .sorted_boundary_indices = ::cuda::device_buffer<std::uint32_t>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), configuration.boundary_particles.size(), ::cuda::no_init},
            .boundary_cell_offsets   = ::cuda::device_buffer<std::uint32_t>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), cell_count + 1uz, ::cuda::no_init},
            .cell_resolution         = resolution,
            .cell_origin             = configuration.boundary.minimum,
            .cell_size               = configuration.support_radius,
        };
    }

    void NeighborhoodSearch::build(const std::uint64_t step_index, const VectorField& positions, Neighborhood& neighborhood) {
        const DomainConfiguration& configuration = domain.configuration;
        cuda_detail::build_neighborhood(domain.stream, configuration.particle_count, static_cast<std::uint32_t>(configuration.boundary_particles.size()), configuration.support_radius, configuration.time_step, step_index,
            {
                .minimum_x  = configuration.boundary.minimum.x,
                .minimum_y  = configuration.boundary.minimum.y,
                .minimum_z  = configuration.boundary.minimum.z,
                .maximum_x  = configuration.boundary.maximum.x,
                .maximum_y  = configuration.boundary.maximum.y,
                .maximum_z  = configuration.boundary.maximum.z,
                .velocity_x = configuration.boundary.velocity.x,
                .velocity_y = configuration.boundary.velocity.y,
                .velocity_z = configuration.boundary.velocity.z,
                .no_slip    = configuration.boundary.no_slip ? 1u : 0u,
            },
            cuda_detail::vector(positions), cuda_detail::vector(domain.boundary.positions), cuda_detail::vector(domain.boundary.velocities), unsorted_keys.data(), unsorted_particle_indices.data(), unsorted_boundary_keys.data(), unsorted_boundary_indices.data(), sort_scratch.data(), sort_scratch.size(), boundary_sort_scratch.data(), boundary_sort_scratch.size(), neighborhood.sorted_keys.data(), neighborhood.sorted_particle_indices.data(), neighborhood.cell_offsets.data(), neighborhood.sorted_boundary_keys.data(), neighborhood.sorted_boundary_indices.data(), neighborhood.boundary_cell_offsets.data());
        neighborhood.boundary_time = static_cast<float>(step_index) * configuration.time_step;
        neighborhood.cell_origin   = {
              .x = configuration.boundary.minimum.x + neighborhood.boundary_time * configuration.boundary.velocity.x,
              .y = configuration.boundary.minimum.y + neighborhood.boundary_time * configuration.boundary.velocity.y,
              .z = configuration.boundary.minimum.z + neighborhood.boundary_time * configuration.boundary.velocity.z,
        };
    }
} // namespace physica::fluids::liquid::particle
