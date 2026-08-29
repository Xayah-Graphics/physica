module;

#include <physica/fluids/liquid/interop.h>
#include "neighborhood-kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.operators.neighborhood;

import std;

namespace physica::fluids::liquid::operators {
    Neighborhood UniformGridNeighborhood::allocate_cache(const meshfree::Model& model) const {
        const meshfree::Configuration& configuration = model.configuration;
        const std::array<std::uint32_t, 3u> resolution{
            static_cast<std::uint32_t>(std::ceil((configuration.boundary.bounds.maximum.x - configuration.boundary.bounds.minimum.x) / configuration.support_radius)),
            static_cast<std::uint32_t>(std::ceil((configuration.boundary.bounds.maximum.y - configuration.boundary.bounds.minimum.y) / configuration.support_radius)),
            static_cast<std::uint32_t>(std::ceil((configuration.boundary.bounds.maximum.z - configuration.boundary.bounds.minimum.z) / configuration.support_radius)),
        };
        const std::size_t cell_count = static_cast<std::size_t>(resolution[0]) * resolution[1] * resolution[2];
        return {
            .sorted_keys             = ::cuda::device_buffer<std::uint64_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), configuration.particle_count, ::cuda::no_init},
            .sorted_particle_indices = ::cuda::device_buffer<std::uint32_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), configuration.particle_count, ::cuda::no_init},
            .cell_offsets            = ::cuda::device_buffer<std::uint32_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), cell_count + 1uz, ::cuda::no_init},
            .sorted_boundary_keys    = ::cuda::device_buffer<std::uint64_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), configuration.boundary_particles.size(), ::cuda::no_init},
            .sorted_boundary_indices = ::cuda::device_buffer<std::uint32_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), configuration.boundary_particles.size(), ::cuda::no_init},
            .boundary_cell_offsets   = ::cuda::device_buffer<std::uint32_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), cell_count + 1uz, ::cuda::no_init},
            .cell_resolution         = resolution,
            .cell_origin             = configuration.boundary.bounds.minimum,
            .cell_size               = configuration.support_radius,
        };
    }

    UniformGridNeighborhood::Workspace UniformGridNeighborhood::allocate_workspace(const meshfree::Model& model) const {
        return {
            .unsorted_keys             = ::cuda::device_buffer<std::uint64_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), model.configuration.particle_count, ::cuda::no_init},
            .unsorted_particle_indices = ::cuda::device_buffer<std::uint32_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), model.configuration.particle_count, ::cuda::no_init},
            .unsorted_boundary_keys    = ::cuda::device_buffer<std::uint64_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), model.configuration.boundary_particles.size(), ::cuda::no_init},
            .unsorted_boundary_indices = ::cuda::device_buffer<std::uint32_t>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), model.configuration.boundary_particles.size(), ::cuda::no_init},
            .sort_scratch              = ::cuda::device_buffer<std::byte>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), kernels::neighborhood::radix_sort_storage_size(model.configuration.particle_count), ::cuda::no_init},
            .boundary_sort_scratch     = ::cuda::device_buffer<std::byte>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), kernels::neighborhood::radix_sort_storage_size(static_cast<std::uint32_t>(model.configuration.boundary_particles.size())), ::cuda::no_init},
        };
    }

    void UniformGridNeighborhood::build(const meshfree::Model& model, const std::uint64_t step_index, const VectorField<float>& positions, Neighborhood& neighborhood, Workspace& workspace) const {
        const meshfree::Configuration& configuration = model.configuration;
        kernels::neighborhood::build_neighborhood(model.fields.stream, configuration.particle_count, static_cast<std::uint32_t>(configuration.boundary_particles.size()), configuration.support_radius, configuration.time_step, step_index,
            {
                .bounds = {
                    .minimum = {configuration.boundary.bounds.minimum.x, configuration.boundary.bounds.minimum.y, configuration.boundary.bounds.minimum.z},
                    .maximum = {configuration.boundary.bounds.maximum.x, configuration.boundary.bounds.maximum.y, configuration.boundary.bounds.maximum.z},
                },
                .velocity = {configuration.boundary.velocity.x, configuration.boundary.velocity.y, configuration.boundary.velocity.z},
                .no_slip  = configuration.boundary.no_slip ? 1u : 0u,
            },
            field::view(positions), field::view(model.boundary.positions), field::view(model.boundary.velocities), workspace.unsorted_keys.data(), workspace.unsorted_particle_indices.data(), workspace.unsorted_boundary_keys.data(), workspace.unsorted_boundary_indices.data(), workspace.sort_scratch.data(), workspace.sort_scratch.size(), workspace.boundary_sort_scratch.data(), workspace.boundary_sort_scratch.size(), neighborhood.sorted_keys.data(), neighborhood.sorted_particle_indices.data(), neighborhood.cell_offsets.data(), neighborhood.sorted_boundary_keys.data(), neighborhood.sorted_boundary_indices.data(), neighborhood.boundary_cell_offsets.data());
        neighborhood.boundary_time = static_cast<float>(step_index) * configuration.time_step;
        neighborhood.cell_origin   = {
              .x = configuration.boundary.bounds.minimum.x + neighborhood.boundary_time * configuration.boundary.velocity.x,
              .y = configuration.boundary.bounds.minimum.y + neighborhood.boundary_time * configuration.boundary.velocity.y,
              .z = configuration.boundary.bounds.minimum.z + neighborhood.boundary_time * configuration.boundary.velocity.z,
        };
    }
} // namespace physica::fluids::liquid::operators
