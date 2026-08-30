module;

#include "particle-step-kernels.h"
#include <fluids/grid/interop.h>
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.fluids.liquid.solvers.pic.particle_step;

import std;

namespace physica::fluids::liquid::solvers::pic {
    ParticleStep::ParticleStep(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ParticleStep::Workspace ParticleStep::allocate_workspace(const Model& model) const {
        const ::cuda::stream_ref stream          = model.grid.stream;
        const std::size_t maximum_particle_count = model.maximum_particle_count;
        const std::size_t reduction_count        = std::max(model.grid.cell_count, maximum_particle_count);
        return {
            .raw_counts           = model.grid.allocate_cell_field<std::uint32_t>(),
            .survivor_counts      = model.grid.allocate_cell_field<std::uint32_t>(),
            .keep_flags           = simulation::ScalarField<std::uint32_t>(model.grid.stream, maximum_particle_count),
            .destinations         = simulation::ScalarField<std::uint32_t>(model.grid.stream, maximum_particle_count),
            .seed_counts          = model.grid.allocate_cell_field<std::uint32_t>(),
            .seed_offsets         = model.grid.allocate_cell_field<std::uint32_t>(),
            .compacted_positions  = simulation::VectorField<float>(model.grid.stream, maximum_particle_count),
            .compacted_velocities = simulation::VectorField<float>(model.grid.stream, maximum_particle_count),
            .speeds               = simulation::ScalarField<float>(model.grid.stream, maximum_particle_count),
            .diagnostic_values    = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), maximum_particle_count * 7u, ::cuda::no_init},
            .diagnostic_output    = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), 7u, ::cuda::no_init},
            .reduction_output     = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), 1u, ::cuda::no_init},
            .maintenance_totals   = ::cuda::device_buffer<std::uint32_t>{stream, ::cuda::device_default_memory_pool(stream.device()), 2u, ::cuda::no_init},
            .reduction_scratch    = ::cuda::device_buffer<std::byte>{stream, ::cuda::device_default_memory_pool(stream.device()), kernels::particle_step::reduction_storage_size(reduction_count), ::cuda::no_init},
            .scan_scratch         = ::cuda::device_buffer<std::byte>{stream, ::cuda::device_default_memory_pool(stream.device()), kernels::particle_step::scan_storage_size(reduction_count), ::cuda::no_init},
        };
    }

    float ParticleStep::maximum_speed(const Model& model, const std::uint32_t particle_count, const simulation::VectorField<float>& velocities, Workspace& workspace) const {
        kernels::particle_step::particle_speeds(model.grid.stream, particle_count, simulation::view(velocities), workspace.speeds.values.data());
        kernels::particle_step::reduce_maximum(model.grid.stream, workspace.speeds.values.data(), particle_count, workspace.reduction_output.data(), workspace.reduction_scratch.data(), workspace.reduction_scratch.size());
        float maximum{};
        ::cuda::copy_bytes(model.grid.stream, workspace.reduction_output, ::cuda::std::span{&maximum, 1u});
        model.grid.stream.sync();
        return maximum;
    }

    ParticleStep::Diagnostics ParticleStep::diagnostics(const Model& model, const float time, const std::uint32_t particle_count, const float particle_mass, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, Workspace& workspace) const {
        const std::size_t stride = model.maximum_particle_count;
        kernels::particle_step::particle_diagnostics(model.grid.stream, grid::device::grid(model.grid.configuration, time), particle_count, stride, particle_mass, simulation::view(positions), simulation::view(velocities), workspace.diagnostic_values.data());
        for (std::size_t component = 0uz; component < 7uz; ++component) kernels::particle_step::reduce_sum(model.grid.stream, workspace.diagnostic_values.data() + component * stride, particle_count, workspace.diagnostic_output.data() + component, workspace.reduction_scratch.data(), workspace.reduction_scratch.size());
        std::array<float, 7u> values{};
        ::cuda::copy_bytes(model.grid.stream, workspace.diagnostic_output, ::cuda::std::span{values.data(), values.size()});
        model.grid.stream.sync();
        return {
            .kinetic_energy   = values[0],
            .linear_momentum  = {.x = values[1], .y = values[2], .z = values[3]},
            .angular_momentum = {.x = values[4], .y = values[5], .z = values[6]},
        };
    }

    void ParticleStep::advect(const Model& model, const float time, const float time_step, const std::uint32_t particle_count, const simulation::VectorField<float>& grid_velocity, simulation::VectorField<float>& positions, simulation::VectorField<float>& velocities) const {
        kernels::particle_step::advect(model.grid.stream, grid::device::grid(model.grid.configuration, time), model.particle_radius, model.no_slip, time_step, particle_count, simulation::view(grid_velocity), simulation::view(positions), simulation::view(velocities));
    }

    ParticleStep::Maintenance ParticleStep::plan_maintenance(const Model& model, const float time, const std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::ScalarField<std::uint32_t>& cell_types, const simulation::ScalarField<float>& level_set, Workspace& workspace) const {
        kernels::particle_step::plan_maintenance(model.grid.stream, grid::device::grid(model.grid.configuration, time), particle_count, configuration.minimum_per_cell, configuration.target_per_cell, configuration.maximum_per_cell, simulation::view(positions), cell_types.values.data(), level_set.values.data(), workspace.raw_counts.values.data(), workspace.survivor_counts.values.data(), workspace.keep_flags.values.data(), workspace.destinations.values.data(), workspace.seed_counts.values.data(), workspace.seed_offsets.values.data(), workspace.maintenance_totals.data(), workspace.scan_scratch.data(), workspace.scan_scratch.size());
        std::array<std::uint32_t, 2u> totals{};
        ::cuda::copy_bytes(model.grid.stream, workspace.maintenance_totals, ::cuda::std::span{totals.data(), totals.size()});
        model.grid.stream.sync();
        return {.survivor_count = totals[0], .seed_count = totals[1]};
    }

    void ParticleStep::compact_and_seed(const Model& model, const float time, const std::uint64_t seed, const std::uint32_t particle_count, const Maintenance& maintenance, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>& grid_velocity, Workspace& workspace) const {
        kernels::particle_step::compact_and_seed(model.grid.stream, grid::device::grid(model.grid.configuration, time), seed, particle_count, maintenance.survivor_count, maintenance.seed_count, simulation::view(positions), simulation::view(velocities), workspace.keep_flags.values.data(), workspace.destinations.values.data(), workspace.seed_counts.values.data(), workspace.seed_offsets.values.data(), simulation::view(grid_velocity), simulation::view(workspace.compacted_positions), simulation::view(workspace.compacted_velocities));
    }
} // namespace physica::fluids::liquid::solvers::pic
