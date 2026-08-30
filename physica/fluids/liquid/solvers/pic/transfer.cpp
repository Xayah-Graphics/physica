module;

#include <simulation/field/device.cuh>
#include <fluids/grid/interop.h>
#include "transfer-kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.solvers.pic.transfer;

import std;

namespace physica::fluids::liquid::solvers::pic {
    FlipTransfer::FlipTransfer(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    FlipTransfer::State FlipTransfer::allocate_state(const Model&) const {
        return {};
    }

    FlipTransfer::Workspace FlipTransfer::allocate_workspace(const Model&) const {
        return {};
    }

    void FlipTransfer::clear_state(const Model&, State&) const {}

    void FlipTransfer::copy_state(const Model&, const State&, State&) const {}

    void FlipTransfer::particle_to_grid(const Model& model, const float time, const std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const State&, simulation::VectorField<float>& momentum, simulation::VectorField<float>& mass) const {
        kernels::transfer::flip_particle_to_grid(model.grid.stream, grid::device::grid(model.grid.configuration, time), particle_count, simulation::view(positions), simulation::view(velocities), simulation::view(momentum), simulation::view(mass));
    }

    void FlipTransfer::grid_to_particle(const Model& model, const float time, const std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& input_velocities, const State&, const simulation::VectorField<float>& old_grid_velocity, const simulation::VectorField<float>& new_grid_velocity, simulation::VectorField<float>& output_velocities, State&) const {
        kernels::transfer::flip_grid_to_particle(model.grid.stream, grid::device::grid(model.grid.configuration, time), particle_count, configuration.ratio, simulation::view(positions), simulation::view(input_velocities), simulation::view(old_grid_velocity), simulation::view(new_grid_velocity), simulation::view(output_velocities));
    }

    void FlipTransfer::compact_and_seed(const Model&, const float, const std::uint32_t, const ParticleStep::Maintenance&, const simulation::VectorField<float>&, const simulation::VectorField<float>&, const State&, State&, const simulation::ScalarField<std::uint32_t>&, const simulation::ScalarField<std::uint32_t>&, Workspace&) const {}

    void FlipTransfer::commit_compaction(State&, Workspace&) const {}

    ApicTransfer::ApicTransfer(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ApicTransfer::State ApicTransfer::allocate_state(const Model& model) const {
        return {.affine = simulation::Matrix3Field<float>(model.grid.stream, model.maximum_particle_count)};
    }

    ApicTransfer::Workspace ApicTransfer::allocate_workspace(const Model& model) const {
        return {.compacted_affine = simulation::Matrix3Field<float>(model.grid.stream, model.maximum_particle_count)};
    }

    void ApicTransfer::clear_state(const Model& model, State& state) const {
        simulation::clear(model.grid.stream, state.affine);
    }

    void ApicTransfer::copy_state(const Model& model, const State& source, State& destination) const {
        simulation::copy(model.grid.stream, source.affine, destination.affine);
    }

    void ApicTransfer::particle_to_grid(const Model& model, const float time, const std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const State& state, simulation::VectorField<float>& momentum, simulation::VectorField<float>& mass) const {
        kernels::transfer::apic_particle_to_grid(model.grid.stream, grid::device::grid(model.grid.configuration, time), particle_count, simulation::view(positions), simulation::view(velocities), simulation::matrix3_view(state.affine), simulation::view(momentum), simulation::view(mass));
    }

    void ApicTransfer::grid_to_particle(const Model& model, const float time, const std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::VectorField<float>&, const State&, const simulation::VectorField<float>&, const simulation::VectorField<float>& new_grid_velocity, simulation::VectorField<float>& output_velocities, State& output_state) const {
        kernels::transfer::apic_grid_to_particle(model.grid.stream, grid::device::grid(model.grid.configuration, time), particle_count, configuration.affine_ratio, simulation::view(positions), simulation::view(new_grid_velocity), simulation::view(output_velocities), simulation::matrix3_view(output_state.affine));
    }

    void ApicTransfer::compact_and_seed(const Model& model, const float time, const std::uint32_t source_particle_count, const ParticleStep::Maintenance& maintenance, const simulation::VectorField<float>& compacted_positions, const simulation::VectorField<float>& grid_velocity, const State& source, State&, const simulation::ScalarField<std::uint32_t>& keep_flags, const simulation::ScalarField<std::uint32_t>& destinations, Workspace& workspace) const {
        kernels::transfer::apic_compact_and_seed(model.grid.stream, grid::device::grid(model.grid.configuration, time), source_particle_count, maintenance.survivor_count, maintenance.seed_count, configuration.affine_ratio, simulation::view(compacted_positions), simulation::view(grid_velocity), simulation::matrix3_view(source.affine), keep_flags.values.data(), destinations.values.data(), simulation::matrix3_view(workspace.compacted_affine));
    }

    void ApicTransfer::commit_compaction(State& state, Workspace& workspace) const {
        std::swap(state.affine, workspace.compacted_affine);
    }
} // namespace physica::fluids::liquid::solvers::pic
