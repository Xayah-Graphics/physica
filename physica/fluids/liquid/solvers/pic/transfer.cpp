module;

#include <physica/field/device.cuh>
#include <physica/fluids/grid/interop.h>
#include "transfer-kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.pic;

import std;
import :transfer;

namespace physica::fluids::liquid::pic {
    FlipTransfer::FlipTransfer(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    FlipTransfer::State FlipTransfer::allocate_state(const Model&) const {
        return {};
    }

    FlipTransfer::Workspace FlipTransfer::allocate_workspace(const Model&) const {
        return {};
    }

    void FlipTransfer::clear_state(const Model&, State&) const {}

    void FlipTransfer::copy_state(const Model&, const State&, State&) const {}

    void FlipTransfer::particle_to_grid(const Model& model, const float time, const std::uint32_t particle_count, const VectorField<float>& positions, const VectorField<float>& velocities, const State&, VectorField<float>& momentum, VectorField<float>& mass) const {
        kernels::transfer::flip_particle_to_grid(model.grid.fields.stream, grid::device::grid(model.grid.configuration, time), particle_count, field::view(positions), field::view(velocities), field::view(momentum), field::view(mass));
    }

    void FlipTransfer::grid_to_particle(const Model& model, const float time, const std::uint32_t particle_count, const VectorField<float>& positions, const VectorField<float>& input_velocities, const State&, const VectorField<float>& old_grid_velocity, const VectorField<float>& new_grid_velocity, VectorField<float>& output_velocities, State&) const {
        kernels::transfer::flip_grid_to_particle(model.grid.fields.stream, grid::device::grid(model.grid.configuration, time), particle_count, configuration.ratio, field::view(positions), field::view(input_velocities), field::view(old_grid_velocity), field::view(new_grid_velocity), field::view(output_velocities));
    }

    void FlipTransfer::compact_and_seed(const Model&, const float, const std::uint32_t, const ParticleStep::Maintenance&, const VectorField<float>&, const VectorField<float>&, const State&, State&, const ScalarField<std::uint32_t>&, const ScalarField<std::uint32_t>&, Workspace&) const {}

    void FlipTransfer::commit_compaction(State&, Workspace&) const {}

    ApicTransfer::ApicTransfer(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ApicTransfer::State ApicTransfer::allocate_state(const Model& model) const {
        return {.affine = model.grid.fields.allocate_matrix3_field<float>(model.maximum_particle_count)};
    }

    ApicTransfer::Workspace ApicTransfer::allocate_workspace(const Model& model) const {
        return {.compacted_affine = model.grid.fields.allocate_matrix3_field<float>(model.maximum_particle_count)};
    }

    void ApicTransfer::clear_state(const Model& model, State& state) const {
        model.grid.fields.clear(state.affine);
    }

    void ApicTransfer::copy_state(const Model& model, const State& source, State& destination) const {
        model.grid.fields.copy(source.affine, destination.affine);
    }

    void ApicTransfer::particle_to_grid(const Model& model, const float time, const std::uint32_t particle_count, const VectorField<float>& positions, const VectorField<float>& velocities, const State& state, VectorField<float>& momentum, VectorField<float>& mass) const {
        kernels::transfer::apic_particle_to_grid(model.grid.fields.stream, grid::device::grid(model.grid.configuration, time), particle_count, field::view(positions), field::view(velocities), field::matrix3_view(state.affine), field::view(momentum), field::view(mass));
    }

    void ApicTransfer::grid_to_particle(const Model& model, const float time, const std::uint32_t particle_count, const VectorField<float>& positions, const VectorField<float>&, const State&, const VectorField<float>&, const VectorField<float>& new_grid_velocity, VectorField<float>& output_velocities, State& output_state) const {
        kernels::transfer::apic_grid_to_particle(model.grid.fields.stream, grid::device::grid(model.grid.configuration, time), particle_count, configuration.affine_ratio, field::view(positions), field::view(new_grid_velocity), field::view(output_velocities), field::matrix3_view(output_state.affine));
    }

    void ApicTransfer::compact_and_seed(const Model& model, const float time, const std::uint32_t source_particle_count, const ParticleStep::Maintenance& maintenance, const VectorField<float>& compacted_positions, const VectorField<float>& grid_velocity, const State& source, State&, const ScalarField<std::uint32_t>& keep_flags, const ScalarField<std::uint32_t>& destinations, Workspace& workspace) const {
        kernels::transfer::apic_compact_and_seed(model.grid.fields.stream, grid::device::grid(model.grid.configuration, time), source_particle_count, maintenance.survivor_count, maintenance.seed_count, configuration.affine_ratio, field::view(compacted_positions), field::view(grid_velocity), field::matrix3_view(source.affine), keep_flags.values.data(), destinations.values.data(), field::matrix3_view(workspace.compacted_affine));
    }

    void ApicTransfer::commit_compaction(State& state, Workspace& workspace) const {
        std::swap(state.affine, workspace.compacted_affine);
    }
} // namespace physica::fluids::liquid::pic
