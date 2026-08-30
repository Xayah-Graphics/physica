module;

#include "grid-step-kernels.h"
#include <fluids/grid/interop.h>
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.fluids.liquid.solvers.pic.grid_step;

import std;

namespace physica::fluids::liquid::solvers::pic {
    GridStep::GridStep(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    GridStep::State GridStep::allocate_state(const Model& model) const {
        return {
            .velocity_before_projection = model.grid.allocate_mac_field<float>(),
            .velocity                   = model.grid.allocate_mac_field<float>(),
            .cell_types                 = model.grid.allocate_cell_field<std::uint32_t>(),
            .level_set                  = model.grid.allocate_cell_field<float>(),
            .divergence                 = model.grid.allocate_cell_field<float>(),
            .pressure                   = model.grid.allocate_cell_field<float>(),
        };
    }

    GridStep::Workspace GridStep::allocate_workspace(const Model& model) const {
        const ::cuda::stream_ref stream = model.grid.stream;
        return {
            .face_mass                 = model.grid.allocate_mac_field<float>(),
            .valid_faces               = model.grid.allocate_mac_field<std::uint32_t>(),
            .projection_input_velocity = model.grid.allocate_mac_field<float>(),
            .velocity_scratch          = model.grid.allocate_mac_field<float>(),
            .valid_faces_scratch       = model.grid.allocate_mac_field<std::uint32_t>(),
            .particle_counts           = model.grid.allocate_cell_field<std::uint32_t>(),
            .divergence_metrics        = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), 3u, ::cuda::no_init},
        };
    }

    void GridStep::begin_transfer(const Model& model, State& state, Workspace& workspace) const {
        model.grid.clear(state.velocity_before_projection);
        model.grid.clear(workspace.face_mass);
    }

    void GridStep::classify_and_normalize(const Model& model, const float time, const std::uint32_t particle_count, const simulation::VectorField<float>& positions, State& state, Workspace& workspace) const {
        const grid::device::Grid device_grid = grid::device::grid(model.grid.configuration, time);
        kernels::grid_step::classify(model.grid.stream, device_grid, particle_count, configuration.level_set_radius_cells * model.grid.configuration.cell_size, simulation::view(positions), workspace.particle_counts.values.data(), state.cell_types.values.data(), state.level_set.values.data());
        kernels::grid_step::normalize(model.grid.stream, device_grid, simulation::view(workspace.face_mass), simulation::view(state.velocity_before_projection), simulation::view(workspace.valid_faces));
    }

    void GridStep::extrapolate_before_projection(const Model& model, const float time, State& state, Workspace& workspace) const {
        const grid::device::Grid device_grid = grid::device::grid(model.grid.configuration, time);
        for (std::uint32_t layer = 0u; layer < configuration.extrapolation_layers; ++layer) {
            kernels::grid_step::extrapolate_layer(model.grid.stream, device_grid, simulation::view(state.velocity_before_projection), simulation::view(workspace.valid_faces), simulation::view(workspace.velocity_scratch), simulation::view(workspace.valid_faces_scratch));
            std::swap(state.velocity_before_projection, workspace.velocity_scratch);
            std::swap(workspace.valid_faces, workspace.valid_faces_scratch);
        }
    }

    void GridStep::apply_force_and_constrain(const Model& model, const float time, const float time_step, const simulation::ScalarField<std::uint32_t>& cell_types, simulation::VectorField<float>& velocity) const {
        kernels::grid_step::add_force_and_constrain(model.grid.stream, grid::device::grid(model.grid.configuration, time), model.no_slip, time_step, {configuration.acceleration.x, configuration.acceleration.y, configuration.acceleration.z}, cell_types.values.data(), simulation::view(velocity));
    }

    void GridStep::prepare_after_projection(const Model& model, const float time, State& state, Workspace& workspace) const {
        const grid::device::Grid device_grid = grid::device::grid(model.grid.configuration, time);
        kernels::grid_step::mark_fluid_faces(model.grid.stream, device_grid, model.no_slip, state.cell_types.values.data(), simulation::view(workspace.valid_faces));
        for (std::uint32_t layer = 0u; layer < configuration.extrapolation_layers; ++layer) {
            kernels::grid_step::extrapolate_layer(model.grid.stream, device_grid, simulation::view(state.velocity), simulation::view(workspace.valid_faces), simulation::view(workspace.velocity_scratch), simulation::view(workspace.valid_faces_scratch));
            std::swap(state.velocity, workspace.velocity_scratch);
            std::swap(workspace.valid_faces, workspace.valid_faces_scratch);
        }
    }

    GridStep::Diagnostics GridStep::divergence(const Model& model, const float time, const simulation::ScalarField<std::uint32_t>& cell_types, const simulation::VectorField<float>& velocity, simulation::ScalarField<float>& values, Workspace& workspace) const {
        ::cuda::fill_bytes(model.grid.stream, workspace.divergence_metrics, 0u);
        kernels::grid_step::compute_divergence(model.grid.stream, grid::device::grid(model.grid.configuration, time), cell_types.values.data(), simulation::view(velocity), values.values.data(), workspace.divergence_metrics.data());
        std::array<float, 3u> metrics{};
        ::cuda::copy_bytes(model.grid.stream, workspace.divergence_metrics, ::cuda::std::span{metrics.data(), metrics.size()});
        model.grid.stream.sync();
        return {.l2 = metrics[2] == 0.0F ? 0.0F : std::sqrt(metrics[0] / metrics[2]), .maximum = metrics[1]};
    }
} // namespace physica::fluids::liquid::solvers::pic
