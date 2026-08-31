module;

#include "pbd-kernels.h"
#include "../position-dynamics-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.pbd;

import std;

namespace physica::deformables::cloth::solvers::pbd {
    Solver::Solver(const Model<float>& model, Configuration configuration)
        : time_step(configuration.time_step), iteration_count(configuration.iteration_count), gravity(configuration.gravity), coloring(build_edge_coloring(model.topology.edges, model.particle_count)), colored_edges(model.stream, coloring.edges.size()), rest_lengths(model.stream, model.topology.edges.size()), fixed_vertex_mask(model.stream, model.particle_count), fixed_positions(model.stream, model.particle_count) {
        std::vector<float> host_rest_lengths(model.topology.edges.size());
        for (std::size_t edge_index = 0uz; edge_index < model.topology.edges.size(); ++edge_index) {
            const Edge edge               = model.topology.edges[edge_index];
            host_rest_lengths[edge_index] = length(model.configuration.rest_positions[edge.second] - model.configuration.rest_positions[edge.first]);
        }

        std::vector<std::uint32_t> host_fixed_vertex_mask(model.particle_count);
        std::vector<Vector3<float>> host_fixed_positions = model.configuration.rest_positions;
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            host_fixed_vertex_mask[fixed_vertex.particle] = 1u;
            host_fixed_positions[fixed_vertex.particle]    = fixed_vertex.position;
        }

        ::cuda::copy_bytes(model.stream, coloring.edges, colored_edges.values);
        ::cuda::copy_bytes(model.stream, host_rest_lengths, rest_lengths.values);
        ::cuda::copy_bytes(model.stream, host_fixed_vertex_mask, fixed_vertex_mask.values);
        simulation::upload(model.stream, host_fixed_positions, fixed_positions);
        model.stream.sync();
    }

    State<float> Solver::allocate_state(const Model<float>& model) const {
        State<float> result(model.stream, model.particle_count);
        simulation::clear(model.stream, result.positions);
        simulation::clear(model.stream, result.velocities);
        return result;
    }

    Control<float> Solver::allocate_control(const Model<float>& model) const {
        Control<float> result(model.stream, model.particle_count);
        simulation::clear(model.stream, result.external_forces);
        return result;
    }

    Solver::Parameters Solver::allocate_parameters(const Model<float>& model) const {
        Parameters result{.masses = simulation::ScalarField<float>(model.stream, model.particle_count)};
        simulation::clear(model.stream, result.masses);
        return result;
    }

    Solver::StepCache Solver::allocate_step_cache(const Model<float>&) const {
        return {};
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>&) const {
        return {};
    }

    void Solver::forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters& parameters, State<float>& next_state, StepCache&, Workspace&) const {
        position_dynamics::kernels::predict(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, gravity, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_forces), parameters.masses.values.data(), simulation::view(next_state.positions));
        for (std::uint32_t iteration = 0u; iteration < iteration_count; ++iteration) {
            for (std::size_t color = 0uz; color + 1uz < coloring.offsets.size(); ++color) {
                const std::uint32_t color_offset = coloring.offsets[color];
                const std::uint32_t edge_count   = coloring.offsets[color + 1uz] - color_offset;
                kernels::project_distance(model.stream, edge_count, color_offset, colored_edges.values.data(), model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), rest_lengths.values.data(), fixed_vertex_mask.values.data(), parameters.masses.values.data(), simulation::view(next_state.positions));
            }
        }
        position_dynamics::kernels::reconstruct_velocities(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(state.positions), simulation::view(next_state.positions), simulation::view(next_state.velocities));
    }
} // namespace physica::deformables::cloth::solvers::pbd
