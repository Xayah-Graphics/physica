module;

#include "xpbd-kernels.h"
#include "../position-dynamics-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.xpbd;

import std;

namespace physica::deformables::cloth::solvers::xpbd {
    Solver::Solver(const Model<float>& model, Configuration configuration)
        : time_step(configuration.time_step),
          iteration_count(configuration.iteration_count),
          gravity(configuration.gravity),
          constraints(build_constraints(model)),
          coloring(build_edge_coloring(constraints, model.particle_count)),
          colored_constraints(model.stream, coloring.edges.size()),
          constraint_first(model.stream, constraints.size()),
          constraint_second(model.stream, constraints.size()),
          rest_lengths(model.stream, constraints.size()),
          compliances(model.stream, constraints.size()),
          fixed_vertex_mask(model.stream, model.particle_count),
          fixed_positions(model.stream, model.particle_count) {
        std::vector<std::uint32_t> host_first(constraints.size());
        std::vector<std::uint32_t> host_second(constraints.size());
        std::vector<float> host_rest_lengths(constraints.size());
        std::vector<float> host_compliances(constraints.size());
        for (std::size_t constraint = 0uz; constraint < constraints.size(); ++constraint) {
            const Edge edge                 = constraints[constraint];
            host_first[constraint]          = edge.first;
            host_second[constraint]         = edge.second;
            host_rest_lengths[constraint]   = length(model.configuration.rest_positions[edge.second] - model.configuration.rest_positions[edge.first]);
            host_compliances[constraint]    = constraint < model.topology.edges.size() ? configuration.stretch_compliance : configuration.bending_compliance;
        }

        std::vector<std::uint32_t> host_fixed_vertex_mask(model.particle_count);
        std::vector<Vector3<float>> host_fixed_positions = model.configuration.rest_positions;
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            host_fixed_vertex_mask[fixed_vertex.particle] = 1u;
            host_fixed_positions[fixed_vertex.particle]    = fixed_vertex.position;
        }

        ::cuda::copy_bytes(model.stream, coloring.edges, colored_constraints.values);
        ::cuda::copy_bytes(model.stream, host_first, constraint_first.values);
        ::cuda::copy_bytes(model.stream, host_second, constraint_second.values);
        ::cuda::copy_bytes(model.stream, host_rest_lengths, rest_lengths.values);
        ::cuda::copy_bytes(model.stream, host_compliances, compliances.values);
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

    Solver::StepCache Solver::allocate_step_cache(const Model<float>& model) const {
        StepCache result{.lambdas = simulation::ScalarField<float>(model.stream, constraints.size())};
        simulation::clear(model.stream, result.lambdas);
        return result;
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>&) const {
        return {};
    }

    void Solver::forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters& parameters, State<float>& next_state, StepCache& cache, Workspace&) const {
        simulation::clear(model.stream, cache.lambdas);
        position_dynamics::kernels::predict(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, gravity, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_forces), parameters.masses.values.data(), simulation::view(next_state.positions));
        for (std::uint32_t iteration = 0u; iteration < iteration_count; ++iteration) {
            for (std::size_t color = 0uz; color + 1uz < coloring.offsets.size(); ++color) {
                const std::uint32_t color_offset     = coloring.offsets[color];
                const std::uint32_t constraint_count = coloring.offsets[color + 1uz] - color_offset;
                kernels::project(model.stream, constraint_count, color_offset, 1.0F / (time_step * time_step), colored_constraints.values.data(), constraint_first.values.data(), constraint_second.values.data(), rest_lengths.values.data(), compliances.values.data(), fixed_vertex_mask.values.data(), parameters.masses.values.data(), cache.lambdas.values.data(), simulation::view(next_state.positions));
            }
        }
        position_dynamics::kernels::reconstruct_velocities(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(state.positions), simulation::view(next_state.positions), simulation::view(next_state.velocities));
    }

    std::vector<Edge> Solver::build_constraints(const Model<float>& model) {
        std::vector<Edge> result = model.topology.edges;
        result.reserve(result.size() + model.topology.hinges.size());
        for (const Hinge hinge : model.topology.hinges) result.push_back({.first = hinge.first_opposite, .second = hinge.second_opposite});
        return result;
    }
} // namespace physica::deformables::cloth::solvers::xpbd
