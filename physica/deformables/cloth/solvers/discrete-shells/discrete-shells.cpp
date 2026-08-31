module;

#include "discrete-shells-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.discrete_shells;

import std;

namespace physica::deformables::cloth::solvers::discrete_shells {
    State::State(const ::cuda::stream_ref stream, const std::size_t particle_count) : positions(stream, particle_count), velocities(stream, particle_count), accelerations(stream, particle_count) {}

    Solver::Solver(const Model<float>& model, Configuration configuration) : Solver(model, configuration, build_host_data(model, configuration)) {}

    State Solver::allocate_state(const Model<float>& model) const {
        State result(model.stream, model.particle_count);
        simulation::clear(model.stream, result.positions);
        simulation::clear(model.stream, result.velocities);
        simulation::clear(model.stream, result.accelerations);
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
        const std::size_t edge_count = model.topology.edges.size();
        const std::size_t triangle_count = model.configuration.triangles.size();
        const std::size_t hinge_count = model.topology.hinges.size();
        StepCache result{
            .position_predictor = simulation::VectorField<float>(model.stream, model.particle_count),
            .velocity_predictor = simulation::VectorField<float>(model.stream, model.particle_count),
            .edge_length_conditions = simulation::ScalarField<float>(model.stream, edge_count),
            .edge_energies = simulation::ScalarField<float>(model.stream, edge_count),
            .edge_energy_gradients = simulation::VectorField<float>(model.stream, 2uz * edge_count),
            .edge_energy_hessians = simulation::ScalarField<float>(model.stream, 36uz * edge_count),
            .triangle_area_conditions = simulation::ScalarField<float>(model.stream, triangle_count),
            .triangle_energies = simulation::ScalarField<float>(model.stream, triangle_count),
            .triangle_energy_gradients = simulation::VectorField<float>(model.stream, 3uz * triangle_count),
            .triangle_energy_hessians = simulation::ScalarField<float>(model.stream, 81uz * triangle_count),
            .previous_hinge_angles = simulation::ScalarField<float>(model.stream, hinge_count),
            .hinge_angles = simulation::ScalarField<float>(model.stream, hinge_count),
            .hinge_angle_deltas = simulation::ScalarField<float>(model.stream, hinge_count),
            .hinge_angle_rates = simulation::ScalarField<float>(model.stream, hinge_count),
            .hinge_energies = simulation::ScalarField<float>(model.stream, hinge_count),
            .hinge_damping_potentials = simulation::ScalarField<float>(model.stream, hinge_count),
            .hinge_angle_gradients = simulation::VectorField<float>(model.stream, 4uz * hinge_count),
            .hinge_angle_hessians = simulation::ScalarField<float>(model.stream, 144uz * hinge_count),
            .hinge_energy_gradients = simulation::VectorField<float>(model.stream, 4uz * hinge_count),
            .hinge_energy_hessians = simulation::ScalarField<float>(model.stream, 144uz * hinge_count),
            .hinge_damping_residuals = simulation::VectorField<float>(model.stream, 4uz * hinge_count),
            .hinge_damping_jacobians = simulation::ScalarField<float>(model.stream, 144uz * hinge_count),
            .energy_gradient = simulation::VectorField<float>(model.stream, model.particle_count),
            .damping_residual = simulation::VectorField<float>(model.stream, model.particle_count),
            .residual = simulation::VectorField<float>(model.stream, model.particle_count),
            .energy_hessian = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .damping_jacobian = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .unregularized_system = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .minimum_gershgorin_bound = simulation::ScalarField<double>(model.stream, 1uz),
            .regularization_shift = simulation::ScalarField<float>(model.stream, 1uz),
            .incremental_potential = simulation::ScalarField<double>(model.stream, 1uz),
            .directional_derivative = simulation::ScalarField<double>(model.stream, 1uz),
            .line_search_potentials = simulation::ScalarField<double>(model.stream, line_search_candidate_count),
            .accepted_step_size = simulation::ScalarField<float>(model.stream, 1uz),
            .accepted_candidate = simulation::ScalarField<std::uint32_t>(model.stream, 1uz),
        };
        simulation::clear(model.stream, result.minimum_gershgorin_bound);
        simulation::clear(model.stream, result.regularization_shift);
        simulation::clear(model.stream, result.incremental_potential);
        simulation::clear(model.stream, result.directional_derivative);
        simulation::clear(model.stream, result.line_search_potentials);
        simulation::clear(model.stream, result.accepted_step_size);
        simulation::clear(model.stream, result.accepted_candidate);
        return result;
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>& model) const {
        block_pcg::BlockCsrMatrix system(model, pattern.row_offsets, pattern.column_indices);
        block_pcg::Solver::Workspace pcg = block_solver.allocate_workspace(model, system);
        return {
            .system = std::move(system),
            .right_hand_side = simulation::VectorField<float>(model.stream, model.particle_count),
            .newton_direction = simulation::VectorField<float>(model.stream, model.particle_count),
            .gershgorin_lower_bounds = simulation::ScalarField<float>(model.stream, model.particle_count),
            .pcg = std::move(pcg),
        };
    }

    void Solver::forward(const Model<float>& model, const State& state, const Control<float>& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
        const std::uint32_t particle_count = static_cast<std::uint32_t>(model.particle_count);
        const std::uint32_t edge_count = static_cast<std::uint32_t>(model.topology.edges.size());
        const std::uint32_t triangle_count = static_cast<std::uint32_t>(model.configuration.triangles.size());
        const std::uint32_t hinge_count = static_cast<std::uint32_t>(model.topology.hinges.size());
        simulation::clear(model.stream, cache.accepted_step_size);
        simulation::clear(model.stream, cache.accepted_candidate);
        simulation::clear(model.stream, cache.directional_derivative);
        simulation::clear(model.stream, cache.line_search_potentials);
        kernels::evaluate_hinge_angles(model.stream, hinge_count, model.topology.device.hinges.edge_first.values.data(), model.topology.device.hinges.edge_second.values.data(), model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), simulation::view(state.positions), cache.previous_hinge_angles.values.data());
        kernels::prepare_newmark(model.stream, particle_count, time_step, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(state.accelerations), simulation::view(cache.position_predictor), simulation::view(cache.velocity_predictor), simulation::view(next_state.positions));

        for (std::uint32_t iteration = 0u; iteration < newton_iteration_count; ++iteration) {
            evaluate_system(model, control, parameters, next_state.positions, cache, workspace);
            if (particle_count == 0u) continue;
            kernels::negate(model.stream, particle_count, simulation::view(cache.residual), simulation::view(workspace.right_hand_side));
            block_solver.solve(model, workspace.system, workspace.right_hand_side, fixed_vertex_mask, workspace.newton_direction, workspace.pcg);
            kernels::evaluate_directional_derivative(model.stream, particle_count, fixed_vertex_mask.values.data(), simulation::view(cache.residual), simulation::view(workspace.newton_direction), cache.directional_derivative.values.data());
            evaluate_potential(model, control, parameters, next_state.positions, cache);
            kernels::evaluate_candidate_potentials(model.stream, line_search_candidate_count, particle_count, edge_count, triangle_count, hinge_count, time_step, length_stiffness, area_stiffness, bending_stiffness, bending_damping, gravity, line_search_steps.values.data(), model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), model.topology.device.hinges.edge_first.values.data(), model.topology.device.hinges.edge_second.values.data(), model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), edge_rest_lengths.values.data(), triangle_rest_areas.values.data(), hinge_rest_angles.values.data(), cache.previous_hinge_angles.values.data(), hinge_weights.values.data(), parameters.masses.values.data(), simulation::view(control.external_forces), simulation::view(cache.position_predictor), simulation::view(next_state.positions), simulation::view(workspace.newton_direction), cache.line_search_potentials.values.data());
            kernels::select_step_size(model.stream, line_search_candidate_count, armijo_coefficient, line_search_steps.values.data(), cache.incremental_potential.values.data(), cache.directional_derivative.values.data(), cache.line_search_potentials.values.data(), cache.accepted_step_size.values.data(), cache.accepted_candidate.values.data(), cache.incremental_potential.values.data());
            kernels::update_positions(model.stream, particle_count, cache.accepted_step_size.values.data(), simulation::view(workspace.newton_direction), simulation::view(next_state.positions));
        }

        evaluate_system(model, control, parameters, next_state.positions, cache, workspace);
        evaluate_potential(model, control, parameters, next_state.positions, cache);
        kernels::reconstruct_newmark(model.stream, particle_count, time_step, simulation::view(cache.position_predictor), simulation::view(cache.velocity_predictor), simulation::view(next_state.positions), simulation::view(next_state.velocities), simulation::view(next_state.accelerations));
    }

    Solver::HostData Solver::build_host_data(const Model<float>& model, const Configuration& configuration) {
        const std::size_t edge_count = model.topology.edges.size();
        const std::size_t triangle_count = model.configuration.triangles.size();
        const std::size_t hinge_count = model.topology.hinges.size();
        HostData result{
            .pattern = {.row_offsets = {}, .column_indices = {}, .energy_hessian_contribution_offsets = {}, .energy_hessian_contributions = {}, .damping_jacobian_contribution_offsets = {}, .damping_jacobian_contributions = {}},
            .edge_rest_lengths = std::vector<float>(edge_count),
            .triangle_rest_areas = std::vector<float>(triangle_count),
            .hinge_rest_angles = std::vector<float>(hinge_count),
            .hinge_weights = std::vector<float>(hinge_count),
            .fixed_vertex_mask = std::vector<std::uint32_t>(model.particle_count),
            .fixed_positions = model.configuration.rest_positions,
            .line_search_steps = std::vector<float>(configuration.line_search_candidate_count),
        };

        for (std::size_t edge = 0uz; edge < edge_count; ++edge) {
            const Edge vertices = model.topology.edges[edge];
            result.edge_rest_lengths[edge] = length(model.configuration.rest_positions[vertices.second] - model.configuration.rest_positions[vertices.first]);
        }
        for (std::size_t triangle = 0uz; triangle < triangle_count; ++triangle) {
            const Triangle vertices = model.configuration.triangles[triangle];
            result.triangle_rest_areas[triangle] = 0.5F * length(cross(model.configuration.rest_positions[vertices.second] - model.configuration.rest_positions[vertices.first], model.configuration.rest_positions[vertices.third] - model.configuration.rest_positions[vertices.first]));
        }
        for (std::size_t hinge = 0uz; hinge < hinge_count; ++hinge) {
            const Hinge vertices = model.topology.hinges[hinge];
            result.hinge_rest_angles[hinge] = signed_dihedral(model.configuration.rest_positions, vertices);
            const float edge_length = result.edge_rest_lengths[vertices.edge];
            result.hinge_weights[hinge] = 3.0F * edge_length * edge_length / (result.triangle_rest_areas[vertices.first_triangle] + result.triangle_rest_areas[vertices.second_triangle]);
        }

        std::vector<std::set<std::uint32_t>> rows(model.particle_count);
        for (std::uint32_t particle = 0u; particle < model.particle_count; ++particle) rows[particle].insert(particle);
        for (const Edge edge : model.topology.edges) {
            const std::array vertices{edge.first, edge.second};
            for (const std::uint32_t row : vertices)
                for (const std::uint32_t column : vertices) rows[row].insert(column);
        }
        for (const Triangle triangle : model.configuration.triangles) {
            const std::array vertices{triangle.first, triangle.second, triangle.third};
            for (const std::uint32_t row : vertices)
                for (const std::uint32_t column : vertices) rows[row].insert(column);
        }
        for (const Hinge hinge : model.topology.hinges) {
            const std::array vertices{hinge.edge_first, hinge.edge_second, hinge.first_opposite, hinge.second_opposite};
            for (const std::uint32_t row : vertices)
                for (const std::uint32_t column : vertices) rows[row].insert(column);
        }
        result.pattern.row_offsets.resize(model.particle_count + 1uz);
        for (std::size_t row = 0uz; row < rows.size(); ++row) {
            result.pattern.row_offsets[row + 1uz] = result.pattern.row_offsets[row] + static_cast<std::uint32_t>(rows[row].size());
            result.pattern.column_indices.insert(result.pattern.column_indices.end(), rows[row].begin(), rows[row].end());
        }

        std::vector<std::vector<std::uint32_t>> energy_contributions(result.pattern.column_indices.size());
        std::vector<std::vector<std::uint32_t>> damping_contributions(result.pattern.column_indices.size());
        for (std::uint32_t edge = 0u; edge < edge_count; ++edge) {
            const std::array vertices{model.topology.edges[edge].first, model.topology.edges[edge].second};
            for (std::uint32_t local_row = 0u; local_row < 2u; ++local_row)
                for (std::uint32_t local_column = 0u; local_column < 2u; ++local_column) energy_contributions[find_block(result.pattern, vertices[local_row], vertices[local_column])].push_back(4u * edge + 2u * local_row + local_column);
        }
        const std::uint32_t triangle_block_offset = static_cast<std::uint32_t>(4uz * edge_count);
        for (std::uint32_t triangle = 0u; triangle < triangle_count; ++triangle) {
            const Triangle element = model.configuration.triangles[triangle];
            const std::array vertices{element.first, element.second, element.third};
            for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row)
                for (std::uint32_t local_column = 0u; local_column < 3u; ++local_column) energy_contributions[find_block(result.pattern, vertices[local_row], vertices[local_column])].push_back(triangle_block_offset + 9u * triangle + 3u * local_row + local_column);
        }
        const std::uint32_t hinge_block_offset = triangle_block_offset + static_cast<std::uint32_t>(9uz * triangle_count);
        for (std::uint32_t hinge = 0u; hinge < hinge_count; ++hinge) {
            const Hinge element = model.topology.hinges[hinge];
            const std::array vertices{element.edge_first, element.edge_second, element.first_opposite, element.second_opposite};
            for (std::uint32_t local_row = 0u; local_row < 4u; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < 4u; ++local_column) {
                    const std::uint32_t block = find_block(result.pattern, vertices[local_row], vertices[local_column]);
                    energy_contributions[block].push_back(hinge_block_offset + 16u * hinge + 4u * local_row + local_column);
                    damping_contributions[block].push_back(16u * hinge + 4u * local_row + local_column);
                }
            }
        }

        result.pattern.energy_hessian_contribution_offsets.resize(energy_contributions.size() + 1uz);
        result.pattern.damping_jacobian_contribution_offsets.resize(damping_contributions.size() + 1uz);
        for (std::size_t block = 0uz; block < energy_contributions.size(); ++block) {
            result.pattern.energy_hessian_contribution_offsets[block + 1uz] = result.pattern.energy_hessian_contribution_offsets[block] + static_cast<std::uint32_t>(energy_contributions[block].size());
            result.pattern.energy_hessian_contributions.insert(result.pattern.energy_hessian_contributions.end(), energy_contributions[block].begin(), energy_contributions[block].end());
            result.pattern.damping_jacobian_contribution_offsets[block + 1uz] = result.pattern.damping_jacobian_contribution_offsets[block] + static_cast<std::uint32_t>(damping_contributions[block].size());
            result.pattern.damping_jacobian_contributions.insert(result.pattern.damping_jacobian_contributions.end(), damping_contributions[block].begin(), damping_contributions[block].end());
        }
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            result.fixed_vertex_mask[fixed_vertex.particle] = 1u;
            result.fixed_positions[fixed_vertex.particle] = fixed_vertex.position;
        }
        for (std::uint32_t candidate = 0u; candidate < configuration.line_search_candidate_count; ++candidate) result.line_search_steps[candidate] = std::pow(configuration.line_search_contraction, static_cast<float>(candidate));
        return result;
    }

    std::uint32_t Solver::find_block(const Pattern& pattern, const std::uint32_t row, const std::uint32_t column) {
        const auto first = pattern.column_indices.begin() + pattern.row_offsets[row];
        const auto last = pattern.column_indices.begin() + pattern.row_offsets[row + 1u];
        return static_cast<std::uint32_t>(std::lower_bound(first, last, column) - pattern.column_indices.begin());
    }

    float Solver::signed_dihedral(const std::span<const Vector3<float>> positions, const Hinge& hinge) {
        const Vector3<float> edge_first = positions[hinge.edge_first];
        const Vector3<float> edge_second = positions[hinge.edge_second];
        const Vector3<float> first_opposite = positions[hinge.first_opposite];
        const Vector3<float> second_opposite = positions[hinge.second_opposite];
        const Vector3<float> edge = normalized(edge_second - edge_first);
        const Vector3<float> first_normal = normalized(cross(edge_second - edge_first, first_opposite - edge_first));
        const Vector3<float> second_normal = normalized(cross(edge_first - edge_second, second_opposite - edge_second));
        return std::atan2(dot(cross(first_normal, second_normal), edge), dot(first_normal, second_normal));
    }

    Solver::Solver(const Model<float>& model, const Configuration& configuration, HostData host_data)
        : time_step(configuration.time_step),
          newton_iteration_count(configuration.newton_iteration_count),
          line_search_candidate_count(configuration.line_search_candidate_count),
          gravity(configuration.gravity),
          length_stiffness(configuration.length_stiffness),
          area_stiffness(configuration.area_stiffness),
          bending_stiffness(configuration.bending_stiffness),
          bending_damping(configuration.bending_damping),
          hessian_positive_margin(configuration.hessian_positive_margin),
          armijo_coefficient(configuration.armijo_coefficient),
          pattern(std::move(host_data.pattern)),
          block_solver(model, {.iteration_count = configuration.pcg_iteration_count}),
          edge_rest_lengths(model.stream, host_data.edge_rest_lengths.size()),
          triangle_rest_areas(model.stream, host_data.triangle_rest_areas.size()),
          hinge_rest_angles(model.stream, host_data.hinge_rest_angles.size()),
          hinge_weights(model.stream, host_data.hinge_weights.size()),
          energy_hessian_contribution_offsets(model.stream, pattern.energy_hessian_contribution_offsets.size()),
          energy_hessian_contributions(model.stream, pattern.energy_hessian_contributions.size()),
          damping_jacobian_contribution_offsets(model.stream, pattern.damping_jacobian_contribution_offsets.size()),
          damping_jacobian_contributions(model.stream, pattern.damping_jacobian_contributions.size()),
          fixed_vertex_mask(model.stream, host_data.fixed_vertex_mask.size()),
          fixed_positions(model.stream, host_data.fixed_positions.size()),
          line_search_steps(model.stream, host_data.line_search_steps.size()) {
        ::cuda::copy_bytes(model.stream, host_data.edge_rest_lengths, edge_rest_lengths.values);
        ::cuda::copy_bytes(model.stream, host_data.triangle_rest_areas, triangle_rest_areas.values);
        ::cuda::copy_bytes(model.stream, host_data.hinge_rest_angles, hinge_rest_angles.values);
        ::cuda::copy_bytes(model.stream, host_data.hinge_weights, hinge_weights.values);
        ::cuda::copy_bytes(model.stream, pattern.energy_hessian_contribution_offsets, energy_hessian_contribution_offsets.values);
        ::cuda::copy_bytes(model.stream, pattern.energy_hessian_contributions, energy_hessian_contributions.values);
        ::cuda::copy_bytes(model.stream, pattern.damping_jacobian_contribution_offsets, damping_jacobian_contribution_offsets.values);
        ::cuda::copy_bytes(model.stream, pattern.damping_jacobian_contributions, damping_jacobian_contributions.values);
        ::cuda::copy_bytes(model.stream, host_data.fixed_vertex_mask, fixed_vertex_mask.values);
        simulation::upload(model.stream, host_data.fixed_positions, fixed_positions);
        ::cuda::copy_bytes(model.stream, host_data.line_search_steps, line_search_steps.values);
        model.stream.sync();
    }

    void Solver::evaluate_system(const Model<float>& model, const Control<float>& control, const Parameters& parameters, const simulation::VectorField<float>& positions, StepCache& cache, Workspace& workspace) const {
        const std::uint32_t particle_count = static_cast<std::uint32_t>(model.particle_count);
        const std::uint32_t edge_count = static_cast<std::uint32_t>(model.topology.edges.size());
        const std::uint32_t triangle_count = static_cast<std::uint32_t>(model.configuration.triangles.size());
        const std::uint32_t hinge_count = static_cast<std::uint32_t>(model.topology.hinges.size());
        const float mass_coefficient = 4.0F / (time_step * time_step);
        kernels::evaluate_edges(model.stream, edge_count, length_stiffness, model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), edge_rest_lengths.values.data(), simulation::view(positions), cache.edge_length_conditions.values.data(), cache.edge_energies.values.data(), simulation::view(cache.edge_energy_gradients), cache.edge_energy_hessians.values.data());
        kernels::evaluate_triangles(model.stream, triangle_count, area_stiffness, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), triangle_rest_areas.values.data(), simulation::view(positions), cache.triangle_area_conditions.values.data(), cache.triangle_energies.values.data(), simulation::view(cache.triangle_energy_gradients), cache.triangle_energy_hessians.values.data());
        kernels::evaluate_hinges(model.stream, hinge_count, time_step, bending_stiffness, bending_damping, model.topology.device.hinges.edge_first.values.data(), model.topology.device.hinges.edge_second.values.data(), model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), hinge_rest_angles.values.data(), cache.previous_hinge_angles.values.data(), hinge_weights.values.data(), simulation::view(positions), cache.hinge_angles.values.data(), cache.hinge_angle_deltas.values.data(), cache.hinge_angle_rates.values.data(), cache.hinge_energies.values.data(), cache.hinge_damping_potentials.values.data(), simulation::view(cache.hinge_angle_gradients), cache.hinge_angle_hessians.values.data(), simulation::view(cache.hinge_energy_gradients), cache.hinge_energy_hessians.values.data(), simulation::view(cache.hinge_damping_residuals), cache.hinge_damping_jacobians.values.data());
        kernels::assemble_system(model.stream, particle_count, edge_count, triangle_count, mass_coefficient, gravity, model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), model.topology.device.hinges.edge_first.values.data(), model.topology.device.hinges.edge_second.values.data(), model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), model.topology.device.vertex_edges.offsets.values.data(), model.topology.device.vertex_edges.indices.values.data(), model.topology.device.vertex_triangles.offsets.values.data(), model.topology.device.vertex_triangles.indices.values.data(), model.topology.device.vertex_hinges.offsets.values.data(), model.topology.device.vertex_hinges.indices.values.data(), cache.unregularized_system.row_offsets.values.data(), cache.unregularized_system.column_indices.values.data(), energy_hessian_contribution_offsets.values.data(), energy_hessian_contributions.values.data(), damping_jacobian_contribution_offsets.values.data(), damping_jacobian_contributions.values.data(), parameters.masses.values.data(), simulation::view(control.external_forces), simulation::view(cache.position_predictor), simulation::view(positions), simulation::view(cache.edge_energy_gradients), cache.edge_energy_hessians.values.data(), simulation::view(cache.triangle_energy_gradients), cache.triangle_energy_hessians.values.data(), simulation::view(cache.hinge_energy_gradients), cache.hinge_energy_hessians.values.data(), simulation::view(cache.hinge_damping_residuals), cache.hinge_damping_jacobians.values.data(), simulation::view(cache.energy_gradient), simulation::view(cache.damping_residual), simulation::view(cache.residual), cache.energy_hessian.block_values.values.data(), cache.damping_jacobian.block_values.values.data(), cache.unregularized_system.block_values.values.data());
        if (particle_count == 0u) {
            simulation::clear(model.stream, cache.minimum_gershgorin_bound);
            simulation::clear(model.stream, cache.regularization_shift);
            return;
        }
        kernels::compute_gershgorin_bounds(model.stream, particle_count, cache.unregularized_system.row_offsets.values.data(), cache.unregularized_system.column_indices.values.data(), fixed_vertex_mask.values.data(), cache.unregularized_system.block_values.values.data(), workspace.gershgorin_lower_bounds.values.data());
        kernels::reduce_minimum(model.stream, particle_count, workspace.gershgorin_lower_bounds.values.data(), cache.minimum_gershgorin_bound.values.data());
        kernels::choose_regularization(model.stream, hessian_positive_margin, cache.minimum_gershgorin_bound.values.data(), cache.regularization_shift.values.data());
        kernels::build_regularized_system(model.stream, particle_count, cache.unregularized_system.row_offsets.values.data(), cache.unregularized_system.column_indices.values.data(), fixed_vertex_mask.values.data(), cache.regularization_shift.values.data(), cache.unregularized_system.block_values.values.data(), workspace.system.block_values.values.data());
    }

    void Solver::evaluate_potential(const Model<float>& model, const Control<float>& control, const Parameters& parameters, const simulation::VectorField<float>& positions, StepCache& cache) const {
        kernels::evaluate_potential(model.stream, static_cast<std::uint32_t>(model.particle_count), static_cast<std::uint32_t>(model.topology.edges.size()), static_cast<std::uint32_t>(model.configuration.triangles.size()), static_cast<std::uint32_t>(model.topology.hinges.size()), time_step, length_stiffness, area_stiffness, bending_stiffness, bending_damping, gravity, model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), model.topology.device.hinges.edge_first.values.data(), model.topology.device.hinges.edge_second.values.data(), model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), edge_rest_lengths.values.data(), triangle_rest_areas.values.data(), hinge_rest_angles.values.data(), cache.previous_hinge_angles.values.data(), hinge_weights.values.data(), parameters.masses.values.data(), simulation::view(control.external_forces), simulation::view(cache.position_predictor), simulation::view(positions), cache.incremental_potential.values.data());
    }
} // namespace physica::deformables::cloth::solvers::discrete_shells
