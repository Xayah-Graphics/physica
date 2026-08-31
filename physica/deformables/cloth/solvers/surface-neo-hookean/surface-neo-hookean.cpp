module;

#include "surface-neo-hookean-kernels.h"
#include "../position-dynamics-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.surface_neo_hookean;

import std;

namespace physica::deformables::cloth::solvers::surface_neo_hookean {
    Solver::Solver(const Model<float>& model, Configuration configuration) : Solver(model, configuration, build_host_data(model, configuration)) {}

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

    Solver::Parameters Solver::allocate_parameters(const Model<float>&) const {
        return {};
    }

    Solver::StepCache Solver::allocate_step_cache(const Model<float>& model) const {
        const std::size_t triangle_count = model.configuration.triangles.size();
        StepCache result{
            .predicted_positions                       = simulation::VectorField<float>(model.stream, model.particle_count),
            .deformation_gradient_first_columns       = simulation::VectorField<float>(model.stream, triangle_count),
            .deformation_gradient_second_columns      = simulation::VectorField<float>(model.stream, triangle_count),
            .surface_jacobians                         = simulation::ScalarField<float>(model.stream, triangle_count),
            .log_surface_jacobians                     = simulation::ScalarField<float>(model.stream, triangle_count),
            .triangle_energies                         = simulation::ScalarField<float>(model.stream, triangle_count),
            .triangle_gradients                        = simulation::VectorField<float>(model.stream, 3uz * triangle_count),
            .triangle_hessians                         = simulation::ScalarField<float>(model.stream, 81uz * triangle_count),
            .gradient                                  = simulation::VectorField<float>(model.stream, model.particle_count),
            .hessian                                   = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .regularization_shift                      = simulation::ScalarField<float>(model.stream, 1uz),
            .minimum_gershgorin_bound                  = simulation::ScalarField<double>(model.stream, 1uz),
            .accepted_step_size                        = simulation::ScalarField<float>(model.stream, 1uz),
            .accepted_candidate                        = simulation::ScalarField<std::uint32_t>(model.stream, 1uz),
            .incremental_potential                     = simulation::ScalarField<double>(model.stream, 1uz),
            .directional_derivative                    = simulation::ScalarField<double>(model.stream, 1uz),
            .triangle_domain_steps                     = simulation::ScalarField<float>(model.stream, triangle_count),
            .maximum_domain_step                       = simulation::ScalarField<float>(model.stream, 1uz),
            .line_search_potentials                    = simulation::ScalarField<double>(model.stream, line_search_candidate_count),
        };
        simulation::clear(model.stream, result.regularization_shift);
        simulation::clear(model.stream, result.minimum_gershgorin_bound);
        simulation::clear(model.stream, result.accepted_step_size);
        simulation::clear(model.stream, result.accepted_candidate);
        simulation::clear(model.stream, result.incremental_potential);
        simulation::clear(model.stream, result.directional_derivative);
        simulation::clear(model.stream, result.maximum_domain_step);
        simulation::clear(model.stream, result.line_search_potentials);
        return result;
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>& model) const {
        block_pcg::BlockCsrMatrix regularized_hessian(model, pattern.row_offsets, pattern.column_indices);
        block_pcg::Solver::Workspace pcg = block_solver.allocate_workspace(model, regularized_hessian);
        return {
            .regularized_hessian     = std::move(regularized_hessian),
            .right_hand_side         = simulation::VectorField<float>(model.stream, model.particle_count),
            .newton_direction        = simulation::VectorField<float>(model.stream, model.particle_count),
            .gershgorin_lower_bounds = simulation::ScalarField<float>(model.stream, model.particle_count),
            .pcg                      = std::move(pcg),
        };
    }

    void Solver::forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters& parameters, State<float>& next_state, StepCache& cache, Workspace& workspace) const {
        const std::uint32_t particle_count = static_cast<std::uint32_t>(model.particle_count);
        const std::uint32_t triangle_count = static_cast<std::uint32_t>(model.configuration.triangles.size());
        const float inverse_time_step_squared = 1.0F / (time_step * time_step);
        simulation::clear(model.stream, cache.accepted_step_size);
        simulation::clear(model.stream, cache.accepted_candidate);
        simulation::clear(model.stream, cache.directional_derivative);
        simulation::clear(model.stream, cache.line_search_potentials);
        if (particle_count != 0u) position_dynamics::kernels::predict(model.stream, particle_count, time_step, gravity, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_forces), masses.values.data(), simulation::view(cache.predicted_positions));
        simulation::copy(model.stream, cache.predicted_positions, next_state.positions);

        for (std::uint32_t iteration = 0u; iteration < newton_iteration_count; ++iteration) {
            evaluate_system(model, parameters, next_state.positions, cache, workspace);
            if (particle_count == 0u) continue;
            kernels::negate(model.stream, particle_count, simulation::view(cache.gradient), simulation::view(workspace.right_hand_side));
            block_solver.solve(model, workspace.regularized_hessian, workspace.right_hand_side, fixed_vertex_mask, workspace.newton_direction, workspace.pcg);
            kernels::evaluate_directional_derivative(model.stream, particle_count, fixed_vertex_mask.values.data(), simulation::view(cache.gradient), simulation::view(workspace.newton_direction), cache.directional_derivative.values.data());
            kernels::evaluate_domain_steps(model.stream, triangle_count, domain_safety, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), simulation::view(material_u_gradients), simulation::view(material_v_gradients), simulation::view(next_state.positions), simulation::view(workspace.newton_direction), cache.triangle_domain_steps.values.data());
            kernels::reduce_domain_step(model.stream, triangle_count, cache.triangle_domain_steps.values.data(), cache.maximum_domain_step.values.data());
            kernels::evaluate_potential(model.stream, particle_count, triangle_count, inverse_time_step_squared, lame_lambda, lame_mu, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), simulation::view(material_u_gradients), simulation::view(material_v_gradients), triangle_weights.values.data(), masses.values.data(), simulation::view(cache.predicted_positions), simulation::view(next_state.positions), cache.incremental_potential.values.data());
            kernels::evaluate_candidate_potentials(model.stream, line_search_candidate_count, particle_count, triangle_count, inverse_time_step_squared, lame_lambda, lame_mu, line_search_contractions.values.data(), cache.maximum_domain_step.values.data(), model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), simulation::view(material_u_gradients), simulation::view(material_v_gradients), triangle_weights.values.data(), masses.values.data(), simulation::view(cache.predicted_positions), simulation::view(next_state.positions), simulation::view(workspace.newton_direction), cache.line_search_potentials.values.data());
            kernels::select_step_size(model.stream, line_search_candidate_count, armijo_coefficient, line_search_contractions.values.data(), cache.maximum_domain_step.values.data(), cache.incremental_potential.values.data(), cache.directional_derivative.values.data(), cache.line_search_potentials.values.data(), cache.accepted_step_size.values.data(), cache.accepted_candidate.values.data(), cache.incremental_potential.values.data());
            kernels::update_positions(model.stream, particle_count, cache.accepted_step_size.values.data(), simulation::view(workspace.newton_direction), simulation::view(next_state.positions));
        }

        evaluate_system(model, parameters, next_state.positions, cache, workspace);
        kernels::evaluate_potential(model.stream, particle_count, triangle_count, inverse_time_step_squared, lame_lambda, lame_mu, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), simulation::view(material_u_gradients), simulation::view(material_v_gradients), triangle_weights.values.data(), masses.values.data(), simulation::view(cache.predicted_positions), simulation::view(next_state.positions), cache.incremental_potential.values.data());
        if (particle_count != 0u) position_dynamics::kernels::reconstruct_velocities(model.stream, particle_count, time_step, simulation::view(state.positions), simulation::view(next_state.positions), simulation::view(next_state.velocities));
    }

    Solver::HostData Solver::build_host_data(const Model<float>& model, const Configuration& configuration) {
        const std::size_t triangle_count = model.configuration.triangles.size();
        HostData result{
            .pattern = {.row_offsets = {}, .column_indices = {}, .block_contribution_offsets = {}, .block_contributions = {}},
            .material_u_gradients = std::vector<Vector3<float>>(triangle_count),
            .material_v_gradients = std::vector<Vector3<float>>(triangle_count),
            .triangle_weights = std::vector<float>(triangle_count),
            .fixed_vertex_mask = std::vector<std::uint32_t>(model.particle_count),
            .fixed_positions = model.configuration.rest_positions,
            .line_search_contractions = std::vector<float>(configuration.line_search_candidate_count),
        };

        std::vector<std::set<std::uint32_t>> rows(model.particle_count);
        for (std::uint32_t particle = 0u; particle < model.particle_count; ++particle) rows[particle].insert(particle);
        for (const Triangle triangle : model.configuration.triangles) {
            const std::array vertices{triangle.first, triangle.second, triangle.third};
            for (const std::uint32_t row : vertices)
                for (const std::uint32_t column : vertices) rows[row].insert(column);
        }
        result.pattern.row_offsets.resize(model.particle_count + 1uz);
        for (std::size_t row = 0uz; row < rows.size(); ++row) {
            result.pattern.row_offsets[row + 1uz] = result.pattern.row_offsets[row] + static_cast<std::uint32_t>(rows[row].size());
            result.pattern.column_indices.insert(result.pattern.column_indices.end(), rows[row].begin(), rows[row].end());
        }

        std::vector<std::vector<std::uint32_t>> block_contributions(result.pattern.column_indices.size());
        for (std::uint32_t triangle_index = 0u; triangle_index < triangle_count; ++triangle_index) {
            const Triangle triangle = model.configuration.triangles[triangle_index];
            const TriangleMaterialCoordinates<float> material = model.configuration.material_coordinates[triangle_index];
            const float material_10_u = material.second.u - material.first.u;
            const float material_10_v = material.second.v - material.first.v;
            const float material_20_u = material.third.u - material.first.u;
            const float material_20_v = material.third.v - material.first.v;
            const float determinant   = material_10_u * material_20_v - material_20_u * material_10_v;
            const float inverse_00    = material_20_v / determinant;
            const float inverse_01    = -material_20_u / determinant;
            const float inverse_10    = -material_10_v / determinant;
            const float inverse_11    = material_10_u / determinant;
            result.material_u_gradients[triangle_index] = {.x = -inverse_00 - inverse_10, .y = inverse_00, .z = inverse_10};
            result.material_v_gradients[triangle_index] = {.x = -inverse_01 - inverse_11, .y = inverse_01, .z = inverse_11};
            result.triangle_weights[triangle_index]     = 0.5F * std::abs(determinant) * configuration.thickness;

            const std::array vertices{triangle.first, triangle.second, triangle.third};
            for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < 3u; ++local_column) {
                    const auto first = result.pattern.column_indices.begin() + result.pattern.row_offsets[vertices[local_row]];
                    const auto last  = result.pattern.column_indices.begin() + result.pattern.row_offsets[vertices[local_row] + 1u];
                    const std::uint32_t block = static_cast<std::uint32_t>(std::lower_bound(first, last, vertices[local_column]) - result.pattern.column_indices.begin());
                    block_contributions[block].push_back(9u * triangle_index + 3u * local_row + local_column);
                }
            }
        }
        result.pattern.block_contribution_offsets.resize(block_contributions.size() + 1uz);
        for (std::size_t block = 0uz; block < block_contributions.size(); ++block) {
            result.pattern.block_contribution_offsets[block + 1uz] = result.pattern.block_contribution_offsets[block] + static_cast<std::uint32_t>(block_contributions[block].size());
            result.pattern.block_contributions.insert(result.pattern.block_contributions.end(), block_contributions[block].begin(), block_contributions[block].end());
        }

        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            result.fixed_vertex_mask[fixed_vertex.particle] = 1u;
            result.fixed_positions[fixed_vertex.particle]   = fixed_vertex.position;
        }
        for (std::uint32_t candidate = 0u; candidate < configuration.line_search_candidate_count; ++candidate) result.line_search_contractions[candidate] = std::pow(configuration.line_search_contraction, static_cast<float>(candidate));
        return result;
    }

    Solver::Solver(const Model<float>& model, const Configuration& configuration, HostData host_data)
        : time_step(configuration.time_step),
          newton_iteration_count(configuration.newton_iteration_count),
          line_search_candidate_count(configuration.line_search_candidate_count),
          gravity(configuration.gravity),
          lame_lambda(configuration.young_modulus * configuration.poisson_ratio / (1.0F - configuration.poisson_ratio * configuration.poisson_ratio)),
          lame_mu(configuration.young_modulus / (2.0F * (1.0F + configuration.poisson_ratio))),
          hessian_positive_margin(configuration.hessian_positive_margin),
          armijo_coefficient(configuration.armijo_coefficient),
          domain_safety(configuration.domain_safety),
          pattern(std::move(host_data.pattern)),
          block_solver(model, {.iteration_count = configuration.pcg_iteration_count}),
          masses(model.stream, configuration.masses.size()),
          material_u_gradients(model.stream, host_data.material_u_gradients.size()),
          material_v_gradients(model.stream, host_data.material_v_gradients.size()),
          triangle_weights(model.stream, host_data.triangle_weights.size()),
          block_contribution_offsets(model.stream, pattern.block_contribution_offsets.size()),
          block_contributions(model.stream, pattern.block_contributions.size()),
          fixed_vertex_mask(model.stream, host_data.fixed_vertex_mask.size()),
          fixed_positions(model.stream, host_data.fixed_positions.size()),
          line_search_contractions(model.stream, host_data.line_search_contractions.size()) {
        ::cuda::copy_bytes(model.stream, configuration.masses, masses.values);
        simulation::upload(model.stream, host_data.material_u_gradients, material_u_gradients);
        simulation::upload(model.stream, host_data.material_v_gradients, material_v_gradients);
        ::cuda::copy_bytes(model.stream, host_data.triangle_weights, triangle_weights.values);
        ::cuda::copy_bytes(model.stream, pattern.block_contribution_offsets, block_contribution_offsets.values);
        ::cuda::copy_bytes(model.stream, pattern.block_contributions, block_contributions.values);
        ::cuda::copy_bytes(model.stream, host_data.fixed_vertex_mask, fixed_vertex_mask.values);
        simulation::upload(model.stream, host_data.fixed_positions, fixed_positions);
        ::cuda::copy_bytes(model.stream, host_data.line_search_contractions, line_search_contractions.values);
        model.stream.sync();
    }

    void Solver::evaluate_system(const Model<float>& model, const Parameters&, const simulation::VectorField<float>& positions, StepCache& cache, Workspace& workspace) const {
        const std::uint32_t particle_count = static_cast<std::uint32_t>(model.particle_count);
        const std::uint32_t triangle_count = static_cast<std::uint32_t>(model.configuration.triangles.size());
        const float inverse_time_step_squared = 1.0F / (time_step * time_step);
        kernels::evaluate_elements(model.stream, triangle_count, lame_lambda, lame_mu, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), simulation::view(material_u_gradients), simulation::view(material_v_gradients), triangle_weights.values.data(), simulation::view(positions), simulation::view(cache.deformation_gradient_first_columns), simulation::view(cache.deformation_gradient_second_columns), cache.surface_jacobians.values.data(), cache.log_surface_jacobians.values.data(), cache.triangle_energies.values.data(), simulation::view(cache.triangle_gradients), cache.triangle_hessians.values.data());
        kernels::assemble_incremental_system(model.stream, particle_count, inverse_time_step_squared, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), model.topology.device.vertex_triangles.offsets.values.data(), model.topology.device.vertex_triangles.indices.values.data(), cache.hessian.row_offsets.values.data(), cache.hessian.column_indices.values.data(), block_contribution_offsets.values.data(), block_contributions.values.data(), masses.values.data(), simulation::view(cache.predicted_positions), simulation::view(positions), simulation::view(cache.triangle_gradients), cache.triangle_hessians.values.data(), simulation::view(cache.gradient), cache.hessian.block_values.values.data());
        if (particle_count == 0u) {
            simulation::clear(model.stream, cache.regularization_shift);
            simulation::clear(model.stream, cache.minimum_gershgorin_bound);
            return;
        }
        kernels::compute_gershgorin_bounds(model.stream, particle_count, cache.hessian.row_offsets.values.data(), cache.hessian.column_indices.values.data(), fixed_vertex_mask.values.data(), cache.hessian.block_values.values.data(), workspace.gershgorin_lower_bounds.values.data());
        kernels::reduce_minimum(model.stream, particle_count, workspace.gershgorin_lower_bounds.values.data(), cache.minimum_gershgorin_bound.values.data());
        kernels::choose_regularization(model.stream, hessian_positive_margin, cache.minimum_gershgorin_bound.values.data(), cache.regularization_shift.values.data());
        kernels::build_regularized_hessian(model.stream, particle_count, cache.hessian.row_offsets.values.data(), cache.hessian.column_indices.values.data(), fixed_vertex_mask.values.data(), cache.regularization_shift.values.data(), cache.hessian.block_values.values.data(), workspace.regularized_hessian.block_values.values.data());
    }
} // namespace physica::deformables::cloth::solvers::surface_neo_hookean
