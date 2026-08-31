module;

#include "choi-ko-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.choi_ko;

import std;

namespace physica::deformables::cloth::solvers::choi_ko {
    State::State(const ::cuda::stream_ref stream, const std::size_t particle_count)
        : positions(stream, particle_count), velocities(stream, particle_count), previous_positions(stream, particle_count), previous_velocities(stream, particle_count) {}

    Solver::Solver(const Model<float>& model, Configuration configuration)
        : time_step(configuration.time_step),
          gravity(configuration.gravity),
          stretch_u_stiffness(configuration.stretch_u_stiffness),
          stretch_v_stiffness(configuration.stretch_v_stiffness),
          diagonal_u_stiffness(configuration.diagonal_u_stiffness),
          diagonal_v_stiffness(configuration.diagonal_v_stiffness),
          imperfection_stiffness(configuration.imperfection_stiffness),
          stretch_u_damping(configuration.stretch_u_damping),
          stretch_v_damping(configuration.stretch_v_damping),
          diagonal_u_damping(configuration.diagonal_u_damping),
          diagonal_v_damping(configuration.diagonal_v_damping),
          bending_damping(configuration.bending_damping),
          pattern(build_pattern(model)),
          block_solver(model, {.iteration_count = configuration.pcg_iteration_count}),
          triangle_direction_coefficients(model.stream, 4uz * model.configuration.triangles.size()),
          triangle_areas(model.stream, model.configuration.triangles.size()),
          hinge_rest_spans(model.stream, model.topology.hinges.size()),
          hinge_area_sums(model.stream, model.topology.hinges.size()),
          hinge_stiffnesses(model.stream, model.topology.hinges.size()),
          force_contribution_offsets(model.stream, pattern.force_contribution_offsets.size()),
          force_contribution_indices(model.stream, pattern.force_contribution_indices.size()),
          block_contribution_offsets(model.stream, pattern.block_contribution_offsets.size()),
          block_contribution_indices(model.stream, pattern.block_contribution_indices.size()),
          fixed_vertex_mask(model.stream, model.particle_count),
          fixed_positions(model.stream, model.particle_count) {
        constexpr float inverse_square_root_two = 0.70710678118654752440F;
        std::vector<Vector3<float>> host_direction_coefficients(4uz * model.configuration.triangles.size());
        std::vector<float> host_triangle_areas(model.configuration.triangles.size());
        for (std::size_t triangle_index = 0uz; triangle_index < model.configuration.triangles.size(); ++triangle_index) {
            const TriangleMaterialCoordinates<float> coordinates = model.configuration.material_coordinates[triangle_index];
            const float delta_u_first                             = coordinates.second.u - coordinates.first.u;
            const float delta_v_first                             = coordinates.second.v - coordinates.first.v;
            const float delta_u_second                            = coordinates.third.u - coordinates.first.u;
            const float delta_v_second                            = coordinates.third.v - coordinates.first.v;
            const float determinant                               = delta_u_first * delta_v_second - delta_u_second * delta_v_first;
            const float inverse_00                                = delta_v_second / determinant;
            const float inverse_01                                = -delta_u_second / determinant;
            const float inverse_10                                = -delta_v_first / determinant;
            const float inverse_11                                = delta_u_first / determinant;
            const Vector3<float> u_coefficients{.x = -inverse_00 - inverse_10, .y = inverse_00, .z = inverse_10};
            const Vector3<float> v_coefficients{.x = -inverse_01 - inverse_11, .y = inverse_01, .z = inverse_11};
            host_direction_coefficients[4uz * triangle_index]       = u_coefficients;
            host_direction_coefficients[4uz * triangle_index + 1uz] = v_coefficients;
            host_direction_coefficients[4uz * triangle_index + 2uz] = inverse_square_root_two * (u_coefficients - v_coefficients);
            host_direction_coefficients[4uz * triangle_index + 3uz] = inverse_square_root_two * (u_coefficients + v_coefficients);
            host_triangle_areas[triangle_index] = 0.5F * std::abs(determinant);
        }

        std::vector<float> host_hinge_rest_spans(model.topology.hinges.size());
        std::vector<float> host_hinge_area_sums(model.topology.hinges.size());
        std::vector<float> host_hinge_stiffnesses(model.topology.hinges.size());
        for (std::size_t hinge_index = 0uz; hinge_index < model.topology.hinges.size(); ++hinge_index) {
            const Hinge hinge = model.topology.hinges[hinge_index];
            host_hinge_rest_spans[hinge_index] = length(model.configuration.rest_positions[hinge.second_opposite] - model.configuration.rest_positions[hinge.first_opposite]);
            host_hinge_area_sums[hinge_index] = host_triangle_areas[hinge.first_triangle] + host_triangle_areas[hinge.second_triangle];
            const MaterialCoordinate<float> first_coordinate  = material_coordinate(model, hinge.first_triangle, hinge.first_opposite);
            const MaterialCoordinate<float> second_coordinate = material_coordinate(model, hinge.second_triangle, hinge.second_opposite);
            const float delta_u = second_coordinate.u - first_coordinate.u;
            const float delta_v = second_coordinate.v - first_coordinate.v;
            host_hinge_stiffnesses[hinge_index] = std::sqrt((configuration.bend_u_stiffness * configuration.bend_u_stiffness * delta_u * delta_u + configuration.bend_v_stiffness * configuration.bend_v_stiffness * delta_v * delta_v) / (delta_u * delta_u + delta_v * delta_v));
        }

        std::vector<std::uint32_t> host_fixed_vertex_mask(model.particle_count);
        std::vector<Vector3<float>> host_fixed_positions = model.configuration.rest_positions;
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            host_fixed_vertex_mask[fixed_vertex.particle] = 1u;
            host_fixed_positions[fixed_vertex.particle]    = fixed_vertex.position;
        }

        simulation::upload(model.stream, host_direction_coefficients, triangle_direction_coefficients);
        ::cuda::copy_bytes(model.stream, host_triangle_areas, triangle_areas.values);
        ::cuda::copy_bytes(model.stream, host_hinge_rest_spans, hinge_rest_spans.values);
        ::cuda::copy_bytes(model.stream, host_hinge_area_sums, hinge_area_sums.values);
        ::cuda::copy_bytes(model.stream, host_hinge_stiffnesses, hinge_stiffnesses.values);
        ::cuda::copy_bytes(model.stream, pattern.force_contribution_offsets, force_contribution_offsets.values);
        ::cuda::copy_bytes(model.stream, pattern.force_contribution_indices, force_contribution_indices.values);
        ::cuda::copy_bytes(model.stream, pattern.block_contribution_offsets, block_contribution_offsets.values);
        ::cuda::copy_bytes(model.stream, pattern.block_contribution_indices, block_contribution_indices.values);
        ::cuda::copy_bytes(model.stream, host_fixed_vertex_mask, fixed_vertex_mask.values);
        simulation::upload(model.stream, host_fixed_positions, fixed_positions);
        model.stream.sync();
    }

    State Solver::allocate_state(const Model<float>& model) const {
        return State(model.stream, model.particle_count);
    }

    Control<float> Solver::allocate_control(const Model<float>& model) const {
        return Control<float>(model.stream, model.particle_count);
    }

    Solver::Parameters Solver::allocate_parameters(const Model<float>& model) const {
        return {.masses = simulation::ScalarField<float>(model.stream, model.particle_count)};
    }

    Solver::StepCache Solver::allocate_step_cache(const Model<float>& model) const {
        return {
            .triangle_conditions                    = simulation::ScalarField<float>(model.stream, 4uz * model.configuration.triangles.size()),
            .hinge_curvatures                       = simulation::ScalarField<float>(model.stream, model.topology.hinges.size()),
            .hinge_curvature_first_derivatives      = simulation::ScalarField<float>(model.stream, model.topology.hinges.size()),
            .hinge_curvature_second_derivatives     = simulation::ScalarField<float>(model.stream, model.topology.hinges.size()),
            .hinge_responses                        = simulation::ScalarField<float>(model.stream, model.topology.hinges.size()),
            .hinge_response_derivatives             = simulation::ScalarField<float>(model.stream, model.topology.hinges.size()),
            .forces                                 = simulation::VectorField<float>(model.stream, model.particle_count),
            .symmetric_force_position_derivative    = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .force_velocity_derivative              = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .system                                 = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .right_hand_side                        = simulation::VectorField<float>(model.stream, model.particle_count),
            .prescribed_displacement                = simulation::VectorField<float>(model.stream, model.particle_count),
            .solution                               = simulation::VectorField<float>(model.stream, model.particle_count),
            .bdf2_displacement                      = simulation::VectorField<float>(model.stream, model.particle_count),
        };
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>& model) const {
        const std::size_t local_force_count = 3uz * model.configuration.triangles.size() + 2uz * model.topology.hinges.size();
        const std::size_t local_block_count = 9uz * model.configuration.triangles.size() + 4uz * model.topology.hinges.size();
        block_pcg::BlockCsrMatrix matrix(model, pattern.row_offsets, pattern.column_indices);
        return {
            .local_forces                       = simulation::VectorField<float>(model.stream, local_force_count),
            .local_symmetric_force_position_derivatives = simulation::ScalarField<float>(model.stream, 9uz * local_block_count),
            .local_force_velocity_derivatives  = simulation::ScalarField<float>(model.stream, 9uz * local_block_count),
            .system_times_prescribed_displacement = simulation::VectorField<float>(model.stream, model.particle_count),
            .reduced_right_hand_side            = simulation::VectorField<float>(model.stream, model.particle_count),
            .pcg                                = block_solver.allocate_workspace(model, matrix),
        };
    }

    void Solver::forward(const Model<float>& model, const State& state, const Control<float>& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
        kernels::assemble_triangles(model.stream, static_cast<std::uint32_t>(model.configuration.triangles.size()), stretch_u_stiffness, stretch_v_stiffness, diagonal_u_stiffness, diagonal_v_stiffness, stretch_u_damping, stretch_v_damping, diagonal_u_damping, diagonal_v_damping, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), simulation::view(triangle_direction_coefficients), triangle_areas.values.data(), simulation::view(state.positions), simulation::view(state.velocities), cache.triangle_conditions.values.data(), simulation::view(workspace.local_forces), workspace.local_symmetric_force_position_derivatives.values.data(), workspace.local_force_velocity_derivatives.values.data());
        kernels::assemble_hinges(model.stream, static_cast<std::uint32_t>(model.topology.hinges.size()), static_cast<std::uint32_t>(3uz * model.configuration.triangles.size()), static_cast<std::uint32_t>(9uz * model.configuration.triangles.size()), imperfection_stiffness, bending_damping, model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), hinge_rest_spans.values.data(), hinge_area_sums.values.data(), hinge_stiffnesses.values.data(), simulation::view(state.positions), simulation::view(state.velocities), cache.hinge_curvatures.values.data(), cache.hinge_curvature_first_derivatives.values.data(), cache.hinge_curvature_second_derivatives.values.data(), cache.hinge_responses.values.data(), cache.hinge_response_derivatives.values.data(), simulation::view(workspace.local_forces), workspace.local_symmetric_force_position_derivatives.values.data(), workspace.local_force_velocity_derivatives.values.data());
        kernels::gather(model.stream, static_cast<std::uint32_t>(model.particle_count), static_cast<std::uint32_t>(pattern.column_indices.size()), gravity, parameters.masses.values.data(), simulation::view(control.external_forces), force_contribution_offsets.values.data(), force_contribution_indices.values.data(), simulation::view(workspace.local_forces), block_contribution_offsets.values.data(), block_contribution_indices.values.data(), workspace.local_symmetric_force_position_derivatives.values.data(), workspace.local_force_velocity_derivatives.values.data(), simulation::view(cache.forces), cache.symmetric_force_position_derivative.block_values.values.data(), cache.force_velocity_derivative.block_values.values.data());
        kernels::build_system_and_right_hand_side(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, cache.system.row_offsets.values.data(), cache.system.column_indices.values.data(), parameters.masses.values.data(), cache.symmetric_force_position_derivative.block_values.values.data(), cache.force_velocity_derivative.block_values.values.data(), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(state.previous_positions), simulation::view(state.previous_velocities), simulation::view(cache.forces), cache.system.block_values.values.data(), simulation::view(cache.right_hand_side));
        kernels::build_prescribed_displacement(model.stream, static_cast<std::uint32_t>(model.particle_count), fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(cache.prescribed_displacement));
        block_solver.matvec(model, cache.system, cache.prescribed_displacement, workspace.system_times_prescribed_displacement);
        kernels::subtract(model.stream, static_cast<std::uint32_t>(model.particle_count), simulation::view(cache.right_hand_side), simulation::view(workspace.system_times_prescribed_displacement), simulation::view(workspace.reduced_right_hand_side));
        block_solver.solve(model, cache.system, workspace.reduced_right_hand_side, fixed_vertex_mask, cache.solution, workspace.pcg);
        kernels::finalize(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.previous_positions), simulation::view(state.velocities), simulation::view(cache.solution), simulation::view(cache.prescribed_displacement), simulation::view(cache.bdf2_displacement), simulation::view(next_state.positions), simulation::view(next_state.velocities), simulation::view(next_state.previous_positions), simulation::view(next_state.previous_velocities));
    }

    Solver::Pattern Solver::build_pattern(const Model<float>& model) {
        std::vector<std::set<std::uint32_t>> columns(model.particle_count);
        for (std::uint32_t particle = 0u; particle < model.particle_count; ++particle) columns[particle].insert(particle);
        for (const Triangle triangle : model.configuration.triangles) {
            const std::array vertices{triangle.first, triangle.second, triangle.third};
            for (const std::uint32_t row : vertices)
                for (const std::uint32_t column : vertices) columns[row].insert(column);
        }
        for (const Hinge hinge : model.topology.hinges) {
            columns[hinge.first_opposite].insert(hinge.second_opposite);
            columns[hinge.second_opposite].insert(hinge.first_opposite);
        }

        Pattern result{};
        result.row_offsets.resize(model.particle_count + 1uz);
        for (std::size_t row = 0uz; row < model.particle_count; ++row) {
            result.row_offsets[row + 1uz] = result.row_offsets[row] + static_cast<std::uint32_t>(columns[row].size());
            result.column_indices.insert(result.column_indices.end(), columns[row].begin(), columns[row].end());
        }

        std::vector<std::vector<std::uint32_t>> force_contributions(model.particle_count);
        for (std::uint32_t triangle = 0u; triangle < model.configuration.triangles.size(); ++triangle) {
            const Triangle vertices = model.configuration.triangles[triangle];
            force_contributions[vertices.first].push_back(3u * triangle);
            force_contributions[vertices.second].push_back(3u * triangle + 1u);
            force_contributions[vertices.third].push_back(3u * triangle + 2u);
        }
        const std::uint32_t hinge_force_offset = static_cast<std::uint32_t>(3uz * model.configuration.triangles.size());
        for (std::uint32_t hinge_index = 0u; hinge_index < model.topology.hinges.size(); ++hinge_index) {
            const Hinge hinge = model.topology.hinges[hinge_index];
            force_contributions[hinge.first_opposite].push_back(hinge_force_offset + 2u * hinge_index);
            force_contributions[hinge.second_opposite].push_back(hinge_force_offset + 2u * hinge_index + 1u);
        }
        result.force_contribution_offsets.resize(model.particle_count + 1uz);
        for (std::size_t particle = 0uz; particle < model.particle_count; ++particle) {
            result.force_contribution_offsets[particle + 1uz] = result.force_contribution_offsets[particle] + static_cast<std::uint32_t>(force_contributions[particle].size());
            result.force_contribution_indices.insert(result.force_contribution_indices.end(), force_contributions[particle].begin(), force_contributions[particle].end());
        }

        std::vector<std::vector<std::uint32_t>> block_contributions(result.column_indices.size());
        for (std::uint32_t triangle = 0u; triangle < model.configuration.triangles.size(); ++triangle) {
            const Triangle element = model.configuration.triangles[triangle];
            const std::array vertices{element.first, element.second, element.third};
            for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row)
                for (std::uint32_t local_column = 0u; local_column < 3u; ++local_column) block_contributions[find_block(result, vertices[local_row], vertices[local_column])].push_back(9u * triangle + 3u * local_row + local_column);
        }
        const std::uint32_t hinge_block_offset = static_cast<std::uint32_t>(9uz * model.configuration.triangles.size());
        for (std::uint32_t hinge_index = 0u; hinge_index < model.topology.hinges.size(); ++hinge_index) {
            const Hinge hinge = model.topology.hinges[hinge_index];
            const std::array vertices{hinge.first_opposite, hinge.second_opposite};
            for (std::uint32_t local_row = 0u; local_row < 2u; ++local_row)
                for (std::uint32_t local_column = 0u; local_column < 2u; ++local_column) block_contributions[find_block(result, vertices[local_row], vertices[local_column])].push_back(hinge_block_offset + 4u * hinge_index + 2u * local_row + local_column);
        }
        result.block_contribution_offsets.resize(result.column_indices.size() + 1uz);
        for (std::size_t block = 0uz; block < result.column_indices.size(); ++block) {
            result.block_contribution_offsets[block + 1uz] = result.block_contribution_offsets[block] + static_cast<std::uint32_t>(block_contributions[block].size());
            result.block_contribution_indices.insert(result.block_contribution_indices.end(), block_contributions[block].begin(), block_contributions[block].end());
        }
        return result;
    }

    std::uint32_t Solver::find_block(const Pattern& pattern, const std::uint32_t row, const std::uint32_t column) {
        const auto begin = pattern.column_indices.begin() + pattern.row_offsets[row];
        const auto end   = pattern.column_indices.begin() + pattern.row_offsets[row + 1u];
        return static_cast<std::uint32_t>(std::distance(pattern.column_indices.begin(), std::lower_bound(begin, end, column)));
    }

    MaterialCoordinate<float> Solver::material_coordinate(const Model<float>& model, const std::uint32_t triangle_index, const std::uint32_t particle) {
        const Triangle triangle                               = model.configuration.triangles[triangle_index];
        const TriangleMaterialCoordinates<float> coordinates = model.configuration.material_coordinates[triangle_index];
        if (particle == triangle.first) return coordinates.first;
        if (particle == triangle.second) return coordinates.second;
        return coordinates.third;
    }
} // namespace physica::deformables::cloth::solvers::choi_ko
