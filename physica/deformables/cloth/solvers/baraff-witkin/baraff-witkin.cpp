module;

#include "baraff-witkin-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.baraff_witkin;

import std;

namespace physica::deformables::cloth::solvers::baraff_witkin {
    Solver::Solver(const Model<float>& model, Configuration configuration)
        : time_step(configuration.time_step),
          gravity(configuration.gravity),
          stretch_u_target(configuration.stretch_u_target),
          stretch_v_target(configuration.stretch_v_target),
          stretch_u_stiffness(configuration.stretch_u_stiffness),
          stretch_v_stiffness(configuration.stretch_v_stiffness),
          shear_stiffness(configuration.shear_stiffness),
          stretch_u_damping(configuration.stretch_u_damping),
          stretch_v_damping(configuration.stretch_v_damping),
          shear_damping(configuration.shear_damping),
          pattern(build_pattern(model)),
          triangle_coloring(build_triangle_coloring(model.configuration.triangles, model.particle_count)),
          hinge_coloring(build_hinge_coloring(model)),
          block_solver(model, {.iteration_count = configuration.pcg_iteration_count}),
          triangle_u_coefficients(model.stream, model.configuration.triangles.size()),
          triangle_v_coefficients(model.stream, model.configuration.triangles.size()),
          triangle_areas(model.stream, model.configuration.triangles.size()),
          hinge_rest_angles(model.stream, model.topology.hinges.size()),
          hinge_stiffnesses(model.stream, model.topology.hinges.size()),
          hinge_dampings(model.stream, model.topology.hinges.size()),
          colored_triangles(model.stream, model.configuration.triangles.size()),
          colored_hinges(model.stream, model.topology.hinges.size()),
          triangle_block_indices(model.stream, 9uz * model.configuration.triangles.size()),
          hinge_block_indices(model.stream, 16uz * model.topology.hinges.size()),
          fixed_vertex_mask(model.stream, model.particle_count),
          fixed_positions(model.stream, model.particle_count) {
        std::vector<Vector3<float>> host_triangle_u_coefficients(model.configuration.triangles.size());
        std::vector<Vector3<float>> host_triangle_v_coefficients(model.configuration.triangles.size());
        std::vector<float> host_triangle_areas(model.configuration.triangles.size());
        std::vector<std::uint32_t> host_triangle_block_indices(9uz * model.configuration.triangles.size());
        for (std::size_t triangle_index = 0uz; triangle_index < model.configuration.triangles.size(); ++triangle_index) {
            const Triangle triangle                                = model.configuration.triangles[triangle_index];
            const TriangleMaterialCoordinates<float> coordinates  = model.configuration.material_coordinates[triangle_index];
            const float delta_u_first                              = coordinates.second.u - coordinates.first.u;
            const float delta_v_first                              = coordinates.second.v - coordinates.first.v;
            const float delta_u_second                             = coordinates.third.u - coordinates.first.u;
            const float delta_v_second                             = coordinates.third.v - coordinates.first.v;
            const float determinant                                = delta_u_first * delta_v_second - delta_u_second * delta_v_first;
            const float inverse_00                                 = delta_v_second / determinant;
            const float inverse_01                                 = -delta_u_second / determinant;
            const float inverse_10                                 = -delta_v_first / determinant;
            const float inverse_11                                 = delta_u_first / determinant;
            host_triangle_u_coefficients[triangle_index]           = {.x = -inverse_00 - inverse_10, .y = inverse_00, .z = inverse_10};
            host_triangle_v_coefficients[triangle_index]           = {.x = -inverse_01 - inverse_11, .y = inverse_01, .z = inverse_11};
            host_triangle_areas[triangle_index]                    = 0.5F * std::abs(determinant);
            const std::array vertices{triangle.first, triangle.second, triangle.third};
            for (std::size_t local_row = 0uz; local_row < 3uz; ++local_row)
                for (std::size_t local_column = 0uz; local_column < 3uz; ++local_column) host_triangle_block_indices[9uz * triangle_index + 3uz * local_row + local_column] = find_block(pattern, vertices[local_row], vertices[local_column]);
        }

        std::vector<float> host_hinge_rest_angles(model.topology.hinges.size());
        std::vector<float> host_hinge_stiffnesses(model.topology.hinges.size());
        std::vector<float> host_hinge_dampings(model.topology.hinges.size());
        std::vector<std::uint32_t> host_hinge_block_indices(16uz * model.topology.hinges.size());
        for (std::size_t hinge_index = 0uz; hinge_index < model.topology.hinges.size(); ++hinge_index) {
            const Hinge hinge = model.topology.hinges[hinge_index];
            host_hinge_rest_angles[hinge_index] = signed_dihedral(model.configuration.rest_positions, hinge);
            const MaterialCoordinate<float> edge_first_coordinate  = material_coordinate(model, hinge.first_triangle, hinge.edge_first);
            const MaterialCoordinate<float> edge_second_coordinate = material_coordinate(model, hinge.first_triangle, hinge.edge_second);
            const float delta_u                                    = edge_first_coordinate.u - edge_second_coordinate.u;
            const float delta_v                                    = edge_first_coordinate.v - edge_second_coordinate.v;
            const float inverse_squared_length                     = 1.0F / (delta_u * delta_u + delta_v * delta_v);
            host_hinge_stiffnesses[hinge_index]                    = inverse_squared_length * (configuration.bend_u_stiffness * delta_u * delta_u + configuration.bend_v_stiffness * delta_v * delta_v);
            host_hinge_dampings[hinge_index]                       = inverse_squared_length * (configuration.bend_u_damping * delta_u * delta_u + configuration.bend_v_damping * delta_v * delta_v);
            const std::array vertices{hinge.edge_first, hinge.edge_second, hinge.first_opposite, hinge.second_opposite};
            for (std::size_t local_row = 0uz; local_row < 4uz; ++local_row)
                for (std::size_t local_column = 0uz; local_column < 4uz; ++local_column) host_hinge_block_indices[16uz * hinge_index + 4uz * local_row + local_column] = find_block(pattern, vertices[local_row], vertices[local_column]);
        }

        std::vector<std::uint32_t> host_fixed_vertex_mask(model.particle_count);
        std::vector<Vector3<float>> host_fixed_positions = model.configuration.rest_positions;
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            host_fixed_vertex_mask[fixed_vertex.particle] = 1u;
            host_fixed_positions[fixed_vertex.particle]    = fixed_vertex.position;
        }

        simulation::upload(model.stream, host_triangle_u_coefficients, triangle_u_coefficients);
        simulation::upload(model.stream, host_triangle_v_coefficients, triangle_v_coefficients);
        ::cuda::copy_bytes(model.stream, host_triangle_areas, triangle_areas.values);
        ::cuda::copy_bytes(model.stream, host_hinge_rest_angles, hinge_rest_angles.values);
        ::cuda::copy_bytes(model.stream, host_hinge_stiffnesses, hinge_stiffnesses.values);
        ::cuda::copy_bytes(model.stream, host_hinge_dampings, hinge_dampings.values);
        ::cuda::copy_bytes(model.stream, triangle_coloring.triangles, colored_triangles.values);
        ::cuda::copy_bytes(model.stream, hinge_coloring.hinges, colored_hinges.values);
        ::cuda::copy_bytes(model.stream, host_triangle_block_indices, triangle_block_indices.values);
        ::cuda::copy_bytes(model.stream, host_hinge_block_indices, hinge_block_indices.values);
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
        return {
            .triangle_conditions        = simulation::VectorField<float>(model.stream, model.configuration.triangles.size()),
            .bending_angles             = simulation::ScalarField<float>(model.stream, model.topology.hinges.size()),
            .forces                     = simulation::VectorField<float>(model.stream, model.particle_count),
            .force_position_derivative  = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .force_velocity_derivative  = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .system                     = block_pcg::BlockCsrMatrix(model, pattern.row_offsets, pattern.column_indices),
            .right_hand_side            = simulation::VectorField<float>(model.stream, model.particle_count),
            .constraint_velocity_change = simulation::VectorField<float>(model.stream, model.particle_count),
            .velocity_increment         = simulation::VectorField<float>(model.stream, model.particle_count),
        };
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>& model) const {
        block_pcg::BlockCsrMatrix matrix(model, pattern.row_offsets, pattern.column_indices);
        return {
            .matrix_times_constraint_velocity_change = simulation::VectorField<float>(model.stream, model.particle_count),
            .reduced_right_hand_side                  = simulation::VectorField<float>(model.stream, model.particle_count),
            .free_velocity_change                     = simulation::VectorField<float>(model.stream, model.particle_count),
            .pcg                                      = block_solver.allocate_workspace(model, matrix),
        };
    }

    void Solver::forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters& parameters, State<float>& next_state, StepCache& cache, Workspace& workspace) const {
        block_solver.clear(model, cache.force_position_derivative);
        block_solver.clear(model, cache.force_velocity_derivative);
        kernels::initialize_forces(model.stream, static_cast<std::uint32_t>(model.particle_count), gravity, parameters.masses.values.data(), simulation::view(control.external_forces), simulation::view(cache.forces));
        for (std::size_t color = 0uz; color + 1uz < triangle_coloring.offsets.size(); ++color) kernels::assemble_triangles(model.stream, triangle_coloring.offsets[color + 1uz] - triangle_coloring.offsets[color], colored_triangles.values.data() + triangle_coloring.offsets[color], stretch_u_target, stretch_v_target, stretch_u_stiffness, stretch_v_stiffness, shear_stiffness, stretch_u_damping, stretch_v_damping, shear_damping, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), simulation::view(triangle_u_coefficients), simulation::view(triangle_v_coefficients), triangle_areas.values.data(), triangle_block_indices.values.data(), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.triangle_conditions), simulation::view(cache.forces), cache.force_position_derivative.block_values.values.data(), cache.force_velocity_derivative.block_values.values.data());
        for (std::size_t color = 0uz; color + 1uz < hinge_coloring.offsets.size(); ++color) kernels::assemble_hinges(model.stream, hinge_coloring.offsets[color + 1uz] - hinge_coloring.offsets[color], colored_hinges.values.data() + hinge_coloring.offsets[color], model.topology.device.hinges.edge_first.values.data(), model.topology.device.hinges.edge_second.values.data(), model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), hinge_rest_angles.values.data(), hinge_stiffnesses.values.data(), hinge_dampings.values.data(), hinge_block_indices.values.data(), simulation::view(state.positions), simulation::view(state.velocities), cache.bending_angles.values.data(), simulation::view(cache.forces), cache.force_position_derivative.block_values.values.data(), cache.force_velocity_derivative.block_values.values.data());
        kernels::build_system(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, cache.system.row_offsets.values.data(), cache.system.column_indices.values.data(), parameters.masses.values.data(), cache.force_position_derivative.block_values.values.data(), cache.force_velocity_derivative.block_values.values.data(), simulation::view(state.velocities), simulation::view(cache.forces), cache.system.block_values.values.data(), simulation::view(cache.right_hand_side));
        kernels::build_constraint_velocity_change(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.constraint_velocity_change));
        block_solver.matvec(model, cache.system, cache.constraint_velocity_change, workspace.matrix_times_constraint_velocity_change);
        kernels::subtract(model.stream, static_cast<std::uint32_t>(model.particle_count), simulation::view(cache.right_hand_side), simulation::view(workspace.matrix_times_constraint_velocity_change), simulation::view(workspace.reduced_right_hand_side));
        block_solver.solve(model, cache.system, workspace.reduced_right_hand_side, fixed_vertex_mask, workspace.free_velocity_change, workspace.pcg);
        kernels::finalize(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(state.positions), simulation::view(state.velocities), simulation::view(workspace.free_velocity_change), simulation::view(cache.constraint_velocity_change), simulation::view(cache.velocity_increment), simulation::view(next_state.positions), simulation::view(next_state.velocities));
    }

    Solver::Pattern Solver::build_pattern(const Model<float>& model) {
        std::vector<std::set<std::uint32_t>> rows(model.particle_count);
        for (std::uint32_t particle = 0u; particle < model.particle_count; ++particle) rows[particle].insert(particle);
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

        Pattern result{.row_offsets = std::vector<std::uint32_t>(model.particle_count + 1uz), .column_indices = {}};
        for (std::size_t row = 0uz; row < rows.size(); ++row) {
            result.row_offsets[row + 1uz] = result.row_offsets[row] + static_cast<std::uint32_t>(rows[row].size());
            result.column_indices.insert(result.column_indices.end(), rows[row].begin(), rows[row].end());
        }
        return result;
    }

    Solver::HingeColoring Solver::build_hinge_coloring(const Model<float>& model) {
        HingeColoring result{.offsets = {0u}, .hinges = {}};
        result.hinges.reserve(model.topology.hinges.size());
        std::vector<std::uint32_t> remaining_hinges(model.topology.hinges.size());
        for (std::size_t hinge = 0uz; hinge < remaining_hinges.size(); ++hinge) remaining_hinges[hinge] = static_cast<std::uint32_t>(hinge);
        while (!remaining_hinges.empty()) {
            std::vector<std::uint8_t> occupied_vertices(model.particle_count);
            std::vector<std::uint32_t> next_remaining_hinges;
            next_remaining_hinges.reserve(remaining_hinges.size());
            for (const std::uint32_t hinge_index : remaining_hinges) {
                const Hinge hinge = model.topology.hinges[hinge_index];
                const std::array vertices{hinge.edge_first, hinge.edge_second, hinge.first_opposite, hinge.second_opposite};
                if (occupied_vertices[vertices[0]] != 0u || occupied_vertices[vertices[1]] != 0u || occupied_vertices[vertices[2]] != 0u || occupied_vertices[vertices[3]] != 0u) {
                    next_remaining_hinges.push_back(hinge_index);
                    continue;
                }
                result.hinges.push_back(hinge_index);
                for (const std::uint32_t vertex : vertices) occupied_vertices[vertex] = 1u;
            }
            result.offsets.push_back(static_cast<std::uint32_t>(result.hinges.size()));
            remaining_hinges = std::move(next_remaining_hinges);
        }
        return result;
    }

    std::uint32_t Solver::find_block(const Pattern& pattern, const std::uint32_t row, const std::uint32_t column) {
        return static_cast<std::uint32_t>(std::lower_bound(pattern.column_indices.begin() + pattern.row_offsets[row], pattern.column_indices.begin() + pattern.row_offsets[row + 1u], column) - pattern.column_indices.begin());
    }

    MaterialCoordinate<float> Solver::material_coordinate(const Model<float>& model, const std::uint32_t triangle_index, const std::uint32_t particle) {
        const Triangle triangle                               = model.configuration.triangles[triangle_index];
        const TriangleMaterialCoordinates<float> coordinates = model.configuration.material_coordinates[triangle_index];
        if (particle == triangle.first) return coordinates.first;
        if (particle == triangle.second) return coordinates.second;
        return coordinates.third;
    }

    float Solver::signed_dihedral(const std::vector<Vector3<float>>& positions, const Hinge& hinge) {
        const Vector3<float> edge_first      = positions[hinge.edge_first];
        const Vector3<float> edge_second     = positions[hinge.edge_second];
        const Vector3<float> first_opposite  = positions[hinge.first_opposite];
        const Vector3<float> second_opposite = positions[hinge.second_opposite];
        const Vector3<float> edge            = normalized(edge_second - edge_first);
        const Vector3<float> first_normal    = normalized(cross(edge_second - edge_first, first_opposite - edge_first));
        const Vector3<float> second_normal   = normalized(cross(edge_first - edge_second, second_opposite - edge_second));
        return std::atan2(dot(cross(first_normal, second_normal), edge), dot(first_normal, second_normal));
    }
} // namespace physica::deformables::cloth::solvers::baraff_witkin
