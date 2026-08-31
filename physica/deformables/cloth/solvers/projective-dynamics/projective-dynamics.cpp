module;

#include "projective-dynamics-kernels.h"
#include "../position-dynamics-kernels.h"
#include <cudss.h>
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.projective_dynamics;

import std;

namespace physica::deformables::cloth::solvers::projective_dynamics {
    namespace {
        void require_cudss(const cudssStatus_t status, const std::string_view operation) {
            if (status != CUDSS_STATUS_SUCCESS) throw std::runtime_error(std::format("{} failed with cuDSS status {}", operation, static_cast<int>(status)));
        }
    } // namespace

    Solver::Solver(const Model<float>& model, Configuration configuration) : Solver(model, configuration, build_system(model, configuration)) {}

    Solver::~Solver() {
        destroy_cudss();
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

    Solver::Parameters Solver::allocate_parameters(const Model<float>&) const {
        return {};
    }

    Solver::StepCache Solver::allocate_step_cache(const Model<float>& model) const {
        return {
            .predicted_positions              = simulation::VectorField<float>(model.stream, model.particle_count),
            .projected_frame_first_columns    = simulation::VectorField<float>(model.stream, model.configuration.triangles.size()),
            .projected_frame_second_columns   = simulation::VectorField<float>(model.stream, model.configuration.triangles.size()),
            .bending_directions               = simulation::VectorField<float>(model.stream, model.topology.hinges.size()),
        };
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>&) const {
        return {};
    }

    void Solver::forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters&, State<float>& next_state, StepCache& cache, Workspace&) const {
        position_dynamics::kernels::predict(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, gravity, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_forces), masses.values.data(), simulation::view(cache.predicted_positions));
        simulation::copy(model.stream, cache.predicted_positions, next_state.positions);
        const std::uint32_t triangle_count = static_cast<std::uint32_t>(model.configuration.triangles.size());
        const std::uint32_t hinge_count    = static_cast<std::uint32_t>(model.topology.hinges.size());
        const float inverse_time_step_squared = 1.0F / (time_step * time_step);
        for (std::uint32_t iteration = 0u; iteration < global_iteration_count; ++iteration) {
            kernels::project_membranes(model.stream, triangle_count, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), material_inverse_00.values.data(), material_inverse_01.values.data(), material_inverse_10.values.data(), material_inverse_11.values.data(), simulation::view(next_state.positions), simulation::view(cache.projected_frame_first_columns), simulation::view(cache.projected_frame_second_columns));
            kernels::project_bending(model.stream, hinge_count, model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), simulation::view(next_state.positions), simulation::view(cache.bending_directions));
            kernels::assemble_right_hand_sides(model.stream, free_particle_count, inverse_time_step_squared, bending_stiffness, free_particles.values.data(), model.topology.device.vertex_triangles.offsets.values.data(), model.topology.device.vertex_triangles.indices.values.data(), model.topology.device.vertex_hinges.offsets.values.data(), model.topology.device.vertex_hinges.indices.values.data(), model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), model.topology.device.hinges.first_opposite.values.data(), model.topology.device.hinges.second_opposite.values.data(), material_inverse_00.values.data(), material_inverse_01.values.data(), material_inverse_10.values.data(), material_inverse_11.values.data(), membrane_weights.values.data(), bending_rest_lengths.values.data(), masses.values.data(), simulation::view(cache.predicted_positions), simulation::view(fixed_right_hand_sides), simulation::view(cache.projected_frame_first_columns), simulation::view(cache.projected_frame_second_columns), simulation::view(cache.bending_directions), right_hand_sides.values.data());
            if (free_particle_count != 0u) for (std::size_t component = 0uz; component < right_hand_side_matrices.size(); ++component) require_cudss(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, cudss_configuration, cudss_data, system_matrix, solution_matrices[component], right_hand_side_matrices[component]), "cudssExecute solve");
            kernels::scatter_solution(model.stream, free_particle_count, free_particles.values.data(), solutions.values.data(), simulation::view(next_state.positions));
        }
        position_dynamics::kernels::reconstruct_velocities(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(state.positions), simulation::view(next_state.positions), simulation::view(next_state.velocities));
    }

    Solver::HostSystem Solver::build_system(const Model<float>& model, const Configuration& configuration) {
        const std::size_t triangle_count = model.configuration.triangles.size();
        const std::size_t hinge_count    = model.topology.hinges.size();
        HostSystem result{
            .material_inverse_00      = std::vector<float>(triangle_count),
            .material_inverse_01      = std::vector<float>(triangle_count),
            .material_inverse_10      = std::vector<float>(triangle_count),
            .material_inverse_11      = std::vector<float>(triangle_count),
            .membrane_weights         = std::vector<float>(triangle_count),
            .bending_rest_lengths     = std::vector<float>(hinge_count),
            .fixed_vertex_mask        = std::vector<std::uint32_t>(model.particle_count),
            .fixed_positions          = model.configuration.rest_positions,
            .free_particles           = {},
            .fixed_right_hand_sides   = {},
            .matrix_row_offsets       = {},
            .matrix_column_indices    = {},
            .matrix_values            = {},
        };
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            result.fixed_vertex_mask[fixed_vertex.particle] = 1u;
            result.fixed_positions[fixed_vertex.particle]   = fixed_vertex.position;
        }

        std::vector<std::int32_t> particle_to_free(model.particle_count, -1);
        result.free_particles.reserve(model.particle_count - configuration.fixed_vertices.size());
        for (std::uint32_t particle = 0u; particle < model.particle_count; ++particle) {
            if (result.fixed_vertex_mask[particle] != 0u) continue;
            particle_to_free[particle] = static_cast<std::int32_t>(result.free_particles.size());
            result.free_particles.push_back(particle);
        }
        result.fixed_right_hand_sides.resize(result.free_particles.size());
        std::vector<std::map<std::uint32_t, float>> upper_matrix(result.free_particles.size());
        const float inverse_time_step_squared = 1.0F / (configuration.time_step * configuration.time_step);
        for (std::uint32_t free_particle = 0u; free_particle < result.free_particles.size(); ++free_particle) {
            const std::uint32_t particle = result.free_particles[free_particle];
            upper_matrix[free_particle][free_particle] = configuration.masses[particle] * inverse_time_step_squared;
        }

        for (std::uint32_t triangle = 0u; triangle < triangle_count; ++triangle) {
            const TriangleMaterialCoordinates<float> material = model.configuration.material_coordinates[triangle];
            const float material_10_u = material.second.u - material.first.u;
            const float material_10_v = material.second.v - material.first.v;
            const float material_20_u = material.third.u - material.first.u;
            const float material_20_v = material.third.v - material.first.v;
            const float determinant   = material_10_u * material_20_v - material_20_u * material_10_v;
            result.material_inverse_00[triangle] = material_20_v / determinant;
            result.material_inverse_01[triangle] = -material_20_u / determinant;
            result.material_inverse_10[triangle] = -material_10_v / determinant;
            result.material_inverse_11[triangle] = material_10_u / determinant;
            result.membrane_weights[triangle]    = 0.5F * configuration.membrane_stiffness * std::abs(determinant);

            const Triangle topology = model.configuration.triangles[triangle];
            const std::array<std::uint32_t, 3uz> particles{topology.first, topology.second, topology.third};
            const std::array<MaterialCoordinate<float>, 3uz> gradients{
                MaterialCoordinate<float>{.u = -result.material_inverse_00[triangle] - result.material_inverse_10[triangle], .v = -result.material_inverse_01[triangle] - result.material_inverse_11[triangle]},
                MaterialCoordinate<float>{.u = result.material_inverse_00[triangle], .v = result.material_inverse_01[triangle]},
                MaterialCoordinate<float>{.u = result.material_inverse_10[triangle], .v = result.material_inverse_11[triangle]},
            };
            for (std::size_t local_row = 0uz; local_row < particles.size(); ++local_row) {
                const std::int32_t free_row = particle_to_free[particles[local_row]];
                if (free_row < 0) continue;
                for (std::size_t local_column = 0uz; local_column < particles.size(); ++local_column) {
                    const float coefficient = result.membrane_weights[triangle] * (gradients[local_row].u * gradients[local_column].u + gradients[local_row].v * gradients[local_column].v);
                    const std::int32_t free_column = particle_to_free[particles[local_column]];
                    if (free_column < 0) {
                        result.fixed_right_hand_sides[free_row] = result.fixed_right_hand_sides[free_row] - coefficient * result.fixed_positions[particles[local_column]];
                        continue;
                    }
                    if (free_row <= free_column) upper_matrix[free_row][free_column] += coefficient;
                }
            }
        }

        for (std::uint32_t hinge = 0u; hinge < hinge_count; ++hinge) {
            const Hinge topology = model.topology.hinges[hinge];
            result.bending_rest_lengths[hinge] = length(model.configuration.rest_positions[topology.second_opposite] - model.configuration.rest_positions[topology.first_opposite]);
            const std::array<std::uint32_t, 2uz> particles{topology.first_opposite, topology.second_opposite};
            constexpr std::array<float, 2uz> gradients{-1.0F, 1.0F};
            for (std::size_t local_row = 0uz; local_row < particles.size(); ++local_row) {
                const std::int32_t free_row = particle_to_free[particles[local_row]];
                if (free_row < 0) continue;
                for (std::size_t local_column = 0uz; local_column < particles.size(); ++local_column) {
                    const float coefficient = configuration.bending_stiffness * gradients[local_row] * gradients[local_column];
                    const std::int32_t free_column = particle_to_free[particles[local_column]];
                    if (free_column < 0) {
                        result.fixed_right_hand_sides[free_row] = result.fixed_right_hand_sides[free_row] - coefficient * result.fixed_positions[particles[local_column]];
                        continue;
                    }
                    if (free_row <= free_column) upper_matrix[free_row][free_column] += coefficient;
                }
            }
        }

        result.matrix_row_offsets.resize(result.free_particles.size() + 1uz);
        for (std::uint32_t row = 0u; row < result.free_particles.size(); ++row) {
            result.matrix_row_offsets[row] = static_cast<std::int32_t>(result.matrix_column_indices.size());
            for (const auto [column, value] : upper_matrix[row]) {
                result.matrix_column_indices.push_back(static_cast<std::int32_t>(column));
                result.matrix_values.push_back(value);
            }
        }
        result.matrix_row_offsets[result.free_particles.size()] = static_cast<std::int32_t>(result.matrix_column_indices.size());
        return result;
    }

    Solver::Solver(const Model<float>& model, const Configuration& configuration, HostSystem system)
        : time_step(configuration.time_step), global_iteration_count(configuration.global_iteration_count), gravity(configuration.gravity), bending_stiffness(configuration.bending_stiffness), free_particle_count(static_cast<std::uint32_t>(system.free_particles.size())), masses(model.stream, model.particle_count), material_inverse_00(model.stream, system.material_inverse_00.size()), material_inverse_01(model.stream, system.material_inverse_01.size()), material_inverse_10(model.stream, system.material_inverse_10.size()), material_inverse_11(model.stream, system.material_inverse_11.size()), membrane_weights(model.stream, system.membrane_weights.size()), bending_rest_lengths(model.stream, system.bending_rest_lengths.size()), fixed_vertex_mask(model.stream, system.fixed_vertex_mask.size()), fixed_positions(model.stream, system.fixed_positions.size()), free_particles(model.stream, system.free_particles.size()), fixed_right_hand_sides(model.stream, system.fixed_right_hand_sides.size()), matrix_row_offsets(model.stream, system.matrix_row_offsets.size()), matrix_column_indices(model.stream, system.matrix_column_indices.size()), matrix_values(model.stream, system.matrix_values.size()), right_hand_sides(model.stream, 3uz * free_particle_count), solutions(model.stream, 3uz * free_particle_count), cudss_handle(nullptr), cudss_configuration(nullptr), cudss_data(nullptr), system_matrix(nullptr), right_hand_side_matrices{nullptr, nullptr, nullptr}, solution_matrices{nullptr, nullptr, nullptr} {
        ::cuda::copy_bytes(model.stream, configuration.masses, masses.values);
        ::cuda::copy_bytes(model.stream, system.material_inverse_00, material_inverse_00.values);
        ::cuda::copy_bytes(model.stream, system.material_inverse_01, material_inverse_01.values);
        ::cuda::copy_bytes(model.stream, system.material_inverse_10, material_inverse_10.values);
        ::cuda::copy_bytes(model.stream, system.material_inverse_11, material_inverse_11.values);
        ::cuda::copy_bytes(model.stream, system.membrane_weights, membrane_weights.values);
        ::cuda::copy_bytes(model.stream, system.bending_rest_lengths, bending_rest_lengths.values);
        ::cuda::copy_bytes(model.stream, system.fixed_vertex_mask, fixed_vertex_mask.values);
        simulation::upload(model.stream, system.fixed_positions, fixed_positions);
        ::cuda::copy_bytes(model.stream, system.free_particles, free_particles.values);
        simulation::upload(model.stream, system.fixed_right_hand_sides, fixed_right_hand_sides);
        ::cuda::copy_bytes(model.stream, system.matrix_row_offsets, matrix_row_offsets.values);
        ::cuda::copy_bytes(model.stream, system.matrix_column_indices, matrix_column_indices.values);
        ::cuda::copy_bytes(model.stream, system.matrix_values, matrix_values.values);
        simulation::clear(model.stream, right_hand_sides);
        simulation::clear(model.stream, solutions);

        try {
            if (free_particle_count != 0u) {
                require_cudss(cudssCreate(&cudss_handle), "cudssCreate");
                require_cudss(cudssSetStream(cudss_handle, model.stream.get()), "cudssSetStream");
                require_cudss(cudssConfigCreate(&cudss_configuration), "cudssConfigCreate");
                require_cudss(cudssDataCreate(cudss_handle, &cudss_data), "cudssDataCreate");
                constexpr std::int32_t deterministic_mode = 1;
                require_cudss(cudssConfigSet(cudss_configuration, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic_mode, sizeof(deterministic_mode)), "cudssConfigSet deterministic mode");
                require_cudss(cudssMatrixCreateCsr(&system_matrix, free_particle_count, free_particle_count, static_cast<std::int64_t>(system.matrix_values.size()), matrix_row_offsets.values.data(), nullptr, matrix_column_indices.values.data(), matrix_values.values.data(), CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_32F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO), "cudssMatrixCreateCsr");
                for (std::size_t component = 0uz; component < right_hand_side_matrices.size(); ++component) {
                    require_cudss(cudssMatrixCreateDn(&right_hand_side_matrices[component], free_particle_count, 1, free_particle_count, right_hand_sides.values.data() + component * free_particle_count, CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn right-hand side");
                    require_cudss(cudssMatrixCreateDn(&solution_matrices[component], free_particle_count, 1, free_particle_count, solutions.values.data() + component * free_particle_count, CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn solution");
                }
                require_cudss(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, cudss_configuration, cudss_data, system_matrix, solution_matrices[0], right_hand_side_matrices[0]), "cudssExecute analysis");
                require_cudss(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, cudss_configuration, cudss_data, system_matrix, solution_matrices[0], right_hand_side_matrices[0]), "cudssExecute factorization");
            }
            model.stream.sync();
        } catch (...) {
            destroy_cudss();
            throw;
        }
    }

    void Solver::destroy_cudss() noexcept {
        for (const cudssMatrix_t solution_matrix : solution_matrices) if (solution_matrix != nullptr) cudssMatrixDestroy(solution_matrix);
        for (const cudssMatrix_t right_hand_side_matrix : right_hand_side_matrices) if (right_hand_side_matrix != nullptr) cudssMatrixDestroy(right_hand_side_matrix);
        if (system_matrix != nullptr) cudssMatrixDestroy(system_matrix);
        if (cudss_data != nullptr && cudss_handle != nullptr) cudssDataDestroy(cudss_handle, cudss_data);
        if (cudss_configuration != nullptr) cudssConfigDestroy(cudss_configuration);
        if (cudss_handle != nullptr) cudssDestroy(cudss_handle);
    }
} // namespace physica::deformables::cloth::solvers::projective_dynamics
