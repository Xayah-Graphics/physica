module;

#include "fast-mass-spring-kernels.h"
#include "../position-dynamics-kernels.h"
#include <cudss.h>
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.fast_mass_spring;

import std;

namespace physica::deformables::cloth::solvers::fast_mass_spring {
    namespace {
        void require_cudss(const cudssStatus_t status, const std::string_view operation) {
            if (status != CUDSS_STATUS_SUCCESS) throw std::runtime_error(std::format("{} failed with cuDSS status {}", operation, static_cast<int>(status)));
        }

        std::size_t count_free_free_edges(const Model<float>& model, const std::span<const Solver::FixedVertex> fixed_vertices) {
            std::vector<std::uint32_t> fixed_vertex_mask(model.particle_count);
            for (const Solver::FixedVertex fixed_vertex : fixed_vertices) fixed_vertex_mask[fixed_vertex.particle] = 1u;
            std::size_t result = 0uz;
            for (const Edge edge : model.topology.edges)
                if (fixed_vertex_mask[edge.first] == 0u && fixed_vertex_mask[edge.second] == 0u) ++result;
            return result;
        }
    } // namespace

    Solver::Solver(const Model<float>& model, Configuration configuration)
        : time_step(configuration.time_step), global_iteration_count(configuration.global_iteration_count), gravity(configuration.gravity), spring_stiffness(configuration.spring_stiffness), free_particle_count(static_cast<std::uint32_t>(model.particle_count - configuration.fixed_vertices.size())), masses(model.stream, model.particle_count), rest_lengths(model.stream, model.topology.edges.size()), fixed_vertex_mask(model.stream, model.particle_count), fixed_positions(model.stream, model.particle_count), free_particles(model.stream, free_particle_count), matrix_row_offsets(model.stream, static_cast<std::size_t>(free_particle_count) + 1uz), matrix_column_indices(model.stream, static_cast<std::size_t>(free_particle_count) + count_free_free_edges(model, configuration.fixed_vertices)), matrix_values(model.stream, matrix_column_indices.values.size()), right_hand_sides(model.stream, 3uz * free_particle_count), solutions(model.stream, 3uz * free_particle_count), cudss_handle(nullptr), cudss_configuration(nullptr), cudss_data(nullptr), system_matrix(nullptr), right_hand_side_matrices{}, solution_matrices{} {
        std::vector<float> host_rest_lengths(model.topology.edges.size());
        for (std::size_t edge_index = 0uz; edge_index < model.topology.edges.size(); ++edge_index) {
            const Edge edge               = model.topology.edges[edge_index];
            host_rest_lengths[edge_index] = length(model.configuration.rest_positions[edge.first] - model.configuration.rest_positions[edge.second]);
        }

        std::vector<std::uint32_t> host_fixed_vertex_mask(model.particle_count);
        std::vector<Vector3<float>> host_fixed_positions = model.configuration.rest_positions;
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            host_fixed_vertex_mask[fixed_vertex.particle] = 1u;
            host_fixed_positions[fixed_vertex.particle]    = fixed_vertex.position;
        }

        std::vector<std::uint32_t> host_particle_to_free(model.particle_count);
        std::vector<std::uint32_t> host_free_particles(free_particle_count);
        std::uint32_t free_particle = 0u;
        for (std::uint32_t particle = 0u; particle < model.particle_count; ++particle) {
            if (host_fixed_vertex_mask[particle] != 0u) continue;
            host_particle_to_free[particle] = free_particle;
            host_free_particles[free_particle++] = particle;
        }

        std::vector<std::uint32_t> degrees(model.particle_count);
        std::vector<std::vector<std::int32_t>> upper_neighbors(free_particle_count);
        for (const Edge edge : model.topology.edges) {
            ++degrees[edge.first];
            ++degrees[edge.second];
            if (host_fixed_vertex_mask[edge.first] != 0u || host_fixed_vertex_mask[edge.second] != 0u) continue;
            const std::uint32_t first_free  = host_particle_to_free[edge.first];
            const std::uint32_t second_free = host_particle_to_free[edge.second];
            const std::uint32_t row         = std::min(first_free, second_free);
            const std::uint32_t column      = std::max(first_free, second_free);
            upper_neighbors[row].push_back(static_cast<std::int32_t>(column));
        }
        for (std::vector<std::int32_t>& neighbors : upper_neighbors) std::ranges::sort(neighbors);

        const float inverse_time_step_squared = 1.0F / (time_step * time_step);
        std::vector<std::int32_t> host_matrix_row_offsets(static_cast<std::size_t>(free_particle_count) + 1uz);
        std::vector<std::int32_t> host_matrix_column_indices;
        std::vector<float> host_matrix_values;
        host_matrix_column_indices.reserve(matrix_column_indices.values.size());
        host_matrix_values.reserve(matrix_values.values.size());
        for (std::uint32_t row = 0u; row < free_particle_count; ++row) {
            const std::uint32_t particle   = host_free_particles[row];
            host_matrix_row_offsets[row]   = static_cast<std::int32_t>(host_matrix_column_indices.size());
            host_matrix_column_indices.push_back(static_cast<std::int32_t>(row));
            host_matrix_values.push_back(configuration.masses[particle] * inverse_time_step_squared + spring_stiffness * static_cast<float>(degrees[particle]));
            for (const std::int32_t column : upper_neighbors[row]) {
                host_matrix_column_indices.push_back(column);
                host_matrix_values.push_back(-spring_stiffness);
            }
        }
        host_matrix_row_offsets[free_particle_count] = static_cast<std::int32_t>(host_matrix_column_indices.size());

        ::cuda::copy_bytes(model.stream, configuration.masses, masses.values);
        ::cuda::copy_bytes(model.stream, host_rest_lengths, rest_lengths.values);
        ::cuda::copy_bytes(model.stream, host_fixed_vertex_mask, fixed_vertex_mask.values);
        simulation::upload(model.stream, host_fixed_positions, fixed_positions);
        ::cuda::copy_bytes(model.stream, host_free_particles, free_particles.values);
        ::cuda::copy_bytes(model.stream, host_matrix_row_offsets, matrix_row_offsets.values);
        ::cuda::copy_bytes(model.stream, host_matrix_column_indices, matrix_column_indices.values);
        ::cuda::copy_bytes(model.stream, host_matrix_values, matrix_values.values);
        simulation::clear(model.stream, right_hand_sides);
        simulation::clear(model.stream, solutions);

        try {
            require_cudss(cudssCreate(&cudss_handle), "cudssCreate");
            require_cudss(cudssSetStream(cudss_handle, model.stream.get()), "cudssSetStream");
            require_cudss(cudssConfigCreate(&cudss_configuration), "cudssConfigCreate");
            require_cudss(cudssDataCreate(cudss_handle, &cudss_data), "cudssDataCreate");
            constexpr std::int32_t deterministic_mode = 1;
            require_cudss(cudssConfigSet(cudss_configuration, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic_mode, sizeof(deterministic_mode)), "cudssConfigSet deterministic mode");

            require_cudss(cudssMatrixCreateCsr(&system_matrix, free_particle_count, free_particle_count, static_cast<std::int64_t>(matrix_values.values.size()), matrix_row_offsets.values.data(), nullptr, matrix_column_indices.values.data(), matrix_values.values.data(), CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_32F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO), "cudssMatrixCreateCsr");
            for (std::size_t coordinate = 0uz; coordinate < right_hand_side_matrices.size(); ++coordinate) {
                require_cudss(cudssMatrixCreateDn(&right_hand_side_matrices[coordinate], free_particle_count, 1, free_particle_count, right_hand_sides.values.data() + coordinate * free_particle_count, CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn right-hand side");
                require_cudss(cudssMatrixCreateDn(&solution_matrices[coordinate], free_particle_count, 1, free_particle_count, solutions.values.data() + coordinate * free_particle_count, CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn solution");
            }
            require_cudss(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, cudss_configuration, cudss_data, system_matrix, solution_matrices[0], right_hand_side_matrices[0]), "cudssExecute analysis");
            require_cudss(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, cudss_configuration, cudss_data, system_matrix, solution_matrices[0], right_hand_side_matrices[0]), "cudssExecute factorization");
            model.stream.sync();
        } catch (...) {
            destroy_cudss();
            throw;
        }
    }

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
            .predicted_positions = simulation::VectorField<float>(model.stream, model.particle_count),
            .projected_springs   = simulation::VectorField<float>(model.stream, model.topology.edges.size()),
        };
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>&) const {
        return {};
    }

    void Solver::forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters&, State<float>& next_state, StepCache& cache, Workspace&) const {
        position_dynamics::kernels::predict(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, gravity, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_forces), masses.values.data(), simulation::view(cache.predicted_positions));
        simulation::copy(model.stream, cache.predicted_positions, next_state.positions);
        const std::uint32_t edge_count = static_cast<std::uint32_t>(model.topology.edges.size());
        const float inverse_time_step_squared = 1.0F / (time_step * time_step);
        for (std::uint32_t iteration = 0u; iteration < global_iteration_count; ++iteration) {
            kernels::project_springs(model.stream, edge_count, model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), rest_lengths.values.data(), fixed_vertex_mask.values.data(), simulation::view(next_state.positions), simulation::view(cache.projected_springs));
            kernels::assemble_right_hand_sides(model.stream, free_particle_count, inverse_time_step_squared, spring_stiffness, free_particles.values.data(), model.topology.device.vertex_edges.offsets.values.data(), model.topology.device.vertex_edges.indices.values.data(), model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), fixed_vertex_mask.values.data(), masses.values.data(), simulation::view(cache.predicted_positions), simulation::view(fixed_positions), simulation::view(cache.projected_springs), right_hand_sides.values.data());
            for (std::size_t coordinate = 0uz; coordinate < right_hand_side_matrices.size(); ++coordinate) require_cudss(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, cudss_configuration, cudss_data, system_matrix, solution_matrices[coordinate], right_hand_side_matrices[coordinate]), "cudssExecute solve");
            kernels::scatter_solution(model.stream, free_particle_count, free_particles.values.data(), solutions.values.data(), simulation::view(next_state.positions));
        }
        position_dynamics::kernels::reconstruct_velocities(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(state.positions), simulation::view(next_state.positions), simulation::view(next_state.velocities));
    }

    void Solver::destroy_cudss() noexcept {
        for (const cudssMatrix_t solution_matrix : solution_matrices) if (solution_matrix != nullptr) cudssMatrixDestroy(solution_matrix);
        for (const cudssMatrix_t right_hand_side_matrix : right_hand_side_matrices) if (right_hand_side_matrix != nullptr) cudssMatrixDestroy(right_hand_side_matrix);
        if (system_matrix != nullptr) cudssMatrixDestroy(system_matrix);
        if (cudss_data != nullptr && cudss_handle != nullptr) cudssDataDestroy(cudss_handle, cudss_data);
        if (cudss_configuration != nullptr) cudssConfigDestroy(cudss_configuration);
        if (cudss_handle != nullptr) cudssDestroy(cudss_handle);
    }
} // namespace physica::deformables::cloth::solvers::fast_mass_spring
