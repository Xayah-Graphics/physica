module;

#include "fast-projection-kernels.h"
#include "../position-dynamics-kernels.h"
#include <cublas_v2.h>
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.fast_projection;

import std;

namespace physica::deformables::cloth::solvers::fast_projection {
    Solver::Solver(const Model<float>& model, Configuration configuration)
        : time_step(configuration.time_step), outer_iteration_count(configuration.outer_iteration_count), pcg_iteration_count(configuration.pcg_iteration_count), gravity(configuration.gravity), cublas{}, rest_lengths(model.stream, model.topology.edges.size()), fixed_vertex_mask(model.stream, model.particle_count), fixed_positions(model.stream, model.particle_count) {
        if (const cublasStatus_t status = cublasCreate(std::out_ptr(cublas)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasCreate failed: {}", cublasGetStatusString(status)));
        if (const cublasStatus_t status = cublasSetStream(cublas.get(), model.stream.get()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSetStream failed: {}", cublasGetStatusString(status)));
        if (const cublasStatus_t status = cublasSetPointerMode(cublas.get(), CUBLAS_POINTER_MODE_DEVICE); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSetPointerMode failed: {}", cublasGetStatusString(status)));

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

    Solver::StepCache Solver::allocate_step_cache(const Model<float>& model) const {
        return {
            .constraint_values        = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
            .jacobian_directions      = simulation::VectorField<float>(model.stream, model.topology.edges.size()),
            .jacobi_inverse_diagonal  = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
            .lambdas                  = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
        };
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>& model) const {
        return {
            .residual                = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
            .preconditioned_residual = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
            .search_direction        = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
            .matrix_product          = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
            .vertex_product          = simulation::VectorField<float>(model.stream, model.particle_count),
            .rho                     = simulation::ScalarField<float>(model.stream, 1uz),
            .next_rho                = simulation::ScalarField<float>(model.stream, 1uz),
            .matrix_denominator      = simulation::ScalarField<float>(model.stream, 1uz),
            .alpha                   = simulation::ScalarField<float>(model.stream, 1uz),
            .beta                    = simulation::ScalarField<float>(model.stream, 1uz),
        };
    }

    void Solver::forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters& parameters, State<float>& next_state, StepCache& cache, Workspace& workspace) const {
        position_dynamics::kernels::predict(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, gravity, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_forces), parameters.masses.values.data(), simulation::view(next_state.positions));
        if (!model.topology.edges.empty()) {
            const std::uint32_t constraint_count = static_cast<std::uint32_t>(model.topology.edges.size());
            for (std::uint32_t outer_iteration = 0u; outer_iteration < outer_iteration_count; ++outer_iteration) {
                kernels::linearize_constraints(model.stream, constraint_count, model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), rest_lengths.values.data(), fixed_vertex_mask.values.data(), parameters.masses.values.data(), simulation::view(next_state.positions), cache.constraint_values.values.data(), simulation::view(cache.jacobian_directions), cache.jacobi_inverse_diagonal.values.data());
                kernels::initialize_pcg(model.stream, constraint_count, cache.constraint_values.values.data(), cache.jacobi_inverse_diagonal.values.data(), cache.lambdas.values.data(), workspace.residual.values.data(), workspace.preconditioned_residual.values.data(), workspace.search_direction.values.data());
                if (const cublasStatus_t status = cublasSdot(cublas.get(), static_cast<int>(constraint_count), workspace.residual.values.data(), 1, workspace.preconditioned_residual.values.data(), 1, workspace.rho.values.data()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusString(status)));

                for (std::uint32_t pcg_iteration = 0u; pcg_iteration < pcg_iteration_count; ++pcg_iteration) {
                    simulation::clear(model.stream, workspace.vertex_product);
                    kernels::scatter_jacobian_transpose(model.stream, constraint_count, model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), simulation::view(cache.jacobian_directions), workspace.search_direction.values.data(), simulation::view(workspace.vertex_product));
                    kernels::gather_matrix_product(model.stream, constraint_count, model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), fixed_vertex_mask.values.data(), parameters.masses.values.data(), simulation::view(cache.jacobian_directions), simulation::view(workspace.vertex_product), workspace.matrix_product.values.data());
                    if (const cublasStatus_t status = cublasSdot(cublas.get(), static_cast<int>(constraint_count), workspace.search_direction.values.data(), 1, workspace.matrix_product.values.data(), 1, workspace.matrix_denominator.values.data()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusString(status)));
                    kernels::safe_divide(model.stream, workspace.rho.values.data(), workspace.matrix_denominator.values.data(), workspace.alpha.values.data());
                    kernels::update_solution_residual(model.stream, constraint_count, workspace.alpha.values.data(), workspace.search_direction.values.data(), workspace.matrix_product.values.data(), cache.lambdas.values.data(), workspace.residual.values.data());
                    kernels::apply_preconditioner(model.stream, constraint_count, cache.jacobi_inverse_diagonal.values.data(), workspace.residual.values.data(), workspace.preconditioned_residual.values.data());
                    if (const cublasStatus_t status = cublasSdot(cublas.get(), static_cast<int>(constraint_count), workspace.residual.values.data(), 1, workspace.preconditioned_residual.values.data(), 1, workspace.next_rho.values.data()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusString(status)));
                    kernels::safe_divide(model.stream, workspace.next_rho.values.data(), workspace.rho.values.data(), workspace.beta.values.data());
                    kernels::update_search_direction(model.stream, constraint_count, workspace.beta.values.data(), workspace.preconditioned_residual.values.data(), workspace.search_direction.values.data());
                    ::cuda::copy_bytes(model.stream, workspace.next_rho.values, workspace.rho.values);
                }

                simulation::clear(model.stream, workspace.vertex_product);
                kernels::scatter_jacobian_transpose(model.stream, constraint_count, model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), simulation::view(cache.jacobian_directions), cache.lambdas.values.data(), simulation::view(workspace.vertex_product));
                kernels::apply_position_correction(model.stream, static_cast<std::uint32_t>(model.particle_count), fixed_vertex_mask.values.data(), parameters.masses.values.data(), simulation::view(workspace.vertex_product), simulation::view(next_state.positions));
            }
        }
        position_dynamics::kernels::reconstruct_velocities(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(state.positions), simulation::view(next_state.positions), simulation::view(next_state.velocities));
    }

    void Solver::CublasDeleter::operator()(cublasContext* const handle) const noexcept {
        cublasDestroy(handle);
    }
} // namespace physica::deformables::cloth::solvers::fast_projection
