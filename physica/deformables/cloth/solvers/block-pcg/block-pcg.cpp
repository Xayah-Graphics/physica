module;

#include "block-pcg-kernels.h"
#include <cublas_v2.h>
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.block_pcg;

import std;

namespace physica::deformables::cloth::solvers::block_pcg {
    BlockCsrMatrix::BlockCsrMatrix(const Model<float>& model, const std::span<const std::uint32_t> host_row_offsets, const std::span<const std::uint32_t> host_column_indices)
        : row_count(host_row_offsets.size() - 1uz),
          block_count(host_column_indices.size()),
          row_offsets(model.stream, host_row_offsets.size()),
          column_indices(model.stream, host_column_indices.size()),
          block_values(model.stream, 9uz * host_column_indices.size()) {
        ::cuda::copy_bytes(model.stream, ::cuda::std::span<const std::uint32_t>{host_row_offsets.data(), host_row_offsets.size()}, row_offsets.values);
        ::cuda::copy_bytes(model.stream, ::cuda::std::span<const std::uint32_t>{host_column_indices.data(), host_column_indices.size()}, column_indices.values);
        model.stream.sync();
    }

    BlockDiagonal::BlockDiagonal(const Model<float>& model, const std::size_t row_count) : block_values(model.stream, 9uz * row_count) {}

    Solver::Solver(const Model<float>& model, const Configuration configuration) : iteration_count(configuration.iteration_count), cublas{} {
        if (const cublasStatus_t status = cublasCreate(std::out_ptr(cublas)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasCreate failed: {}", cublasGetStatusString(status)));
        if (const cublasStatus_t status = cublasSetStream(cublas.get(), model.stream.get()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSetStream failed: {}", cublasGetStatusString(status)));
        if (const cublasStatus_t status = cublasSetPointerMode(cublas.get(), CUBLAS_POINTER_MODE_DEVICE); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSetPointerMode failed: {}", cublasGetStatusString(status)));
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>& model, const BlockCsrMatrix& matrix) const {
        return {
            .block_jacobi_inverse    = BlockDiagonal(model, matrix.row_count),
            .residual                = simulation::VectorField<float>(model.stream, matrix.row_count),
            .preconditioned_residual = simulation::VectorField<float>(model.stream, matrix.row_count),
            .search_direction        = simulation::VectorField<float>(model.stream, matrix.row_count),
            .matrix_product          = simulation::VectorField<float>(model.stream, matrix.row_count),
            .dot_components          = simulation::ScalarField<float>(model.stream, 3uz),
            .rho                     = simulation::ScalarField<float>(model.stream, 1uz),
            .next_rho                = simulation::ScalarField<float>(model.stream, 1uz),
            .denominator             = simulation::ScalarField<float>(model.stream, 1uz),
            .alpha                   = simulation::ScalarField<float>(model.stream, 1uz),
            .beta                    = simulation::ScalarField<float>(model.stream, 1uz),
        };
    }

    void Solver::clear(const Model<float>& model, BlockCsrMatrix& matrix) const {
        simulation::clear(model.stream, matrix.block_values);
    }

    void Solver::clear(const Model<float>& model, simulation::VectorField<float>& vector) const {
        simulation::clear(model.stream, vector);
    }

    void Solver::matvec(const Model<float>& model, const BlockCsrMatrix& matrix, const simulation::VectorField<float>& input, simulation::VectorField<float>& output) const {
        kernels::matvec(model.stream, static_cast<std::uint32_t>(matrix.row_count), matrix.row_offsets.values.data(), matrix.column_indices.values.data(), matrix.block_values.values.data(), simulation::view(input), simulation::view(output));
    }

    void Solver::build_block_jacobi_inverse(const Model<float>& model, const BlockCsrMatrix& matrix, const simulation::ScalarField<std::uint32_t>& fixed_vertex_mask, BlockDiagonal& inverse) const {
        kernels::build_block_jacobi_inverse(model.stream, static_cast<std::uint32_t>(matrix.row_count), matrix.row_offsets.values.data(), matrix.column_indices.values.data(), matrix.block_values.values.data(), fixed_vertex_mask.values.data(), inverse.block_values.values.data());
    }

    void Solver::solve(const Model<float>& model, const BlockCsrMatrix& matrix, const simulation::VectorField<float>& right_hand_side, const simulation::ScalarField<std::uint32_t>& fixed_vertex_mask, simulation::VectorField<float>& solution, Workspace& workspace) const {
        build_block_jacobi_inverse(model, matrix, fixed_vertex_mask, workspace.block_jacobi_inverse);
        kernels::initialize_pcg(model.stream, static_cast<std::uint32_t>(matrix.row_count), workspace.block_jacobi_inverse.block_values.values.data(), fixed_vertex_mask.values.data(), simulation::view(right_hand_side), simulation::view(solution), simulation::view(workspace.residual), simulation::view(workspace.preconditioned_residual), simulation::view(workspace.search_direction));
        dot(model, matrix.row_count, workspace.residual, workspace.preconditioned_residual, workspace.dot_components, workspace.rho);

        for (std::uint32_t iteration = 0u; iteration < iteration_count; ++iteration) {
            matvec(model, matrix, workspace.search_direction, workspace.matrix_product);
            kernels::project(model.stream, static_cast<std::uint32_t>(matrix.row_count), fixed_vertex_mask.values.data(), simulation::view(workspace.matrix_product));
            dot(model, matrix.row_count, workspace.search_direction, workspace.matrix_product, workspace.dot_components, workspace.denominator);
            kernels::safe_divide(model.stream, workspace.rho.values.data(), workspace.denominator.values.data(), workspace.alpha.values.data());
            kernels::update_solution_residual(model.stream, static_cast<std::uint32_t>(matrix.row_count), fixed_vertex_mask.values.data(), workspace.alpha.values.data(), simulation::view(workspace.search_direction), simulation::view(workspace.matrix_product), simulation::view(solution), simulation::view(workspace.residual));
            kernels::apply_block_jacobi_inverse(model.stream, static_cast<std::uint32_t>(matrix.row_count), workspace.block_jacobi_inverse.block_values.values.data(), fixed_vertex_mask.values.data(), simulation::view(workspace.residual), simulation::view(workspace.preconditioned_residual));
            dot(model, matrix.row_count, workspace.residual, workspace.preconditioned_residual, workspace.dot_components, workspace.next_rho);
            kernels::safe_divide(model.stream, workspace.next_rho.values.data(), workspace.rho.values.data(), workspace.beta.values.data());
            kernels::update_search_direction(model.stream, static_cast<std::uint32_t>(matrix.row_count), fixed_vertex_mask.values.data(), workspace.beta.values.data(), simulation::view(workspace.preconditioned_residual), simulation::view(workspace.search_direction));
            ::cuda::copy_bytes(model.stream, workspace.next_rho.values, workspace.rho.values);
        }
    }

    void Solver::CublasDeleter::operator()(cublasContext* const handle) const noexcept {
        cublasDestroy(handle);
    }

    void Solver::dot(const Model<float>& model, const std::size_t row_count, const simulation::VectorField<float>& first, const simulation::VectorField<float>& second, simulation::ScalarField<float>& components, simulation::ScalarField<float>& result) const {
        if (const cublasStatus_t status = cublasSdot(cublas.get(), static_cast<int>(row_count), first.x.data(), 1, second.x.data(), 1, components.values.data()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusString(status)));
        if (const cublasStatus_t status = cublasSdot(cublas.get(), static_cast<int>(row_count), first.y.data(), 1, second.y.data(), 1, components.values.data() + 1uz); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusString(status)));
        if (const cublasStatus_t status = cublasSdot(cublas.get(), static_cast<int>(row_count), first.z.data(), 1, second.z.data(), 1, components.values.data() + 2uz); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusString(status)));
        kernels::sum_dot_components(model.stream, components.values.data(), result.values.data());
    }
} // namespace physica::deformables::cloth::solvers::block_pcg
