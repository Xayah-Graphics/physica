module;

#include <cublas_v2.h>
#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.block_pcg;

import std;
import physica.deformables.cloth.model;

export namespace physica::deformables::cloth::solvers::block_pcg {
    struct BlockCsrMatrix final {
        const std::size_t row_count;
        const std::size_t block_count;
        simulation::ScalarField<std::uint32_t> row_offsets;
        simulation::ScalarField<std::uint32_t> column_indices;
        simulation::ScalarField<float> block_values;

        BlockCsrMatrix(const Model<float>& model, std::span<const std::uint32_t> row_offsets, std::span<const std::uint32_t> column_indices);

        BlockCsrMatrix(const BlockCsrMatrix&)            = delete;
        BlockCsrMatrix& operator=(const BlockCsrMatrix&) = delete;
        BlockCsrMatrix(BlockCsrMatrix&&)                 = default;
        BlockCsrMatrix& operator=(BlockCsrMatrix&&)      = delete;
    };

    struct BlockDiagonal final {
        simulation::ScalarField<float> block_values;

        BlockDiagonal(const Model<float>& model, std::size_t row_count);
    };

    struct Solver final {
        struct Configuration final {
            std::uint32_t iteration_count;
        };

        struct Workspace final {
            BlockDiagonal block_jacobi_inverse;
            simulation::VectorField<float> residual;
            simulation::VectorField<float> preconditioned_residual;
            simulation::VectorField<float> search_direction;
            simulation::VectorField<float> matrix_product;
            simulation::ScalarField<float> dot_components;
            simulation::ScalarField<float> rho;
            simulation::ScalarField<float> next_rho;
            simulation::ScalarField<float> denominator;
            simulation::ScalarField<float> alpha;
            simulation::ScalarField<float> beta;
        };

        Solver(const Model<float>& model, Configuration configuration);

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] Workspace allocate_workspace(const Model<float>& model, const BlockCsrMatrix& matrix) const;

        void clear(const Model<float>& model, BlockCsrMatrix& matrix) const;
        void clear(const Model<float>& model, simulation::VectorField<float>& vector) const;
        void matvec(const Model<float>& model, const BlockCsrMatrix& matrix, const simulation::VectorField<float>& input, simulation::VectorField<float>& output) const;
        void build_block_jacobi_inverse(const Model<float>& model, const BlockCsrMatrix& matrix, const simulation::ScalarField<std::uint32_t>& fixed_vertex_mask, BlockDiagonal& inverse) const;
        void solve(const Model<float>& model, const BlockCsrMatrix& matrix, const simulation::VectorField<float>& right_hand_side, const simulation::ScalarField<std::uint32_t>& fixed_vertex_mask, simulation::VectorField<float>& solution, Workspace& workspace) const;

    private:
        struct CublasDeleter final {
            void operator()(cublasContext* handle) const noexcept;
        };

        void dot(const Model<float>& model, std::size_t row_count, const simulation::VectorField<float>& first, const simulation::VectorField<float>& second, simulation::ScalarField<float>& components, simulation::ScalarField<float>& result) const;

        const std::uint32_t iteration_count;
        std::unique_ptr<cublasContext, CublasDeleter> cublas;
    };
} // namespace physica::deformables::cloth::solvers::block_pcg
