#include "block-pcg-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::block_pcg::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __device__ Vector3<float> multiply_block(const float* const block, const Vector3<float> vector) {
            return {
                .x = block[0] * vector.x + block[1] * vector.y + block[2] * vector.z,
                .y = block[3] * vector.x + block[4] * vector.y + block[5] * vector.z,
                .z = block[6] * vector.x + block[7] * vector.y + block[8] * vector.z,
            };
        }

        __global__ void matvec_kernel(const std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* values, const simulation::VectorView<const float> input, const simulation::VectorView<float> output) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            Vector3<float> result{};
            for (std::uint32_t block = row_offsets[row]; block < row_offsets[row + 1u]; ++block) result = result + multiply_block(values + 9u * block, load(input, column_indices[block]));
            store(output, row, result);
        }

        __global__ void build_block_jacobi_inverse_kernel(const std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* values, const std::uint32_t* fixed_vertex_mask, float* inverse) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            float* const row_inverse = inverse + 9u * row;
            if (fixed_vertex_mask[row] != 0u) {
                for (std::uint32_t entry = 0u; entry < 9u; ++entry) row_inverse[entry] = 0.0F;
                return;
            }

            std::uint32_t diagonal_block = row_offsets[row];
            for (std::uint32_t block = row_offsets[row]; block < row_offsets[row + 1u]; ++block) {
                if (column_indices[block] != row) continue;
                diagonal_block = block;
                break;
            }
            const float* const diagonal = values + 9u * diagonal_block;
            const float determinant = diagonal[0] * (diagonal[4] * diagonal[8] - diagonal[5] * diagonal[7]) + diagonal[1] * (diagonal[5] * diagonal[6] - diagonal[3] * diagonal[8]) + diagonal[2] * (diagonal[3] * diagonal[7] - diagonal[4] * diagonal[6]);
            const float inverse_determinant = 1.0F / determinant;
            row_inverse[0] = inverse_determinant * (diagonal[4] * diagonal[8] - diagonal[5] * diagonal[7]);
            row_inverse[1] = inverse_determinant * (diagonal[2] * diagonal[7] - diagonal[1] * diagonal[8]);
            row_inverse[2] = inverse_determinant * (diagonal[1] * diagonal[5] - diagonal[2] * diagonal[4]);
            row_inverse[3] = inverse_determinant * (diagonal[5] * diagonal[6] - diagonal[3] * diagonal[8]);
            row_inverse[4] = inverse_determinant * (diagonal[0] * diagonal[8] - diagonal[2] * diagonal[6]);
            row_inverse[5] = inverse_determinant * (diagonal[2] * diagonal[3] - diagonal[0] * diagonal[5]);
            row_inverse[6] = inverse_determinant * (diagonal[3] * diagonal[7] - diagonal[4] * diagonal[6]);
            row_inverse[7] = inverse_determinant * (diagonal[1] * diagonal[6] - diagonal[0] * diagonal[7]);
            row_inverse[8] = inverse_determinant * (diagonal[0] * diagonal[4] - diagonal[1] * diagonal[3]);
        }

        __global__ void initialize_pcg_kernel(const std::uint32_t row_count, const float* block_jacobi_inverse, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> right_hand_side, const simulation::VectorView<float> solution, const simulation::VectorView<float> residual, const simulation::VectorView<float> preconditioned_residual, const simulation::VectorView<float> search_direction) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            if (fixed_vertex_mask[row] != 0u) {
                store(solution, row, {});
                store(residual, row, {});
                store(preconditioned_residual, row, {});
                store(search_direction, row, {});
                return;
            }
            const Vector3<float> initial_residual = load(right_hand_side, row);
            const Vector3<float> initial_preconditioned_residual = multiply_block(block_jacobi_inverse + 9u * row, initial_residual);
            store(solution, row, {});
            store(residual, row, initial_residual);
            store(preconditioned_residual, row, initial_preconditioned_residual);
            store(search_direction, row, initial_preconditioned_residual);
        }

        __global__ void project_kernel(const std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<float> vector) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            if (fixed_vertex_mask[row] != 0u) store(vector, row, {});
        }

        __global__ void safe_divide_kernel(const float* numerator, const float* denominator, float* quotient) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            quotient[0] = denominator[0] != 0.0F ? numerator[0] / denominator[0] : 0.0F;
        }

        __global__ void update_solution_residual_kernel(const std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, const float* alpha, const simulation::VectorView<const float> search_direction, const simulation::VectorView<const float> matrix_product, const simulation::VectorView<float> solution, const simulation::VectorView<float> residual) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            if (fixed_vertex_mask[row] != 0u) {
                store(solution, row, {});
                store(residual, row, {});
                return;
            }
            store(solution, row, load(solution, row) + alpha[0] * load(search_direction, row));
            store(residual, row, load(residual, row) - alpha[0] * load(matrix_product, row));
        }

        __global__ void apply_block_jacobi_inverse_kernel(const std::uint32_t row_count, const float* block_jacobi_inverse, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> residual, const simulation::VectorView<float> preconditioned_residual) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            if (fixed_vertex_mask[row] != 0u) {
                store(preconditioned_residual, row, {});
                return;
            }
            store(preconditioned_residual, row, multiply_block(block_jacobi_inverse + 9u * row, load(residual, row)));
        }

        __global__ void update_search_direction_kernel(const std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, const float* beta, const simulation::VectorView<const float> preconditioned_residual, const simulation::VectorView<float> search_direction) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            if (fixed_vertex_mask[row] != 0u) {
                store(search_direction, row, {});
                return;
            }
            store(search_direction, row, load(preconditioned_residual, row) + beta[0] * load(search_direction, row));
        }

        __global__ void sum_dot_components_kernel(const float* components, float* result) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            result[0] = components[0] + components[1] + components[2];
        }
    } // namespace

    void matvec(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* values, const simulation::VectorView<const float> input, const simulation::VectorView<float> output) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), matvec_kernel, row_count, row_offsets, column_indices, values, input, output);
    }

    void build_block_jacobi_inverse(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* values, const std::uint32_t* fixed_vertex_mask, float* inverse) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), build_block_jacobi_inverse_kernel, row_count, row_offsets, column_indices, values, fixed_vertex_mask, inverse);
    }

    void initialize_pcg(const ::cuda::stream_ref stream, const std::uint32_t row_count, const float* block_jacobi_inverse, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> right_hand_side, const simulation::VectorView<float> solution, const simulation::VectorView<float> residual, const simulation::VectorView<float> preconditioned_residual, const simulation::VectorView<float> search_direction) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), initialize_pcg_kernel, row_count, block_jacobi_inverse, fixed_vertex_mask, right_hand_side, solution, residual, preconditioned_residual, search_direction);
    }

    void project(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<float> vector) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), project_kernel, row_count, fixed_vertex_mask, vector);
    }

    void safe_divide(const ::cuda::stream_ref stream, const float* numerator, const float* denominator, float* quotient) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(1u), safe_divide_kernel, numerator, denominator, quotient);
    }

    void update_solution_residual(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, const float* alpha, const simulation::VectorView<const float> search_direction, const simulation::VectorView<const float> matrix_product, const simulation::VectorView<float> solution, const simulation::VectorView<float> residual) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), update_solution_residual_kernel, row_count, fixed_vertex_mask, alpha, search_direction, matrix_product, solution, residual);
    }

    void apply_block_jacobi_inverse(const ::cuda::stream_ref stream, const std::uint32_t row_count, const float* block_jacobi_inverse, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> residual, const simulation::VectorView<float> preconditioned_residual) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), apply_block_jacobi_inverse_kernel, row_count, block_jacobi_inverse, fixed_vertex_mask, residual, preconditioned_residual);
    }

    void update_search_direction(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, const float* beta, const simulation::VectorView<const float> preconditioned_residual, const simulation::VectorView<float> search_direction) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), update_search_direction_kernel, row_count, fixed_vertex_mask, beta, preconditioned_residual, search_direction);
    }

    void sum_dot_components(const ::cuda::stream_ref stream, const float* components, float* result) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(1u), sum_dot_components_kernel, components, result);
    }
} // namespace physica::deformables::cloth::solvers::block_pcg::kernels
