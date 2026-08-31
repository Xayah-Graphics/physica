#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_BLOCK_PCG_BLOCK_PCG_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_BLOCK_PCG_BLOCK_PCG_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::block_pcg::kernels {
    void matvec(::cuda::stream_ref stream, std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* values, simulation::VectorView<const float> input, simulation::VectorView<float> output);
    void build_block_jacobi_inverse(::cuda::stream_ref stream, std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* values, const std::uint32_t* fixed_vertex_mask, float* inverse);
    void initialize_pcg(::cuda::stream_ref stream, std::uint32_t row_count, const float* block_jacobi_inverse, const std::uint32_t* fixed_vertex_mask, simulation::VectorView<const float> right_hand_side, simulation::VectorView<float> solution, simulation::VectorView<float> residual, simulation::VectorView<float> preconditioned_residual, simulation::VectorView<float> search_direction);
    void project(::cuda::stream_ref stream, std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, simulation::VectorView<float> vector);
    void safe_divide(::cuda::stream_ref stream, const float* numerator, const float* denominator, float* quotient);
    void update_solution_residual(::cuda::stream_ref stream, std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, const float* alpha, simulation::VectorView<const float> search_direction, simulation::VectorView<const float> matrix_product, simulation::VectorView<float> solution, simulation::VectorView<float> residual);
    void apply_block_jacobi_inverse(::cuda::stream_ref stream, std::uint32_t row_count, const float* block_jacobi_inverse, const std::uint32_t* fixed_vertex_mask, simulation::VectorView<const float> residual, simulation::VectorView<float> preconditioned_residual);
    void update_search_direction(::cuda::stream_ref stream, std::uint32_t row_count, const std::uint32_t* fixed_vertex_mask, const float* beta, simulation::VectorView<const float> preconditioned_residual, simulation::VectorView<float> search_direction);
    void sum_dot_components(::cuda::stream_ref stream, const float* components, float* result);
} // namespace physica::deformables::cloth::solvers::block_pcg::kernels

#endif
