#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_FAST_PROJECTION_FAST_PROJECTION_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_FAST_PROJECTION_FAST_PROJECTION_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::fast_projection::kernels {
    void linearize_constraints(::cuda::stream_ref stream, std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, simulation::VectorView<const float> positions, float* constraint_values, simulation::VectorView<float> jacobian_directions, float* jacobi_inverse_diagonal);
    void initialize_pcg(::cuda::stream_ref stream, std::uint32_t constraint_count, const float* constraint_values, const float* jacobi_inverse_diagonal, float* lambdas, float* residual, float* preconditioned_residual, float* search_direction);
    void scatter_jacobian_transpose(::cuda::stream_ref stream, std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, simulation::VectorView<const float> jacobian_directions, const float* constraint_vector, simulation::VectorView<float> vertex_vector);
    void gather_matrix_product(::cuda::stream_ref stream, std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* fixed_vertex_mask, const float* masses, simulation::VectorView<const float> jacobian_directions, simulation::VectorView<const float> vertex_vector, float* matrix_product);
    void safe_divide(::cuda::stream_ref stream, const float* numerator, const float* denominator, float* quotient);
    void update_solution_residual(::cuda::stream_ref stream, std::uint32_t constraint_count, const float* alpha, const float* search_direction, const float* matrix_product, float* lambdas, float* residual);
    void apply_preconditioner(::cuda::stream_ref stream, std::uint32_t constraint_count, const float* jacobi_inverse_diagonal, const float* residual, float* preconditioned_residual);
    void update_search_direction(::cuda::stream_ref stream, std::uint32_t constraint_count, const float* beta, const float* preconditioned_residual, float* search_direction);
    void apply_position_correction(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* fixed_vertex_mask, const float* masses, simulation::VectorView<const float> vertex_correction, simulation::VectorView<float> positions);
} // namespace physica::deformables::cloth::solvers::fast_projection::kernels

#endif
