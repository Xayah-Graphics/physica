#include "fast-projection-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::fast_projection::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void linearize_constraints_kernel(const std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<const float> positions, float* constraint_values, const simulation::VectorView<float> jacobian_directions, float* jacobi_inverse_diagonal) {
            const std::uint32_t constraint = blockIdx.x * blockDim.x + threadIdx.x;
            if (constraint >= constraint_count) return;
            const std::uint32_t first            = edge_first[constraint];
            const std::uint32_t second           = edge_second[constraint];
            const Vector3<float> displacement    = load(positions, second) - load(positions, first);
            const float distance                 = length(displacement);
            const float first_inverse_mass       = fixed_vertex_mask[first] == 0u ? 1.0F / masses[first] : 0.0F;
            const float second_inverse_mass      = fixed_vertex_mask[second] == 0u ? 1.0F / masses[second] : 0.0F;
            const float diagonal                 = first_inverse_mass + second_inverse_mass;
            constraint_values[constraint]        = distance - rest_lengths[constraint];
            jacobi_inverse_diagonal[constraint]  = diagonal != 0.0F ? 1.0F / diagonal : 0.0F;
            store(jacobian_directions, constraint, displacement / distance);
        }

        __global__ void initialize_pcg_kernel(const std::uint32_t constraint_count, const float* constraint_values, const float* jacobi_inverse_diagonal, float* lambdas, float* residual, float* preconditioned_residual, float* search_direction) {
            const std::uint32_t constraint = blockIdx.x * blockDim.x + threadIdx.x;
            if (constraint >= constraint_count) return;
            const float initial_residual       = -constraint_values[constraint];
            const float initial_preconditioned = jacobi_inverse_diagonal[constraint] * initial_residual;
            lambdas[constraint]                = 0.0F;
            residual[constraint]               = initial_residual;
            preconditioned_residual[constraint] = initial_preconditioned;
            search_direction[constraint]       = initial_preconditioned;
        }

        __global__ void scatter_jacobian_transpose_kernel(const std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const simulation::VectorView<const float> jacobian_directions, const float* constraint_vector, const simulation::VectorView<float> vertex_vector) {
            const std::uint32_t constraint = blockIdx.x * blockDim.x + threadIdx.x;
            if (constraint >= constraint_count) return;
            const std::uint32_t first         = edge_first[constraint];
            const std::uint32_t second        = edge_second[constraint];
            const Vector3<float> contribution = constraint_vector[constraint] * load(jacobian_directions, constraint);
            atomicAdd(vertex_vector.x + first, -contribution.x);
            atomicAdd(vertex_vector.y + first, -contribution.y);
            atomicAdd(vertex_vector.z + first, -contribution.z);
            atomicAdd(vertex_vector.x + second, contribution.x);
            atomicAdd(vertex_vector.y + second, contribution.y);
            atomicAdd(vertex_vector.z + second, contribution.z);
        }

        __global__ void gather_matrix_product_kernel(const std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<const float> jacobian_directions, const simulation::VectorView<const float> vertex_vector, float* matrix_product) {
            const std::uint32_t constraint = blockIdx.x * blockDim.x + threadIdx.x;
            if (constraint >= constraint_count) return;
            const std::uint32_t first       = edge_first[constraint];
            const std::uint32_t second      = edge_second[constraint];
            const float first_inverse_mass  = fixed_vertex_mask[first] == 0u ? 1.0F / masses[first] : 0.0F;
            const float second_inverse_mass = fixed_vertex_mask[second] == 0u ? 1.0F / masses[second] : 0.0F;
            const Vector3<float> difference = second_inverse_mass * load(vertex_vector, second) - first_inverse_mass * load(vertex_vector, first);
            matrix_product[constraint]      = dot(load(jacobian_directions, constraint), difference);
        }

        __global__ void safe_divide_kernel(const float* numerator, const float* denominator, float* quotient) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            quotient[0] = denominator[0] != 0.0F ? numerator[0] / denominator[0] : 0.0F;
        }

        __global__ void update_solution_residual_kernel(const std::uint32_t constraint_count, const float* alpha, const float* search_direction, const float* matrix_product, float* lambdas, float* residual) {
            const std::uint32_t constraint = blockIdx.x * blockDim.x + threadIdx.x;
            if (constraint >= constraint_count) return;
            lambdas[constraint] += alpha[0] * search_direction[constraint];
            residual[constraint] -= alpha[0] * matrix_product[constraint];
        }

        __global__ void apply_preconditioner_kernel(const std::uint32_t constraint_count, const float* jacobi_inverse_diagonal, const float* residual, float* preconditioned_residual) {
            const std::uint32_t constraint = blockIdx.x * blockDim.x + threadIdx.x;
            if (constraint >= constraint_count) return;
            preconditioned_residual[constraint] = jacobi_inverse_diagonal[constraint] * residual[constraint];
        }

        __global__ void update_search_direction_kernel(const std::uint32_t constraint_count, const float* beta, const float* preconditioned_residual, float* search_direction) {
            const std::uint32_t constraint = blockIdx.x * blockDim.x + threadIdx.x;
            if (constraint >= constraint_count) return;
            search_direction[constraint] = preconditioned_residual[constraint] + beta[0] * search_direction[constraint];
        }

        __global__ void apply_position_correction_kernel(const std::uint32_t particle_count, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<const float> vertex_correction, const simulation::VectorView<float> positions) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float inverse_mass = fixed_vertex_mask[particle] == 0u ? 1.0F / masses[particle] : 0.0F;
            store(positions, particle, load(positions, particle) + inverse_mass * load(vertex_correction, particle));
        }
    } // namespace

    void linearize_constraints(const ::cuda::stream_ref stream, const std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<const float> positions, float* constraint_values, const simulation::VectorView<float> jacobian_directions, float* jacobi_inverse_diagonal) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(constraint_count), linearize_constraints_kernel, constraint_count, edge_first, edge_second, rest_lengths, fixed_vertex_mask, masses, positions, constraint_values, jacobian_directions, jacobi_inverse_diagonal);
    }

    void initialize_pcg(const ::cuda::stream_ref stream, const std::uint32_t constraint_count, const float* constraint_values, const float* jacobi_inverse_diagonal, float* lambdas, float* residual, float* preconditioned_residual, float* search_direction) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(constraint_count), initialize_pcg_kernel, constraint_count, constraint_values, jacobi_inverse_diagonal, lambdas, residual, preconditioned_residual, search_direction);
    }

    void scatter_jacobian_transpose(const ::cuda::stream_ref stream, const std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const simulation::VectorView<const float> jacobian_directions, const float* constraint_vector, const simulation::VectorView<float> vertex_vector) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(constraint_count), scatter_jacobian_transpose_kernel, constraint_count, edge_first, edge_second, jacobian_directions, constraint_vector, vertex_vector);
    }

    void gather_matrix_product(const ::cuda::stream_ref stream, const std::uint32_t constraint_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<const float> jacobian_directions, const simulation::VectorView<const float> vertex_vector, float* matrix_product) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(constraint_count), gather_matrix_product_kernel, constraint_count, edge_first, edge_second, fixed_vertex_mask, masses, jacobian_directions, vertex_vector, matrix_product);
    }

    void safe_divide(const ::cuda::stream_ref stream, const float* numerator, const float* denominator, float* quotient) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(1u), safe_divide_kernel, numerator, denominator, quotient);
    }

    void update_solution_residual(const ::cuda::stream_ref stream, const std::uint32_t constraint_count, const float* alpha, const float* search_direction, const float* matrix_product, float* lambdas, float* residual) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(constraint_count), update_solution_residual_kernel, constraint_count, alpha, search_direction, matrix_product, lambdas, residual);
    }

    void apply_preconditioner(const ::cuda::stream_ref stream, const std::uint32_t constraint_count, const float* jacobi_inverse_diagonal, const float* residual, float* preconditioned_residual) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(constraint_count), apply_preconditioner_kernel, constraint_count, jacobi_inverse_diagonal, residual, preconditioned_residual);
    }

    void update_search_direction(const ::cuda::stream_ref stream, const std::uint32_t constraint_count, const float* beta, const float* preconditioned_residual, float* search_direction) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(constraint_count), update_search_direction_kernel, constraint_count, beta, preconditioned_residual, search_direction);
    }

    void apply_position_correction(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<const float> vertex_correction, const simulation::VectorView<float> positions) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), apply_position_correction_kernel, particle_count, fixed_vertex_mask, masses, vertex_correction, positions);
    }
} // namespace physica::deformables::cloth::solvers::fast_projection::kernels
