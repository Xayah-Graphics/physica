#include "xpbd-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::xpbd::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void project_kernel(const std::uint32_t local_constraint_count, const std::uint32_t color_offset, const float inverse_time_step_squared, const std::uint32_t* colored_constraints, const std::uint32_t* constraint_first, const std::uint32_t* constraint_second, const float* rest_lengths, const float* compliances, const std::uint32_t* fixed_vertex_mask, const float* masses, float* lambdas, const simulation::VectorView<float> positions) {
            const std::uint32_t local_constraint = blockIdx.x * blockDim.x + threadIdx.x;
            if (local_constraint >= local_constraint_count) return;
            const std::uint32_t constraint = colored_constraints[color_offset + local_constraint];
            const std::uint32_t first      = constraint_first[constraint];
            const std::uint32_t second     = constraint_second[constraint];
            const Vector3<float> first_position  = load(positions, first);
            const Vector3<float> second_position = load(positions, second);
            const Vector3<float> displacement    = second_position - first_position;
            const float distance                 = length(displacement);
            const Vector3<float> direction       = displacement / distance;
            const float first_inverse_mass       = fixed_vertex_mask[first] == 0u ? 1.0F / masses[first] : 0.0F;
            const float second_inverse_mass      = fixed_vertex_mask[second] == 0u ? 1.0F / masses[second] : 0.0F;
            const float scaled_compliance        = compliances[constraint] * inverse_time_step_squared;
            const float denominator               = first_inverse_mass + second_inverse_mass + scaled_compliance;
            if (denominator == 0.0F) return;
            const float delta_lambda = (rest_lengths[constraint] - distance - scaled_compliance * lambdas[constraint]) / denominator;
            lambdas[constraint] += delta_lambda;
            store(positions, first, first_position - first_inverse_mass * delta_lambda * direction);
            store(positions, second, second_position + second_inverse_mass * delta_lambda * direction);
        }

    } // namespace

    void project(const ::cuda::stream_ref stream, const std::uint32_t constraint_count, const std::uint32_t color_offset, const float inverse_time_step_squared, const std::uint32_t* colored_constraints, const std::uint32_t* constraint_first, const std::uint32_t* constraint_second, const float* rest_lengths, const float* compliances, const std::uint32_t* fixed_vertex_mask, const float* masses, float* lambdas, const simulation::VectorView<float> positions) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(constraint_count), project_kernel, constraint_count, color_offset, inverse_time_step_squared, colored_constraints, constraint_first, constraint_second, rest_lengths, compliances, fixed_vertex_mask, masses, lambdas, positions);
    }
} // namespace physica::deformables::cloth::solvers::xpbd::kernels
