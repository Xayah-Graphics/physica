#include "provot-strain-limit-kernels.h"
#include <cuda/launch>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void initialize_kernel(const std::uint32_t particle_count, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> integrated_positions, const simulation::VectorView<float> projected_positions) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(projected_positions, particle, fixed_vertex_mask[particle] != 0u ? load(fixed_positions, particle) : load(integrated_positions, particle));
        }

        __global__ void project_kernel(const std::uint32_t edge_count, const std::uint32_t color_offset, const std::uint32_t* colored_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* maximum_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<float> projected_positions) {
            const std::uint32_t local_edge = blockIdx.x * blockDim.x + threadIdx.x;
            if (local_edge >= edge_count) return;
            const std::uint32_t edge             = colored_edges[color_offset + local_edge];
            const std::uint32_t first            = edge_first[edge];
            const std::uint32_t second           = edge_second[edge];
            const Vector3<float> first_position  = load(projected_positions, first);
            const Vector3<float> second_position = load(projected_positions, second);
            const Vector3<float> displacement    = second_position - first_position;
            const float current_length           = length(displacement);
            if (current_length <= maximum_lengths[edge]) return;

            const float first_inverse_mass  = fixed_vertex_mask[first] == 0u ? 1.0F / masses[first] : 0.0F;
            const float second_inverse_mass = fixed_vertex_mask[second] == 0u ? 1.0F / masses[second] : 0.0F;
            const float inverse_mass_sum    = first_inverse_mass + second_inverse_mass;
            if (inverse_mass_sum == 0.0F) return;
            const Vector3<float> correction = ((current_length - maximum_lengths[edge]) / current_length) * displacement;
            store(projected_positions, first, first_position + (first_inverse_mass / inverse_mass_sum) * correction);
            store(projected_positions, second, second_position - (second_inverse_mass / inverse_mass_sum) * correction);
        }

        __global__ void reconstruct_velocities_kernel(const std::uint32_t particle_count, const float inverse_time_step, const simulation::VectorView<const float> previous_positions, const simulation::VectorView<const float> projected_positions, const simulation::VectorView<float> reconstructed_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(reconstructed_velocities, particle, inverse_time_step * (load(projected_positions, particle) - load(previous_positions, particle)));
        }
    } // namespace

    void provot_strain_limit_initialize(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> integrated_positions, const simulation::VectorView<float> projected_positions) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), initialize_kernel, particle_count, fixed_vertex_mask, fixed_positions, integrated_positions, projected_positions);
    }

    void provot_strain_limit_project(const ::cuda::stream_ref stream, const std::uint32_t edge_count, const std::uint32_t color_offset, const std::uint32_t* colored_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* maximum_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<float> projected_positions) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(edge_count), project_kernel, edge_count, color_offset, colored_edges, edge_first, edge_second, maximum_lengths, fixed_vertex_mask, masses, projected_positions);
    }

    void provot_strain_limit_reconstruct_velocities(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> previous_positions, const simulation::VectorView<const float> projected_positions, const simulation::VectorView<float> reconstructed_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), reconstruct_velocities_kernel, particle_count, 1.0F / time_step, previous_positions, projected_positions, reconstructed_velocities);
    }
} // namespace physica::deformables::cloth::kernels
