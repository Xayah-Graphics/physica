#include "pbd-kernels.h"
#include <cuda/launch>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::pbd::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void project_distance_kernel(const std::uint32_t edge_count, const std::uint32_t color_offset, const std::uint32_t* colored_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<float> positions) {
            const std::uint32_t local_edge = blockIdx.x * blockDim.x + threadIdx.x;
            if (local_edge >= edge_count) return;
            const std::uint32_t edge             = colored_edges[color_offset + local_edge];
            const std::uint32_t first            = edge_first[edge];
            const std::uint32_t second           = edge_second[edge];
            const float first_inverse_mass       = fixed_vertex_mask[first] == 0u ? 1.0F / masses[first] : 0.0F;
            const float second_inverse_mass      = fixed_vertex_mask[second] == 0u ? 1.0F / masses[second] : 0.0F;
            const float inverse_mass_sum         = first_inverse_mass + second_inverse_mass;
            if (inverse_mass_sum == 0.0F) return;
            const Vector3<float> first_position  = load(positions, first);
            const Vector3<float> second_position = load(positions, second);
            const Vector3<float> displacement    = second_position - first_position;
            const float current_length           = length(displacement);
            const Vector3<float> correction      = ((current_length - rest_lengths[edge]) / current_length) * displacement;
            store(positions, first, first_position + (first_inverse_mass / inverse_mass_sum) * correction);
            store(positions, second, second_position - (second_inverse_mass / inverse_mass_sum) * correction);
        }

    } // namespace

    void project_distance(const ::cuda::stream_ref stream, const std::uint32_t edge_count, const std::uint32_t color_offset, const std::uint32_t* colored_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<float> positions) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(edge_count), project_distance_kernel, edge_count, color_offset, colored_edges, edge_first, edge_second, rest_lengths, fixed_vertex_mask, masses, positions);
    }
} // namespace physica::deformables::cloth::solvers::pbd::kernels
