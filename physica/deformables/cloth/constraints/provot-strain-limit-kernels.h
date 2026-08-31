#ifndef PHYSICA_DEFORMABLES_CLOTH_CONSTRAINTS_PROVOT_STRAIN_LIMIT_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_CONSTRAINTS_PROVOT_STRAIN_LIMIT_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::kernels {
    void provot_strain_limit_initialize(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* fixed_vertex_mask, simulation::VectorView<const float> fixed_positions, simulation::VectorView<const float> integrated_positions, simulation::VectorView<float> projected_positions);
    void provot_strain_limit_project(::cuda::stream_ref stream, std::uint32_t edge_count, std::uint32_t color_offset, const std::uint32_t* colored_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* maximum_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, simulation::VectorView<float> projected_positions);
    void provot_strain_limit_reconstruct_velocities(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> previous_positions, simulation::VectorView<const float> projected_positions, simulation::VectorView<float> reconstructed_velocities);
} // namespace physica::deformables::cloth::kernels

#endif
