#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_NEIGHBORHOOD_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_NEIGHBORHOOD_KERNELS_H

#include "../domain/device.h"
#include <cstddef>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::liquid::particle::cuda_detail {
    std::size_t radix_sort_storage_size(std::uint32_t count);
    void build_neighborhood(::cuda::stream_ref stream, std::uint32_t particle_count, std::uint32_t boundary_count, float support_radius, float time_step, std::uint64_t step_index, Box domain, ConstVectorView<float> positions, ConstVectorView<float> boundary_positions, ConstVectorView<float> boundary_velocities, std::uint64_t* unsorted_keys, std::uint32_t* unsorted_particle_indices, std::uint64_t* unsorted_boundary_keys, std::uint32_t* unsorted_boundary_indices, void* sort_scratch, std::size_t sort_scratch_bytes, void* boundary_sort_scratch, std::size_t boundary_sort_scratch_bytes, std::uint64_t* sorted_keys, std::uint32_t* sorted_particle_indices, std::uint32_t* cell_offsets, std::uint64_t* sorted_boundary_keys, std::uint32_t* sorted_boundary_indices, std::uint32_t* boundary_cell_offsets);
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
