#ifndef PHYSICA_FLUIDS_LIQUID_HYBRID_TRANSFER_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_HYBRID_TRANSFER_KERNELS_H

#include <physica/field/device.cuh>
#include <physica/fluids/grid/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::liquid::pic::kernels::transfer {
    void flip_particle_to_grid(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, field::VectorView<const float> positions, field::VectorView<const float> velocities, field::VectorView<float> momentum, field::VectorView<float> mass);
    void flip_grid_to_particle(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, float flip_ratio, field::VectorView<const float> positions, field::VectorView<const float> input_velocities, field::VectorView<const float> old_grid_velocity, field::VectorView<const float> new_grid_velocity, field::VectorView<float> output_velocities);
    void apic_particle_to_grid(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, field::VectorView<const float> positions, field::VectorView<const float> velocities, field::Matrix3View<const float> affine, field::VectorView<float> momentum, field::VectorView<float> mass);
    void apic_grid_to_particle(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, float affine_ratio, field::VectorView<const float> positions, field::VectorView<const float> grid_velocity, field::VectorView<float> output_velocities, field::Matrix3View<float> output_affine);
    void apic_compact_and_seed(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t source_particle_count, std::uint32_t survivor_count, std::uint32_t seed_count, float affine_ratio, field::VectorView<const float> compacted_positions, field::VectorView<const float> grid_velocity, field::Matrix3View<const float> source_affine, const std::uint32_t* keep_flags, const std::uint32_t* destinations, field::Matrix3View<float> output_affine);
} // namespace physica::fluids::liquid::pic::kernels::transfer

#endif
