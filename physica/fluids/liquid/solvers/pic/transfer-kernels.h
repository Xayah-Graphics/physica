#ifndef PHYSICA_FLUIDS_LIQUID_HYBRID_TRANSFER_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_HYBRID_TRANSFER_KERNELS_H

#include <cstdint>
#include <fluids/grid/device.cuh>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::fluids::liquid::solvers::pic::kernels::transfer {
    void flip_particle_to_grid(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::VectorView<float> momentum, simulation::VectorView<float> mass);
    void flip_grid_to_particle(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, float flip_ratio, simulation::VectorView<const float> positions, simulation::VectorView<const float> input_velocities, simulation::VectorView<const float> old_grid_velocity, simulation::VectorView<const float> new_grid_velocity, simulation::VectorView<float> output_velocities);
    void apic_particle_to_grid(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::Matrix3View<const float> affine, simulation::VectorView<float> momentum, simulation::VectorView<float> mass);
    void apic_grid_to_particle(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, float affine_ratio, simulation::VectorView<const float> positions, simulation::VectorView<const float> grid_velocity, simulation::VectorView<float> output_velocities, simulation::Matrix3View<float> output_affine);
    void apic_compact_and_seed(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t source_particle_count, std::uint32_t survivor_count, std::uint32_t seed_count, float affine_ratio, simulation::VectorView<const float> compacted_positions, simulation::VectorView<const float> grid_velocity, simulation::Matrix3View<const float> source_affine, const std::uint32_t* keep_flags, const std::uint32_t* destinations, simulation::Matrix3View<float> output_affine);
} // namespace physica::fluids::liquid::solvers::pic::kernels::transfer

#endif
