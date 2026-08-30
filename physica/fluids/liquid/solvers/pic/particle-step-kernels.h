#ifndef PHYSICA_FLUIDS_LIQUID_HYBRID_PARTICLES_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_HYBRID_PARTICLES_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <fluids/grid/device.cuh>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::fluids::liquid::solvers::pic::kernels::particle_step {
    [[nodiscard]] std::size_t scan_storage_size(std::size_t count);
    [[nodiscard]] std::size_t reduction_storage_size(std::size_t count);
    void particle_speeds(::cuda::stream_ref stream, std::uint32_t particle_count, simulation::VectorView<const float> velocities, float* speeds);
    void particle_diagnostics(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, std::size_t stride, float particle_mass, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, float* values);
    void reduce_maximum(::cuda::stream_ref stream, const float* values, std::size_t count, float* output, void* scratch, std::size_t scratch_bytes);
    void reduce_sum(::cuda::stream_ref stream, const float* values, std::size_t count, float* output, void* scratch, std::size_t scratch_bytes);
    void advect(::cuda::stream_ref stream, grid::device::Grid grid, float particle_radius, bool no_slip, float time_step, std::uint32_t particle_count, simulation::VectorView<const float> grid_velocity, simulation::VectorView<float> positions, simulation::VectorView<float> velocities);
    void plan_maintenance(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, std::uint32_t minimum_particles_per_cell, std::uint32_t target_particles_per_cell, std::uint32_t maximum_particles_per_cell, simulation::VectorView<const float> positions, const std::uint32_t* cell_types, const float* level_set, std::uint32_t* raw_counts, std::uint32_t* survivor_counts, std::uint32_t* keep_flags, std::uint32_t* destinations, std::uint32_t* seed_counts, std::uint32_t* seed_offsets, std::uint32_t* totals, void* scan_scratch, std::size_t scan_scratch_bytes);
    void compact_and_seed(::cuda::stream_ref stream, grid::device::Grid grid, std::uint64_t seed, std::uint32_t particle_count, std::uint32_t survivor_count, std::uint32_t seed_count, simulation::VectorView<const float> source_positions, simulation::VectorView<const float> source_velocities, const std::uint32_t* keep_flags, const std::uint32_t* destinations, const std::uint32_t* seed_counts, const std::uint32_t* seed_offsets, simulation::VectorView<const float> grid_velocity, simulation::VectorView<float> output_positions, simulation::VectorView<float> output_velocities);
} // namespace physica::fluids::liquid::solvers::pic::kernels::particle_step

#endif
