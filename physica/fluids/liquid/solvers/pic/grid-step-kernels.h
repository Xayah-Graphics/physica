#ifndef PHYSICA_FLUIDS_LIQUID_HYBRID_FREE_SURFACE_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_HYBRID_FREE_SURFACE_KERNELS_H

#include <physica/field/device.cuh>
#include <physica/fluids/grid/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::liquid::pic::kernels::grid_step {
    void classify(::cuda::stream_ref stream, grid::device::Grid grid, std::uint32_t particle_count, float level_set_radius, field::VectorView<const float> positions, std::uint32_t* particle_counts, std::uint32_t* cell_types, float* level_set);
    void normalize(::cuda::stream_ref stream, grid::device::Grid grid, field::VectorView<const float> mass, field::VectorView<float> momentum, field::VectorView<std::uint32_t> valid);
    void extrapolate_layer(::cuda::stream_ref stream, grid::device::Grid grid, field::VectorView<const float> input, field::VectorView<const std::uint32_t> input_valid, field::VectorView<float> output, field::VectorView<std::uint32_t> output_valid);
    void add_force_and_constrain(::cuda::stream_ref stream, grid::device::Grid grid, bool no_slip, float time_step, Vector3<float> acceleration, const std::uint32_t* cell_types, field::VectorView<float> velocity);
    void mark_fluid_faces(::cuda::stream_ref stream, grid::device::Grid grid, bool no_slip, const std::uint32_t* cell_types, field::VectorView<std::uint32_t> valid);
    void compute_divergence(::cuda::stream_ref stream, grid::device::Grid grid, const std::uint32_t* cell_types, field::VectorView<const float> velocity, float* divergence, float* metrics);
} // namespace physica::fluids::liquid::pic::kernels::grid_step

#endif
