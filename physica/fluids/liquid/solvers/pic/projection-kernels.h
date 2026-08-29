#ifndef PHYSICA_FLUIDS_LIQUID_HYBRID_PROJECTION_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_HYBRID_PROJECTION_KERNELS_H

#include <physica/fluids/grid/device.cuh>
#include <cstddef>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::liquid::pic::kernels::projection {
    [[nodiscard]] std::size_t reduction_storage_size(std::size_t count);

    void project(
        ::cuda::stream_ref stream,
        grid::device::Grid grid,
        float time_step,
        float density,
        std::uint32_t maximum_iterations,
        float tolerance,
        const std::uint32_t* cell_types,
        const float* level_set,
        field::VectorView<const float> input_velocity,
        field::VectorView<float> output_velocity,
        float* rhs,
        float* pressure,
        float* diagonal,
        float* residual,
        float* preconditioned_residual,
        float* direction,
        float* matrix_direction,
        float* products,
        float* scalars,
        std::uint32_t* state,
        void* reduction_scratch,
        std::size_t reduction_scratch_bytes);
} // namespace physica::fluids::liquid::pic::kernels::projection

#endif
