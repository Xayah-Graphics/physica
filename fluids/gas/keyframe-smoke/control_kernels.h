#ifndef PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_CONTROL_KERNELS_H
#define PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_CONTROL_KERNELS_H

#include "device.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    struct WindData final {
        std::uint32_t begin_step;
        std::uint32_t end_step;
        float width;
        std::uint32_t parameter_offset;
    };

    struct VortexData final {
        std::uint32_t begin_step;
        std::uint32_t end_step;
        float width;
        Vector axis;
        std::uint32_t parameter_offset;
    };

    void control_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t step, const WindData* winds, std::uint32_t wind_count, const VortexData* vortices, std::uint32_t vortex_count, const double* parameters, CenteredVectorView output);
    void control_jvp(::cuda::stream_ref stream, Grid grid, std::uint32_t step, const WindData* winds, std::uint32_t wind_count, const VortexData* vortices, std::uint32_t vortex_count, const double* parameters, const double* direction, CenteredVectorView output_tangent);
    void control_vjp(::cuda::stream_ref stream, Grid grid, std::uint32_t step, const WindData* winds, std::uint32_t wind_count, const VortexData* vortices, std::uint32_t vortex_count, const double* parameters, ConstCenteredVectorAdjointView output_adjoint, double* gradient);
} // namespace physica::fluids::gas::keyframe_smoke::cuda_detail

#endif
