#ifndef PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_CONTROL_KERNELS_H
#define PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_CONTROL_KERNELS_H

#include "../../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::keyframe_smoke::cuda_backend {
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
        detail::cuda::Vector axis;
        std::uint32_t parameter_offset;
    };

    void control_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t step, const WindData* winds, std::uint32_t wind_count, const VortexData* vortices, std::uint32_t vortex_count, const double* parameters, detail::cuda::CenteredVectorView<float> output);
    void control_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t step, const WindData* winds, std::uint32_t wind_count, const VortexData* vortices, std::uint32_t vortex_count, const double* parameters, const double* direction, detail::cuda::CenteredVectorView<float> output_tangent);
    void control_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t step, const WindData* winds, std::uint32_t wind_count, const VortexData* vortices, std::uint32_t vortex_count, const double* parameters, detail::cuda::CenteredVectorView<const double> output_adjoint, double* gradient);
} // namespace physica::fluids::gas::keyframe_smoke::cuda_backend

#endif
