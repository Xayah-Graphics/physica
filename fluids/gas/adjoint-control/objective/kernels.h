#ifndef PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_OBJECTIVE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_OBJECTIVE_KERNELS_H

#include "../domain/device.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::adjoint_control::cuda_detail {
    void residual_forward(::cuda::stream_ref stream, Grid grid, ConstScalarView state, ConstScalarView target, ScalarView residual);
    void residual_forward(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorView state, ConstStaggeredVectorView target, StaggeredVectorView residual);
    void blur_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t radius, const float* weights, ConstScalarView source, ScalarView first, ScalarView second, ScalarView output);
    void blur_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t radius, const float* weights, ConstStaggeredVectorView source, StaggeredVectorView first, StaggeredVectorView second, StaggeredVectorView output);
    void squared_loss(::cuda::stream_ref stream, Grid grid, double weight, ConstScalarView field, double* output);
    void squared_loss(::cuda::stream_ref stream, Grid grid, double weight, ConstStaggeredVectorView field, double* output);
    void directional_loss(::cuda::stream_ref stream, Grid grid, std::uint32_t radius, const float* weights, double density_weight, double velocity_weight, ConstScalarView density_residual, ConstStaggeredVectorView velocity_residual, ConstScalarView density_tangent, ConstStaggeredVectorView velocity_tangent, ScalarView scalar_first, ScalarView scalar_second, ScalarView scalar_output, StaggeredVectorView velocity_first, StaggeredVectorView velocity_second, StaggeredVectorView velocity_output, double* result);
    void inject_keyframe_adjoint(::cuda::stream_ref stream, Grid grid, std::uint32_t radius, const float* weights, double density_weight, double velocity_weight, ConstScalarView blurred_density_residual, ConstStaggeredVectorView blurred_velocity_residual, ScalarView scalar_first, ScalarView scalar_second, StaggeredVectorView velocity_first, StaggeredVectorView velocity_second, ScalarAdjointView density_adjoint, StaggeredVectorAdjointView velocity_adjoint);
    void control_effort(::cuda::stream_ref stream, Grid grid, double weight, ConstCenteredVectorView control, double* result);
    void control_effort_jvp(::cuda::stream_ref stream, Grid grid, double weight, ConstCenteredVectorView control, ConstCenteredVectorView control_tangent, double* result);
    void control_effort_vjp(::cuda::stream_ref stream, Grid grid, double weight, ConstCenteredVectorView control, CenteredVectorAdjointView control_adjoint);
} // namespace physica::fluids::gas::adjoint_control::cuda_detail

#endif
