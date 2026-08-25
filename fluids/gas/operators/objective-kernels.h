#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_OBJECTIVE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_OBJECTIVE_KERNELS_H

#include "../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::cuda_backend {
    void residual_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, detail::cuda::ScalarView<const float> state, detail::cuda::ScalarView<const float> target, detail::cuda::ScalarView<float> residual);
    void residual_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, detail::cuda::StaggeredVectorView<const float> state, detail::cuda::StaggeredVectorView<const float> target, detail::cuda::StaggeredVectorView<float> residual);
    void blur_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t radius, const float* weights, detail::cuda::ScalarView<const float> source, detail::cuda::ScalarView<float> first, detail::cuda::ScalarView<float> second, detail::cuda::ScalarView<float> output);
    void blur_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t radius, const float* weights, detail::cuda::StaggeredVectorView<const float> source, detail::cuda::StaggeredVectorView<float> first, detail::cuda::StaggeredVectorView<float> second, detail::cuda::StaggeredVectorView<float> output);
    void squared_loss(::cuda::stream_ref stream, detail::cuda::Grid grid, double weight, detail::cuda::ScalarView<const float> field, double* output);
    void squared_loss(::cuda::stream_ref stream, detail::cuda::Grid grid, double weight, detail::cuda::StaggeredVectorView<const float> field, double* output);
    void directional_loss(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t radius, const float* weights, double density_weight, double velocity_weight, detail::cuda::ScalarView<const float> density_residual, detail::cuda::StaggeredVectorView<const float> velocity_residual, detail::cuda::ScalarView<const float> density_tangent, detail::cuda::StaggeredVectorView<const float> velocity_tangent, detail::cuda::ScalarView<float> scalar_first, detail::cuda::ScalarView<float> scalar_second, detail::cuda::ScalarView<float> scalar_output, detail::cuda::StaggeredVectorView<float> velocity_first, detail::cuda::StaggeredVectorView<float> velocity_second, detail::cuda::StaggeredVectorView<float> velocity_output, double* result);
    void inject_keyframe_adjoint(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t radius, const float* weights, double density_weight, double velocity_weight, detail::cuda::ScalarView<const float> blurred_density_residual, detail::cuda::StaggeredVectorView<const float> blurred_velocity_residual, detail::cuda::ScalarView<float> scalar_first, detail::cuda::ScalarView<float> scalar_second, detail::cuda::StaggeredVectorView<float> velocity_first, detail::cuda::StaggeredVectorView<float> velocity_second, detail::cuda::ScalarView<double> density_adjoint, detail::cuda::StaggeredVectorView<double> velocity_adjoint);
    void control_effort(::cuda::stream_ref stream, detail::cuda::Grid grid, double weight, detail::cuda::CenteredVectorView<const float> control, double* result);
    void control_effort_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, double weight, detail::cuda::CenteredVectorView<const float> control, detail::cuda::CenteredVectorView<const float> control_tangent, double* result);
    void control_effort_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, double weight, detail::cuda::CenteredVectorView<const float> control, detail::cuda::CenteredVectorView<double> control_adjoint);
} // namespace physica::fluids::gas::operators::cuda_backend

#endif
