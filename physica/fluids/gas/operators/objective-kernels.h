#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_OBJECTIVE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_OBJECTIVE_KERNELS_H

#include <physica/fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void residual_forward(::cuda::stream_ref stream, device::Discretization grid, field::ScalarView<const float> state, field::ScalarView<const float> target, field::ScalarView<float> residual);
    void residual_forward(::cuda::stream_ref stream, device::Discretization grid, field::VectorView<const float> state, field::VectorView<const float> target, field::VectorView<float> residual);
    void blur_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t radius, const float* weights, field::ScalarView<const float> source, field::ScalarView<float> first, field::ScalarView<float> second, field::ScalarView<float> output);
    void blur_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t radius, const float* weights, field::VectorView<const float> source, field::VectorView<float> first, field::VectorView<float> second, field::VectorView<float> output);
    void squared_loss(::cuda::stream_ref stream, device::Discretization grid, double weight, field::ScalarView<const float> field, double* output);
    void squared_loss(::cuda::stream_ref stream, device::Discretization grid, double weight, field::VectorView<const float> field, double* output);
    void directional_loss(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t radius, const float* weights, double density_weight, double velocity_weight, field::ScalarView<const float> density_residual, field::VectorView<const float> velocity_residual, field::ScalarView<const float> density_tangent, field::VectorView<const float> velocity_tangent, field::ScalarView<float> scalar_first, field::ScalarView<float> scalar_second, field::ScalarView<float> scalar_output, field::VectorView<float> velocity_first, field::VectorView<float> velocity_second, field::VectorView<float> velocity_output, double* result);
    void inject_keyframe_adjoint(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t radius, const float* weights, double density_weight, double velocity_weight, field::ScalarView<const float> blurred_density_residual, field::VectorView<const float> blurred_velocity_residual, field::ScalarView<float> scalar_first, field::ScalarView<float> scalar_second, field::VectorView<float> velocity_first, field::VectorView<float> velocity_second, field::ScalarView<double> density_adjoint, field::VectorView<double> velocity_adjoint);
    void control_effort(::cuda::stream_ref stream, device::Discretization grid, double weight, field::VectorView<const float> control, double* result);
    void accumulate(::cuda::stream_ref stream, const double* contribution, double* objective);
    void control_effort_jvp(::cuda::stream_ref stream, device::Discretization grid, double weight, field::VectorView<const float> control, field::VectorView<const float> control_tangent, double* result);
    void control_effort_vjp(::cuda::stream_ref stream, device::Discretization grid, double weight, field::VectorView<const float> control, field::VectorView<double> control_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
