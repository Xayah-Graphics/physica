#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_OBJECTIVE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_OBJECTIVE_KERNELS_H

#include <fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void residual_forward(::cuda::stream_ref stream, device::Discretization grid, simulation::ScalarView<const float> state, simulation::ScalarView<const float> target, simulation::ScalarView<float> residual);
    void residual_forward(::cuda::stream_ref stream, device::Discretization grid, simulation::VectorView<const float> state, simulation::VectorView<const float> target, simulation::VectorView<float> residual);
    void blur_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t radius, const float* weights, simulation::ScalarView<const float> source, simulation::ScalarView<float> first, simulation::ScalarView<float> second, simulation::ScalarView<float> output);
    void blur_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t radius, const float* weights, simulation::VectorView<const float> source, simulation::VectorView<float> first, simulation::VectorView<float> second, simulation::VectorView<float> output);
    void squared_loss(::cuda::stream_ref stream, device::Discretization grid, double weight, simulation::ScalarView<const float> field, double* output);
    void squared_loss(::cuda::stream_ref stream, device::Discretization grid, double weight, simulation::VectorView<const float> field, double* output);
    void directional_loss(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t radius, const float* weights, double density_weight, double velocity_weight, simulation::ScalarView<const float> density_residual, simulation::VectorView<const float> velocity_residual, simulation::ScalarView<const float> density_tangent, simulation::VectorView<const float> velocity_tangent, simulation::ScalarView<float> scalar_first, simulation::ScalarView<float> scalar_second, simulation::ScalarView<float> scalar_output, simulation::VectorView<float> velocity_first, simulation::VectorView<float> velocity_second, simulation::VectorView<float> velocity_output, double* result);
    void inject_keyframe_adjoint(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t radius, const float* weights, double density_weight, double velocity_weight, simulation::ScalarView<const float> blurred_density_residual, simulation::VectorView<const float> blurred_velocity_residual, simulation::ScalarView<float> scalar_first, simulation::ScalarView<float> scalar_second, simulation::VectorView<float> velocity_first, simulation::VectorView<float> velocity_second, simulation::ScalarView<double> density_adjoint, simulation::VectorView<double> velocity_adjoint);
    void control_effort(::cuda::stream_ref stream, device::Discretization grid, double weight, simulation::VectorView<const float> control, double* result);
    void accumulate(::cuda::stream_ref stream, const double* contribution, double* objective);
    void control_effort_jvp(::cuda::stream_ref stream, device::Discretization grid, double weight, simulation::VectorView<const float> control, simulation::VectorView<const float> control_tangent, double* result);
    void control_effort_vjp(::cuda::stream_ref stream, device::Discretization grid, double weight, simulation::VectorView<const float> control, simulation::VectorView<double> control_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
