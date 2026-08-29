#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_POINTWISE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_POINTWISE_KERNELS_H

#include <physica/fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void source_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const float> state, field::ScalarView<const float> source, field::ScalarView<float> output);
    void source_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const float> state_tangent, field::ScalarView<const float> source_tangent, field::ScalarView<float> output_tangent);
    void source_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const double> output_adjoint, field::ScalarView<double> state_adjoint, field::ScalarView<double> source_adjoint);

    void integrate_velocity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity, field::VectorView<const float> force, field::VectorView<float> output);
    void integrate_velocity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity_tangent, field::VectorView<const float> force_tangent, field::VectorView<float> output_tangent);
    void integrate_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const double> output_adjoint, field::VectorView<double> velocity_adjoint, field::VectorView<double> force_adjoint);

    void constrain_velocity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> collider_velocity, field::VectorView<const float> velocity, device::VelocityBoundary boundary, field::VectorView<float> output);
    void constrain_velocity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity_tangent, device::VelocityBoundary boundary, field::VectorView<float> output_tangent);
    void constrain_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const double> output_adjoint, device::VelocityBoundary boundary, field::VectorView<double> velocity_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
