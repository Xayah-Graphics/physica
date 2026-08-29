#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_KERNELS_H

#include <physica/fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void advect_velocity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity, device::VelocityBoundary boundary, field::VectorView<float> output);
    void advect_velocity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity, field::VectorView<const float> velocity_tangent, device::VelocityBoundary boundary, field::VectorView<float> output_tangent);
    void advect_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity, device::VelocityBoundary boundary, field::VectorView<const double> output_adjoint, field::VectorView<double> velocity_adjoint);

    void advect_scalar_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const float> collider_value, field::ScalarView<const float> source, field::VectorView<const float> velocity, device::ScalarBoundary scalar_boundary, device::VelocityBoundary velocity_boundary, field::ScalarView<float> output);
    void advect_scalar_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const float> source, field::ScalarView<const float> source_tangent, field::VectorView<const float> velocity, field::VectorView<const float> velocity_tangent, device::ScalarBoundary scalar_boundary, device::VelocityBoundary velocity_boundary, field::ScalarView<float> output_tangent);
    void advect_scalar_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const float> source, field::VectorView<const float> velocity, device::ScalarBoundary scalar_boundary, device::VelocityBoundary velocity_boundary, field::ScalarView<const double> output_adjoint, field::ScalarView<double> source_adjoint, field::VectorView<double> velocity_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
