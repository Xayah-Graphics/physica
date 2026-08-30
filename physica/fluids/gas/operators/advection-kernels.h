#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_KERNELS_H

#include <fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void advect_velocity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity, device::VelocityBoundary boundary, simulation::VectorView<float> output);
    void advect_velocity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity, simulation::VectorView<const float> velocity_tangent, device::VelocityBoundary boundary, simulation::VectorView<float> output_tangent);
    void advect_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity, device::VelocityBoundary boundary, simulation::VectorView<const double> output_adjoint, simulation::VectorView<double> velocity_adjoint);

    void advect_scalar_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const float> collider_value, simulation::ScalarView<const float> source, simulation::VectorView<const float> velocity, device::ScalarBoundary scalar_boundary, device::VelocityBoundary velocity_boundary, simulation::ScalarView<float> output);
    void advect_scalar_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const float> source, simulation::ScalarView<const float> source_tangent, simulation::VectorView<const float> velocity, simulation::VectorView<const float> velocity_tangent, device::ScalarBoundary scalar_boundary, device::VelocityBoundary velocity_boundary, simulation::ScalarView<float> output_tangent);
    void advect_scalar_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const float> source, simulation::VectorView<const float> velocity, device::ScalarBoundary scalar_boundary, device::VelocityBoundary velocity_boundary, simulation::ScalarView<const double> output_adjoint, simulation::ScalarView<double> source_adjoint, simulation::VectorView<double> velocity_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
