#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_POINTWISE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_POINTWISE_KERNELS_H

#include <fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void source_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const float> state, simulation::ScalarView<const float> source, simulation::ScalarView<float> output);
    void source_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const float> state_tangent, simulation::ScalarView<const float> source_tangent, simulation::ScalarView<float> output_tangent);
    void source_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const double> output_adjoint, simulation::ScalarView<double> state_adjoint, simulation::ScalarView<double> source_adjoint);

    void integrate_velocity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity, simulation::VectorView<const float> force, simulation::VectorView<float> output);
    void integrate_velocity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity_tangent, simulation::VectorView<const float> force_tangent, simulation::VectorView<float> output_tangent);
    void integrate_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const double> output_adjoint, simulation::VectorView<double> velocity_adjoint, simulation::VectorView<double> force_adjoint);

    void constrain_velocity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> collider_velocity, simulation::VectorView<const float> velocity, device::VelocityBoundary boundary, simulation::VectorView<float> output);
    void constrain_velocity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity_tangent, device::VelocityBoundary boundary, simulation::VectorView<float> output_tangent);
    void constrain_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const double> output_adjoint, device::VelocityBoundary boundary, simulation::VectorView<double> velocity_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
