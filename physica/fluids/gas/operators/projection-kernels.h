#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_PROJECTION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_PROJECTION_KERNELS_H

#include <cstdint>
#include <fluids/gas/device.cuh>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void pressure_rhs_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity, simulation::ScalarView<float> rhs);
    void pressure_rhs_vjp(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, simulation::ScalarView<const double> rhs_adjoint, simulation::VectorView<double> velocity_adjoint);
    void project_velocity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity, simulation::ScalarView<const float> pressure, simulation::VectorView<float> output);
    void project_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const double> output_adjoint, simulation::VectorView<double> velocity_adjoint, simulation::ScalarView<double> pressure_adjoint);
    void red_black_gauss_seidel_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, device::ScalarBoundary boundary, simulation::ScalarView<const float> rhs, simulation::ScalarView<float> pressure);
    void red_black_gauss_seidel_vjp(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, device::ScalarBoundary boundary, simulation::ScalarView<double> pressure_adjoint, simulation::ScalarView<double> rhs_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
