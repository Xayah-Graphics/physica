#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_PROJECTION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_PROJECTION_KERNELS_H

#include <physica/fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void pressure_rhs_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, field::VectorView<const float> velocity, field::ScalarView<float> rhs);
    void pressure_rhs_vjp(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, field::ScalarView<const double> rhs_adjoint, field::VectorView<double> velocity_adjoint);
    void project_velocity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity, field::ScalarView<const float> pressure, field::VectorView<float> output);
    void project_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const double> output_adjoint, field::VectorView<double> velocity_adjoint, field::ScalarView<double> pressure_adjoint);
    void red_black_gauss_seidel_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, device::ScalarBoundary boundary, field::ScalarView<const float> rhs, field::ScalarView<float> pressure);
    void red_black_gauss_seidel_vjp(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, device::ScalarBoundary boundary, field::ScalarView<double> pressure_adjoint, field::ScalarView<double> rhs_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
