#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_DIFFUSION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_DIFFUSION_KERNELS_H

#include <physica/fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void identity_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, field::VectorView<const double> output_adjoint, field::VectorView<double> source_adjoint);
    void diffusion_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t iterations, float viscosity, const std::uint32_t* collider_ids, device::VelocityBoundary boundary, field::VectorView<const float> source, field::VectorView<float> first, field::VectorView<float> second, field::VectorView<float> output);
    void diffusion_vjp(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t iterations, float viscosity, const std::uint32_t* collider_ids, device::VelocityBoundary boundary, field::VectorView<const double> output_adjoint, field::VectorView<double> first, field::VectorView<double> second, field::VectorView<double> source_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
