#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_DIFFUSION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_DIFFUSION_KERNELS_H

#include <cstdint>
#include <fluids/gas/device.cuh>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void identity_velocity_vjp(::cuda::stream_ref stream, device::Discretization grid, simulation::VectorView<const double> output_adjoint, simulation::VectorView<double> source_adjoint);
    void diffusion_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t iterations, float viscosity, const std::uint32_t* collider_ids, device::VelocityBoundary boundary, simulation::VectorView<const float> source, simulation::VectorView<float> first, simulation::VectorView<float> second, simulation::VectorView<float> output);
    void diffusion_vjp(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t iterations, float viscosity, const std::uint32_t* collider_ids, device::VelocityBoundary boundary, simulation::VectorView<const double> output_adjoint, simulation::VectorView<double> first, simulation::VectorView<double> second, simulation::VectorView<double> source_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
