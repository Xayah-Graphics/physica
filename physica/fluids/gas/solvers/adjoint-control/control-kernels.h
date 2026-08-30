#ifndef PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_CONTROL_KERNELS_H
#define PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_CONTROL_KERNELS_H

#include <cstdint>
#include <fluids/gas/device.cuh>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::solvers::adjoint_control::kernels {
    struct ControlLatticeData final {
        std::uint32_t x;
        std::uint32_t y;
        std::uint32_t z;
    };

    void control_forward(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, const double* parameters, simulation::VectorView<float> output);
    void control_jvp(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, const double* direction, simulation::VectorView<float> output);
    void control_vjp(::cuda::stream_ref stream, device::Discretization grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, simulation::VectorView<const double> output_adjoint, double* gradient);
} // namespace physica::fluids::gas::solvers::adjoint_control::kernels

#endif
