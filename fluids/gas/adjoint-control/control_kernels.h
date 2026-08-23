#ifndef PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_CONTROL_KERNELS_H
#define PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_CONTROL_KERNELS_H

#include "device.h"
#include <cstdint>
#include <cuda/stream>

namespace physica::fluids::gas::adjoint_control::cuda_detail {
    struct ControlLatticeData final {
        std::uint32_t x;
        std::uint32_t y;
        std::uint32_t z;
    };

    void control_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, const double* parameters, CenteredVectorView output);
    void control_jvp(::cuda::stream_ref stream, Grid grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, const double* direction, CenteredVectorView output);
    void control_vjp(::cuda::stream_ref stream, Grid grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, ConstCenteredVectorAdjointView output_adjoint, double* gradient);
} // namespace physica::fluids::gas::adjoint_control::cuda_detail

#endif
