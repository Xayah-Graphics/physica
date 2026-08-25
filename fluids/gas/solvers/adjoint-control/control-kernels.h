#ifndef PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_CONTROL_KERNELS_H
#define PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_CONTROL_KERNELS_H

#include "../../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::adjoint_control::cuda_backend {
    struct ControlLatticeData final {
        std::uint32_t x;
        std::uint32_t y;
        std::uint32_t z;
    };

    void control_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, const double* parameters, detail::cuda::CenteredVectorView<float> output);
    void control_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, const double* direction, detail::cuda::CenteredVectorView<float> output);
    void control_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t step, ControlLatticeData lattice, float sigma, std::uint32_t step_count, detail::cuda::CenteredVectorView<const double> output_adjoint, double* gradient);
} // namespace physica::fluids::gas::adjoint_control::cuda_backend

#endif
