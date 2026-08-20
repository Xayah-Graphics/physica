#ifndef PHYSICA_FLUIDS_GAS_SMOKE_PROJECTION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_SMOKE_PROJECTION_KERNELS_H

#include "../domain/device.h"
#include <cstdint>
#include <cuda/stream>

namespace physica::fluids::gas::smoke::cuda_detail {
    void pressure_rhs_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, ScalarView rhs);
    void pressure_rhs_vjp(::cuda::stream_ref stream, Grid grid, std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, ConstScalarAdjointView rhs_adjoint, StaggeredVectorAdjointView velocity_adjoint);
    void project_velocity_forward(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, ConstScalarView pressure, StaggeredVectorView output);
    void project_velocity_vjp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView velocity_adjoint, ScalarAdjointView pressure_adjoint);
    void red_black_gauss_seidel_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, ScalarBoundaryData boundary, ConstScalarView rhs, ScalarView pressure);
    void red_black_gauss_seidel_vjp(::cuda::stream_ref stream, Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, ScalarBoundaryData boundary, ScalarAdjointView pressure_adjoint, ScalarAdjointView rhs_adjoint);
} // namespace physica::fluids::gas::smoke::cuda_detail

#endif
