#ifndef PHYSICA_FLUIDS_GAS_SMOKE_TRANSPORT_KERNELS_H
#define PHYSICA_FLUIDS_GAS_SMOKE_TRANSPORT_KERNELS_H

#include "../domain/device.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::smoke::cuda_detail {
    void source_forward(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView state, ConstScalarView source, ScalarView output);
    void source_jvp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView state_tangent, ConstScalarView source_tangent, ScalarView output_tangent);
    void source_vjp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarAdjointView output_adjoint, ScalarAdjointView state_adjoint, ScalarAdjointView source_adjoint);

    void integrate_velocity_forward(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, ConstCenteredVectorView force, StaggeredVectorView output);
    void integrate_velocity_jvp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity_tangent, ConstCenteredVectorView force_tangent, StaggeredVectorView output_tangent);
    void integrate_velocity_vjp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView velocity_adjoint, CenteredVectorAdjointView force_adjoint);

    void advect_velocity_forward(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, StaggeredVectorView output);
    void advect_velocity_jvp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, ConstStaggeredVectorView velocity_tangent, VelocityBoundaryData boundary, StaggeredVectorView output_tangent);
    void advect_velocity_vjp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView velocity_adjoint);

    void constrain_velocity_forward(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView collider_velocity, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, StaggeredVectorView output);
    void constrain_velocity_jvp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity_tangent, VelocityBoundaryData boundary, StaggeredVectorView output_tangent);
    void constrain_velocity_vjp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorAdjointView output_adjoint, VelocityBoundaryData boundary, StaggeredVectorAdjointView velocity_adjoint);

    void advect_scalar_forward(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView collider_value, ConstScalarView source, ConstStaggeredVectorView velocity, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ScalarView output);
    void advect_scalar_jvp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView source, ConstScalarView source_tangent, ConstStaggeredVectorView velocity, ConstStaggeredVectorView velocity_tangent, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ScalarView output_tangent);
    void advect_scalar_vjp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView source, ConstStaggeredVectorView velocity, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ConstScalarAdjointView output_adjoint, ScalarAdjointView source_adjoint, StaggeredVectorAdjointView velocity_adjoint);
} // namespace physica::fluids::gas::smoke::cuda_detail

#endif
