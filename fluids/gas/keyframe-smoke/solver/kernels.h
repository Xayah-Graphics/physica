#ifndef PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_SOLVER_KERNELS_H
#define PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_SOLVER_KERNELS_H

#include "../domain/device.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    struct VorticityView final {
        CenteredVectorView centered_velocity;
        CenteredVectorView vorticity;
        ScalarView magnitude;
        CenteredVectorView normal;
        ScalarView normalizer;
    };

    struct ConstVorticityView final {
        ConstCenteredVectorView centered_velocity;
        ConstCenteredVectorView vorticity;
        ConstScalarView magnitude;
        ConstCenteredVectorView normal;
        ConstScalarView normalizer;
    };

    struct VorticityTangentView final {
        CenteredVectorView centered_velocity;
        CenteredVectorView vorticity;
        ScalarView magnitude;
        CenteredVectorView normal;
    };

    struct VorticityAdjointView final {
        CenteredVectorAdjointView centered_velocity;
        CenteredVectorAdjointView vorticity;
        ScalarAdjointView magnitude;
        CenteredVectorAdjointView normal;
    };

    void force_forward(::cuda::stream_ref stream, Grid grid, float density_buoyancy, float vorticity_confinement, ConstScalarView density, ConstStaggeredVectorView velocity, ConstCenteredVectorView control, VorticityView vorticity, CenteredVectorView physical_force, CenteredVectorView control_force, CenteredVectorView total_force);
    void force_jvp(::cuda::stream_ref stream, Grid grid, float density_buoyancy, float vorticity_confinement, ConstScalarView density_tangent, ConstStaggeredVectorView velocity_tangent, ConstCenteredVectorView control_tangent, ConstVorticityView cache, VorticityTangentView vorticity_tangent, CenteredVectorView physical_force_tangent, CenteredVectorView total_force_tangent);
    void force_vjp(::cuda::stream_ref stream, Grid grid, float density_buoyancy, float vorticity_confinement, ConstVorticityView cache, ConstCenteredVectorAdjointView total_force_adjoint, ScalarAdjointView density_adjoint, StaggeredVectorAdjointView velocity_adjoint, CenteredVectorAdjointView physical_force_adjoint, CenteredVectorAdjointView control_adjoint, VorticityAdjointView vorticity_adjoint);

    void integrate_velocity_forward(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorView velocity, ConstCenteredVectorView force, StaggeredVectorView output);
    void integrate_velocity_vjp(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView velocity_adjoint, CenteredVectorAdjointView force_adjoint);
    void advect_velocity_forward(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, StaggeredVectorView output);
    void advect_velocity_jvp(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorView velocity, ConstStaggeredVectorView velocity_tangent, VelocityBoundaryData boundary, StaggeredVectorView output_tangent);
    void advect_velocity_vjp(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView velocity_adjoint);
    void constrain_velocity_forward(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, StaggeredVectorView output);
    void constrain_velocity_vjp(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorAdjointView output_adjoint, VelocityBoundaryData boundary, StaggeredVectorAdjointView velocity_adjoint);

    void diffusion_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t iterations, float viscosity, VelocityBoundaryData boundary, ConstStaggeredVectorView source, StaggeredVectorView first, StaggeredVectorView second, StaggeredVectorView output);
    void diffusion_vjp(::cuda::stream_ref stream, Grid grid, std::uint32_t iterations, float viscosity, VelocityBoundaryData boundary, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView first, StaggeredVectorAdjointView second, StaggeredVectorAdjointView source_adjoint);

    void pressure_rhs_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t pressure_anchor, ConstStaggeredVectorView velocity, ScalarView rhs);
    void pressure_rhs_vjp(::cuda::stream_ref stream, Grid grid, std::uint32_t pressure_anchor, ConstScalarAdjointView rhs_adjoint, StaggeredVectorAdjointView velocity_adjoint);
    void pressure_forward(::cuda::stream_ref stream, Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, ScalarBoundaryData boundary, ConstScalarView rhs, ScalarView pressure);
    void pressure_vjp(::cuda::stream_ref stream, Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, ScalarBoundaryData boundary, ScalarAdjointView pressure_adjoint, ScalarAdjointView rhs_adjoint);
    void project_velocity_forward(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorView velocity, ConstScalarView pressure, StaggeredVectorView output);
    void project_velocity_vjp(::cuda::stream_ref stream, Grid grid, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView velocity_adjoint, ScalarAdjointView pressure_adjoint);

    void advect_scalar_forward(::cuda::stream_ref stream, Grid grid, ConstScalarView source, ConstStaggeredVectorView velocity, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ScalarView output);
    void advect_scalar_jvp(::cuda::stream_ref stream, Grid grid, ConstScalarView source, ConstScalarView source_tangent, ConstStaggeredVectorView velocity, ConstStaggeredVectorView velocity_tangent, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ScalarView output_tangent);
    void advect_scalar_vjp(::cuda::stream_ref stream, Grid grid, ConstScalarView source, ConstStaggeredVectorView velocity, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ConstScalarAdjointView output_adjoint, ScalarAdjointView source_adjoint, StaggeredVectorAdjointView velocity_adjoint);

    void mass_forward(::cuda::stream_ref stream, Grid grid, float retention, ConstScalarView input, ConstScalarView advected, double* input_mass, double* advected_mass, ScalarView output);
    void mass_jvp(::cuda::stream_ref stream, Grid grid, float retention, ConstScalarView input, ConstScalarView advected, ConstScalarView input_tangent, ConstScalarView advected_tangent, const double* input_mass, const double* advected_mass, double* input_mass_tangent, double* advected_mass_tangent, ScalarView output_tangent);
    void mass_vjp(::cuda::stream_ref stream, Grid grid, float retention, ConstScalarView advected, const double* input_mass, const double* advected_mass, ConstScalarAdjointView output_adjoint, double* density_dot, ScalarAdjointView input_adjoint, ScalarAdjointView advected_adjoint);
} // namespace physica::fluids::gas::keyframe_smoke::cuda_detail

#endif
