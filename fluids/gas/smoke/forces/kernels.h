#ifndef PHYSICA_FLUIDS_GAS_SMOKE_FORCES_KERNELS_H
#define PHYSICA_FLUIDS_GAS_SMOKE_FORCES_KERNELS_H

#include "../domain/device.h"
#include <cstdint>
#include <cuda/stream>

namespace physica::fluids::gas::smoke::cuda_detail {
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

    struct VorticityTangentScratch final {
        CenteredVectorView centered_velocity;
        CenteredVectorView vorticity;
        ScalarView magnitude;
        CenteredVectorView normal;
    };

    struct VorticityAdjointScratch final {
        CenteredVectorAdjointView centered_velocity;
        CenteredVectorAdjointView vorticity;
        ScalarAdjointView magnitude;
        CenteredVectorAdjointView normal;
    };

    void buoyancy_forward(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView density, ConstScalarView temperature, ConstCenteredVectorView external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, CenteredVectorView force);
    void buoyancy_jvp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView density, ConstScalarView temperature, ConstScalarView density_tangent, ConstScalarView temperature_tangent, ConstCenteredVectorView external_acceleration_tangent, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_temperature_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, CenteredVectorView force_tangent);
    void buoyancy_vjp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView density, ConstScalarView temperature, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, ConstCenteredVectorAdjointView force_adjoint, ScalarAdjointView density_adjoint, ScalarAdjointView temperature_adjoint, CenteredVectorAdjointView external_acceleration_adjoint, double* ambient_temperature_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint);

    void vorticity_forward(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, const float* confinement, VorticityView cache, CenteredVectorView force);
    void vorticity_jvp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity_tangent, const float* confinement, const float* confinement_tangent, ConstVorticityView cache, CenteredVectorView force_tangent, VorticityTangentScratch tangent_scratch);
    void vorticity_vjp(::cuda::stream_ref stream, Grid grid, const std::uint32_t* cell_mask, const float* confinement, ConstVorticityView cache, ConstCenteredVectorAdjointView force_adjoint, StaggeredVectorAdjointView velocity_adjoint, double* confinement_adjoint, VorticityAdjointScratch scratch);
} // namespace physica::fluids::gas::smoke::cuda_detail

#endif
