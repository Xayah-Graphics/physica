#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_FORCE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_FORCE_KERNELS_H

#include "../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::cuda_backend {
    void density_buoyancy_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, float buoyancy, detail::cuda::ScalarView<const float> density, detail::cuda::CenteredVectorView<float> force);
    void density_buoyancy_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, float buoyancy, detail::cuda::CenteredVectorView<const double> force_adjoint, detail::cuda::ScalarView<double> density_adjoint);

    struct VorticityView final {
        detail::cuda::CenteredVectorView<float> centered_velocity;
        detail::cuda::CenteredVectorView<float> vorticity;
        detail::cuda::ScalarView<float> magnitude;
        detail::cuda::CenteredVectorView<float> normal;
        detail::cuda::ScalarView<float> normalizer;
    };

    struct ConstVorticityView final {
        detail::cuda::CenteredVectorView<const float> centered_velocity;
        detail::cuda::CenteredVectorView<const float> vorticity;
        detail::cuda::ScalarView<const float> magnitude;
        detail::cuda::CenteredVectorView<const float> normal;
        detail::cuda::ScalarView<const float> normalizer;
    };

    struct VorticityTangentScratch final {
        detail::cuda::CenteredVectorView<float> centered_velocity;
        detail::cuda::CenteredVectorView<float> vorticity;
        detail::cuda::ScalarView<float> magnitude;
        detail::cuda::CenteredVectorView<float> normal;
    };

    struct VorticityAdjointScratch final {
        detail::cuda::CenteredVectorView<double> centered_velocity;
        detail::cuda::CenteredVectorView<double> vorticity;
        detail::cuda::ScalarView<double> magnitude;
        detail::cuda::CenteredVectorView<double> normal;
    };

    void buoyancy_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const float> density, detail::cuda::ScalarView<const float> temperature, detail::cuda::CenteredVectorView<const float> external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, detail::cuda::CenteredVectorView<float> force);
    void buoyancy_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const float> density, detail::cuda::ScalarView<const float> temperature, detail::cuda::ScalarView<const float> density_tangent, detail::cuda::ScalarView<const float> temperature_tangent, detail::cuda::CenteredVectorView<const float> external_acceleration_tangent, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_temperature_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, detail::cuda::CenteredVectorView<float> force_tangent);
    void buoyancy_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const float> density, detail::cuda::ScalarView<const float> temperature, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, detail::cuda::CenteredVectorView<const double> force_adjoint, detail::cuda::ScalarView<double> density_adjoint, detail::cuda::ScalarView<double> temperature_adjoint, detail::cuda::CenteredVectorView<double> external_acceleration_adjoint, double* ambient_temperature_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint);

    void vorticity_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity, const float* confinement, VorticityView cache, detail::cuda::CenteredVectorView<float> force);
    void vorticity_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity, float confinement, VorticityView cache, detail::cuda::CenteredVectorView<float> force);
    void vorticity_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity_tangent, const float* confinement, const float* confinement_tangent, ConstVorticityView cache, detail::cuda::CenteredVectorView<float> force_tangent, VorticityTangentScratch tangent_scratch);
    void vorticity_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity_tangent, float confinement, ConstVorticityView cache, detail::cuda::CenteredVectorView<float> force_tangent, VorticityTangentScratch tangent_scratch);
    void vorticity_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, const float* confinement, ConstVorticityView cache, detail::cuda::CenteredVectorView<const double> force_adjoint, detail::cuda::StaggeredVectorView<double> velocity_adjoint, double* confinement_adjoint, VorticityAdjointScratch scratch);
    void vorticity_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, float confinement, ConstVorticityView cache, detail::cuda::CenteredVectorView<const double> force_adjoint, detail::cuda::StaggeredVectorView<double> velocity_adjoint, VorticityAdjointScratch scratch);

    void combine_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::CenteredVectorView<const float> physical, detail::cuda::CenteredVectorView<const float> control, detail::cuda::CenteredVectorView<float> total);
    void combine_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::CenteredVectorView<const double> total_adjoint, detail::cuda::CenteredVectorView<double> physical_adjoint, detail::cuda::CenteredVectorView<double> control_adjoint);
} // namespace physica::fluids::gas::operators::cuda_backend

#endif
