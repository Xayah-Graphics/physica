#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_FORCE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_FORCE_KERNELS_H

#include <physica/fluids/gas/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void density_buoyancy_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, float buoyancy, field::ScalarView<const float> density, field::VectorView<float> force);
    void density_buoyancy_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, float buoyancy, field::VectorView<const double> force_adjoint, field::ScalarView<double> density_adjoint);

    struct VorticityView final {
        field::VectorView<float> centered_velocity;
        field::VectorView<float> vorticity;
        field::ScalarView<float> magnitude;
        field::VectorView<float> normal;
        field::ScalarView<float> normalizer;
    };

    struct ConstVorticityView final {
        field::VectorView<const float> centered_velocity;
        field::VectorView<const float> vorticity;
        field::ScalarView<const float> magnitude;
        field::VectorView<const float> normal;
        field::ScalarView<const float> normalizer;
    };

    struct VorticityTangentScratch final {
        field::VectorView<float> centered_velocity;
        field::VectorView<float> vorticity;
        field::ScalarView<float> magnitude;
        field::VectorView<float> normal;
    };

    struct VorticityAdjointScratch final {
        field::VectorView<double> centered_velocity;
        field::VectorView<double> vorticity;
        field::ScalarView<double> magnitude;
        field::VectorView<double> normal;
    };

    void buoyancy_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const float> density, field::ScalarView<const float> temperature, field::VectorView<const float> external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, field::VectorView<float> force);
    void buoyancy_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const float> density, field::ScalarView<const float> temperature, field::ScalarView<const float> density_tangent, field::ScalarView<const float> temperature_tangent, field::VectorView<const float> external_acceleration_tangent, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_temperature_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, field::VectorView<float> force_tangent);
    void buoyancy_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::ScalarView<const float> density, field::ScalarView<const float> temperature, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, field::VectorView<const double> force_adjoint, field::ScalarView<double> density_adjoint, field::ScalarView<double> temperature_adjoint, field::VectorView<double> external_acceleration_adjoint, double* ambient_temperature_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint);

    void vorticity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity, const float* confinement, VorticityView cache, field::VectorView<float> force);
    void vorticity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity, float confinement, VorticityView cache, field::VectorView<float> force);
    void vorticity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity_tangent, const float* confinement, const float* confinement_tangent, ConstVorticityView cache, field::VectorView<float> force_tangent, VorticityTangentScratch tangent_scratch);
    void vorticity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> velocity_tangent, float confinement, ConstVorticityView cache, field::VectorView<float> force_tangent, VorticityTangentScratch tangent_scratch);
    void vorticity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, const float* confinement, ConstVorticityView cache, field::VectorView<const double> force_adjoint, field::VectorView<double> velocity_adjoint, double* confinement_adjoint, VorticityAdjointScratch scratch);
    void vorticity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, float confinement, ConstVorticityView cache, field::VectorView<const double> force_adjoint, field::VectorView<double> velocity_adjoint, VorticityAdjointScratch scratch);

    void combine_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const float> physical, field::VectorView<const float> control, field::VectorView<float> total);
    void combine_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, field::VectorView<const double> total_adjoint, field::VectorView<double> physical_adjoint, field::VectorView<double> control_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
