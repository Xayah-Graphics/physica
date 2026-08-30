#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_FORCE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_FORCE_KERNELS_H

#include <cstdint>
#include <fluids/gas/device.cuh>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void density_buoyancy_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, float buoyancy, simulation::ScalarView<const float> density, simulation::VectorView<float> force);
    void density_buoyancy_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, float buoyancy, simulation::VectorView<const double> force_adjoint, simulation::ScalarView<double> density_adjoint);

    struct VorticityView final {
        simulation::VectorView<float> centered_velocity;
        simulation::VectorView<float> vorticity;
        simulation::ScalarView<float> magnitude;
        simulation::VectorView<float> normal;
        simulation::ScalarView<float> normalizer;
    };

    struct ConstVorticityView final {
        simulation::VectorView<const float> centered_velocity;
        simulation::VectorView<const float> vorticity;
        simulation::ScalarView<const float> magnitude;
        simulation::VectorView<const float> normal;
        simulation::ScalarView<const float> normalizer;
    };

    struct VorticityTangentScratch final {
        simulation::VectorView<float> centered_velocity;
        simulation::VectorView<float> vorticity;
        simulation::ScalarView<float> magnitude;
        simulation::VectorView<float> normal;
    };

    struct VorticityAdjointScratch final {
        simulation::VectorView<double> centered_velocity;
        simulation::VectorView<double> vorticity;
        simulation::ScalarView<double> magnitude;
        simulation::VectorView<double> normal;
    };

    void buoyancy_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const float> density, simulation::ScalarView<const float> temperature, simulation::VectorView<const float> external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, simulation::VectorView<float> force);
    void buoyancy_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const float> density, simulation::ScalarView<const float> temperature, simulation::ScalarView<const float> density_tangent, simulation::ScalarView<const float> temperature_tangent, simulation::VectorView<const float> external_acceleration_tangent, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_temperature_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, simulation::VectorView<float> force_tangent);
    void buoyancy_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::ScalarView<const float> density, simulation::ScalarView<const float> temperature, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, simulation::VectorView<const double> force_adjoint, simulation::ScalarView<double> density_adjoint, simulation::ScalarView<double> temperature_adjoint, simulation::VectorView<double> external_acceleration_adjoint, double* ambient_temperature_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint);

    void vorticity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity, const float* confinement, VorticityView cache, simulation::VectorView<float> force);
    void vorticity_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity, float confinement, VorticityView cache, simulation::VectorView<float> force);
    void vorticity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity_tangent, const float* confinement, const float* confinement_tangent, ConstVorticityView cache, simulation::VectorView<float> force_tangent, VorticityTangentScratch tangent_scratch);
    void vorticity_jvp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> velocity_tangent, float confinement, ConstVorticityView cache, simulation::VectorView<float> force_tangent, VorticityTangentScratch tangent_scratch);
    void vorticity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, const float* confinement, ConstVorticityView cache, simulation::VectorView<const double> force_adjoint, simulation::VectorView<double> velocity_adjoint, double* confinement_adjoint, VorticityAdjointScratch scratch);
    void vorticity_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, float confinement, ConstVorticityView cache, simulation::VectorView<const double> force_adjoint, simulation::VectorView<double> velocity_adjoint, VorticityAdjointScratch scratch);

    void combine_forward(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const float> physical, simulation::VectorView<const float> control, simulation::VectorView<float> total);
    void combine_vjp(::cuda::stream_ref stream, device::Discretization grid, const std::uint32_t* collider_ids, simulation::VectorView<const double> total_adjoint, simulation::VectorView<double> physical_adjoint, simulation::VectorView<double> control_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
