module;

#include <physica/fluids/gas/interop.h>
#include "force-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.force;

import std;

namespace physica::fluids::gas::operators {
    DensityBuoyancy::DensityBuoyancy(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    DensityBuoyancy::Cache DensityBuoyancy::allocate_cache(const Domain& domain) const {
        return {.force = domain.grid.allocate_cell_vector_field<float>()};
    }

    DensityBuoyancy::TangentWorkspace DensityBuoyancy::allocate_tangent_workspace(const Domain& domain) const {
        return {.force = domain.grid.allocate_cell_vector_field<float>()};
    }

    DensityBuoyancy::AdjointWorkspace DensityBuoyancy::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void DensityBuoyancy::forward(const Domain& domain, const ScalarField<float>& density, Cache& cache) const {
        kernels::density_buoyancy_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, field::scalar_view(density), field::view(cache.force));
    }

    void DensityBuoyancy::jvp(const Domain& domain, const ScalarField<float>& density_tangent, const Cache&, TangentWorkspace& workspace) const {
        kernels::density_buoyancy_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, field::scalar_view(density_tangent), field::view(workspace.force));
    }

    void DensityBuoyancy::vjp(const Domain& domain, const Cache&, const VectorField<double>& force_adjoint, ScalarField<double>& density_adjoint, AdjointWorkspace&) const {
        kernels::density_buoyancy_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, field::view(force_adjoint), field::scalar_view(density_adjoint));
    }

    namespace {
        kernels::VorticityView view(VorticityConfinement::Cache& cache) {
            return {.centered_velocity = field::view(cache.centered_velocity), .vorticity = field::view(cache.vorticity), .magnitude = field::scalar_view(cache.magnitude), .normal = field::view(cache.normal), .normalizer = field::scalar_view(cache.normalizer)};
        }

        kernels::ConstVorticityView view(const VorticityConfinement::Cache& cache) {
            return {.centered_velocity = field::view(cache.centered_velocity), .vorticity = field::view(cache.vorticity), .magnitude = field::scalar_view(cache.magnitude), .normal = field::view(cache.normal), .normalizer = field::scalar_view(cache.normalizer)};
        }

        kernels::VorticityTangentScratch view(VorticityConfinement::TangentWorkspace& workspace) {
            return {.centered_velocity = field::view(workspace.centered_velocity), .vorticity = field::view(workspace.vorticity), .magnitude = field::scalar_view(workspace.magnitude), .normal = field::view(workspace.normal)};
        }

        kernels::VorticityAdjointScratch view(VorticityConfinement::AdjointWorkspace& workspace) {
            return {.centered_velocity = field::view(workspace.centered_velocity), .vorticity = field::view(workspace.vorticity), .magnitude = field::scalar_view(workspace.magnitude), .normal = field::view(workspace.normal)};
        }
    } // namespace

    VorticityConfinement::Cache VorticityConfinement::allocate_cache(const Domain& domain) const {
        return {
            .centered_velocity = domain.grid.allocate_cell_vector_field<float>(),
            .vorticity         = domain.grid.allocate_cell_vector_field<float>(),
            .magnitude         = domain.grid.allocate_cell_field<float>(),
            .normal            = domain.grid.allocate_cell_vector_field<float>(),
            .normalizer        = domain.grid.allocate_cell_field<float>(),
        };
    }

    VorticityConfinement::TangentWorkspace VorticityConfinement::allocate_tangent_workspace(const Domain& domain) const {
        return {
            .centered_velocity = domain.grid.allocate_cell_vector_field<float>(),
            .vorticity         = domain.grid.allocate_cell_vector_field<float>(),
            .magnitude         = domain.grid.allocate_cell_field<float>(),
            .normal            = domain.grid.allocate_cell_vector_field<float>(),
        };
    }

    VorticityConfinement::AdjointWorkspace VorticityConfinement::allocate_adjoint_workspace(const Domain& domain) const {
        return {
            .centered_velocity = domain.grid.allocate_cell_vector_field<double>(),
            .vorticity         = domain.grid.allocate_cell_vector_field<double>(),
            .magnitude         = domain.grid.allocate_cell_field<double>(),
            .normal            = domain.grid.allocate_cell_vector_field<double>(),
        };
    }

    void VorticityConfinement::forward(const Domain& domain, const VectorField<float>& velocity, const float* confinement, VectorField<float>& force, Cache& cache) const {
        kernels::vorticity_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity), confinement, view(cache), field::view(force));
    }

    void VorticityConfinement::forward(const Domain& domain, const VectorField<float>& velocity, const float confinement, VectorField<float>& force, Cache& cache) const {
        kernels::vorticity_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity), confinement, view(cache), field::view(force));
    }

    void VorticityConfinement::jvp(const Domain& domain, const VectorField<float>& velocity_tangent, const float* confinement, const float* confinement_tangent, const Cache& cache, VectorField<float>& force_tangent, TangentWorkspace& workspace) const {
        kernels::vorticity_jvp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity_tangent), confinement, confinement_tangent, view(cache), field::view(force_tangent), view(workspace));
    }

    void VorticityConfinement::jvp(const Domain& domain, const VectorField<float>& velocity_tangent, const float confinement, const Cache& cache, VectorField<float>& force_tangent, TangentWorkspace& workspace) const {
        kernels::vorticity_jvp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity_tangent), confinement, view(cache), field::view(force_tangent), view(workspace));
    }

    void VorticityConfinement::vjp(const Domain& domain, const float* confinement, const Cache& cache, const VectorField<double>& force_adjoint, VectorField<double>& velocity_adjoint, double* confinement_adjoint, AdjointWorkspace& workspace) const {
        kernels::vorticity_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), confinement, view(cache), field::view(force_adjoint), field::view(velocity_adjoint), confinement_adjoint, view(workspace));
    }

    void VorticityConfinement::vjp(const Domain& domain, const float confinement, const Cache& cache, const VectorField<double>& force_adjoint, VectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const {
        kernels::vorticity_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), confinement, view(cache), field::view(force_adjoint), field::view(velocity_adjoint), view(workspace));
    }

    ThermalBuoyancyVorticity::ThermalBuoyancyVorticity(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ThermalBuoyancyVorticity::Parameters ThermalBuoyancyVorticity::allocate_parameters(const Domain& domain) const {
        Parameters result{.values = ::cuda::device_buffer<float>(domain.grid.fields.stream, ::cuda::device_default_memory_pool(domain.grid.fields.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.grid.fields.stream, result.values, 0u);
        return result;
    }

    ThermalBuoyancyVorticity::ParameterTangent ThermalBuoyancyVorticity::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent result{.values = ::cuda::device_buffer<float>(domain.grid.fields.stream, ::cuda::device_default_memory_pool(domain.grid.fields.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.grid.fields.stream, result.values, 0u);
        return result;
    }

    ThermalBuoyancyVorticity::ParameterAdjoint ThermalBuoyancyVorticity::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint result{.values = ::cuda::device_buffer<double>(domain.grid.fields.stream, ::cuda::device_default_memory_pool(domain.grid.fields.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.grid.fields.stream, result.values, 0u);
        return result;
    }

    ThermalBuoyancyVorticity::Cache ThermalBuoyancyVorticity::allocate_cache(const Domain& domain) const {
        return {.force = domain.grid.allocate_cell_vector_field<float>(), .vorticity = vorticity.allocate_cache(domain)};
    }

    ThermalBuoyancyVorticity::TangentWorkspace ThermalBuoyancyVorticity::allocate_tangent_workspace(const Domain& domain) const {
        return {.force = domain.grid.allocate_cell_vector_field<float>(), .vorticity = vorticity.allocate_tangent_workspace(domain)};
    }

    ThermalBuoyancyVorticity::AdjointWorkspace ThermalBuoyancyVorticity::allocate_adjoint_workspace(const Domain& domain) const {
        return {.vorticity = vorticity.allocate_adjoint_workspace(domain)};
    }

    void ThermalBuoyancyVorticity::forward(const Domain& domain, const ScalarField<float>& density, const ScalarField<float>& temperature, const VectorField<float>& velocity, const VectorField<float>& external_acceleration, const Parameters& parameters, Cache& cache) const {
        kernels::buoyancy_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(density), field::scalar_view(temperature), field::view(external_acceleration), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, field::view(cache.force));
        if (configuration.vorticity_confinement_enabled) vorticity.forward(domain, velocity, parameters.values.data() + vorticity_confinement, cache.force, cache.vorticity);
    }

    void ThermalBuoyancyVorticity::jvp(const Domain& domain, const ScalarField<float>& density, const ScalarField<float>& temperature, const ScalarField<float>& density_tangent, const ScalarField<float>& temperature_tangent, const VectorField<float>& velocity_tangent, const VectorField<float>& external_acceleration_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Cache& cache, TangentWorkspace& workspace) const {
        kernels::buoyancy_jvp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(density), field::scalar_view(temperature), field::scalar_view(density_tangent), field::scalar_view(temperature_tangent), field::view(external_acceleration_tangent), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, parameter_tangent.values.data() + ambient_temperature, parameter_tangent.values.data() + density_buoyancy, parameter_tangent.values.data() + temperature_buoyancy, field::view(workspace.force));
        if (configuration.vorticity_confinement_enabled) vorticity.jvp(domain, velocity_tangent, parameters.values.data() + vorticity_confinement, parameter_tangent.values.data() + vorticity_confinement, cache.vorticity, workspace.force, workspace.vorticity);
    }

    void ThermalBuoyancyVorticity::vjp(const Domain& domain, const ScalarField<float>& density, const ScalarField<float>& temperature, const Parameters& parameters, const Cache& cache, const VectorField<double>& force_adjoint, VectorField<double>& velocity_adjoint, ScalarField<double>& density_adjoint, ScalarField<double>& temperature_adjoint, VectorField<double>& external_acceleration_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
        kernels::buoyancy_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(density), field::scalar_view(temperature), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, field::view(force_adjoint), field::scalar_view(density_adjoint), field::scalar_view(temperature_adjoint), field::view(external_acceleration_adjoint), parameter_adjoint.values.data() + ambient_temperature, parameter_adjoint.values.data() + density_buoyancy, parameter_adjoint.values.data() + temperature_buoyancy);
        if (configuration.vorticity_confinement_enabled) vorticity.vjp(domain, parameters.values.data() + vorticity_confinement, cache.vorticity, force_adjoint, velocity_adjoint, parameter_adjoint.values.data() + vorticity_confinement, workspace.vorticity);
    }

    ControlledDensityBuoyancyVorticity::ControlledDensityBuoyancyVorticity(Configuration configuration) : vorticity_confinement(configuration.vorticity_confinement), buoyancy({.buoyancy = configuration.density_buoyancy}) {}

    ControlledDensityBuoyancyVorticity::Cache ControlledDensityBuoyancyVorticity::allocate_cache(const Domain& domain) const {
        return {
            .buoyancy  = buoyancy.allocate_cache(domain),
            .total     = domain.grid.allocate_cell_vector_field<float>(),
            .vorticity = vorticity.allocate_cache(domain),
        };
    }

    ControlledDensityBuoyancyVorticity::TangentWorkspace ControlledDensityBuoyancyVorticity::allocate_tangent_workspace(const Domain& domain) const {
        return {
            .buoyancy  = buoyancy.allocate_tangent_workspace(domain),
            .total     = domain.grid.allocate_cell_vector_field<float>(),
            .vorticity = vorticity.allocate_tangent_workspace(domain),
        };
    }

    ControlledDensityBuoyancyVorticity::AdjointWorkspace ControlledDensityBuoyancyVorticity::allocate_adjoint_workspace(const Domain& domain) const {
        return {
            .physical  = domain.grid.allocate_cell_vector_field<double>(),
            .buoyancy  = buoyancy.allocate_adjoint_workspace(domain),
            .vorticity = vorticity.allocate_adjoint_workspace(domain),
        };
    }

    void ControlledDensityBuoyancyVorticity::forward(const Domain& domain, const ScalarField<float>& density, const VectorField<float>& velocity, const VectorField<float>& control, Cache& cache) const {
        buoyancy.forward(domain, density, cache.buoyancy);
        vorticity.forward(domain, velocity, vorticity_confinement, cache.buoyancy.force, cache.vorticity);
        kernels::combine_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(cache.buoyancy.force), field::view(control), field::view(cache.total));
    }

    void ControlledDensityBuoyancyVorticity::jvp(const Domain& domain, const ScalarField<float>& density_tangent, const VectorField<float>& velocity_tangent, const VectorField<float>& control_tangent, const Cache& cache, TangentWorkspace& workspace) const {
        buoyancy.jvp(domain, density_tangent, cache.buoyancy, workspace.buoyancy);
        vorticity.jvp(domain, velocity_tangent, vorticity_confinement, cache.vorticity, workspace.buoyancy.force, workspace.vorticity);
        kernels::combine_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(workspace.buoyancy.force), field::view(control_tangent), field::view(workspace.total));
    }

    void ControlledDensityBuoyancyVorticity::vjp(const Domain& domain, const Cache& cache, const VectorField<double>& total_adjoint, ScalarField<double>& density_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& control_adjoint, AdjointWorkspace& workspace) const {
        kernels::combine_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(total_adjoint), field::view(workspace.physical), field::view(control_adjoint));
        buoyancy.vjp(domain, cache.buoyancy, workspace.physical, density_adjoint, workspace.buoyancy);
        vorticity.vjp(domain, vorticity_confinement, cache.vorticity, workspace.physical, velocity_adjoint, workspace.vorticity);
    }
} // namespace physica::fluids::gas::operators
