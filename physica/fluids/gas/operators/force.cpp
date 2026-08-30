module;

#include <fluids/gas/interop.h>
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

    void DensityBuoyancy::forward(const Domain& domain, const simulation::ScalarField<float>& density, Cache& cache) const {
        kernels::density_buoyancy_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, simulation::scalar_view(density), simulation::view(cache.force));
    }

    void DensityBuoyancy::jvp(const Domain& domain, const simulation::ScalarField<float>& density_tangent, const Cache&, TangentWorkspace& workspace) const {
        kernels::density_buoyancy_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, simulation::scalar_view(density_tangent), simulation::view(workspace.force));
    }

    void DensityBuoyancy::vjp(const Domain& domain, const Cache&, const simulation::VectorField<double>& force_adjoint, simulation::ScalarField<double>& density_adjoint, AdjointWorkspace&) const {
        kernels::density_buoyancy_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, simulation::view(force_adjoint), simulation::scalar_view(density_adjoint));
    }

    namespace {
        kernels::VorticityView view(VorticityConfinement::Cache& cache) {
            return {.centered_velocity = simulation::view(cache.centered_velocity), .vorticity = simulation::view(cache.vorticity), .magnitude = simulation::scalar_view(cache.magnitude), .normal = simulation::view(cache.normal), .normalizer = simulation::scalar_view(cache.normalizer)};
        }

        kernels::ConstVorticityView view(const VorticityConfinement::Cache& cache) {
            return {.centered_velocity = simulation::view(cache.centered_velocity), .vorticity = simulation::view(cache.vorticity), .magnitude = simulation::scalar_view(cache.magnitude), .normal = simulation::view(cache.normal), .normalizer = simulation::scalar_view(cache.normalizer)};
        }

        kernels::VorticityTangentScratch view(VorticityConfinement::TangentWorkspace& workspace) {
            return {.centered_velocity = simulation::view(workspace.centered_velocity), .vorticity = simulation::view(workspace.vorticity), .magnitude = simulation::scalar_view(workspace.magnitude), .normal = simulation::view(workspace.normal)};
        }

        kernels::VorticityAdjointScratch view(VorticityConfinement::AdjointWorkspace& workspace) {
            return {.centered_velocity = simulation::view(workspace.centered_velocity), .vorticity = simulation::view(workspace.vorticity), .magnitude = simulation::scalar_view(workspace.magnitude), .normal = simulation::view(workspace.normal)};
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

    void VorticityConfinement::forward(const Domain& domain, const simulation::VectorField<float>& velocity, const float* confinement, simulation::VectorField<float>& force, Cache& cache) const {
        kernels::vorticity_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity), confinement, view(cache), simulation::view(force));
    }

    void VorticityConfinement::forward(const Domain& domain, const simulation::VectorField<float>& velocity, const float confinement, simulation::VectorField<float>& force, Cache& cache) const {
        kernels::vorticity_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity), confinement, view(cache), simulation::view(force));
    }

    void VorticityConfinement::jvp(const Domain& domain, const simulation::VectorField<float>& velocity_tangent, const float* confinement, const float* confinement_tangent, const Cache& cache, simulation::VectorField<float>& force_tangent, TangentWorkspace& workspace) const {
        kernels::vorticity_jvp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity_tangent), confinement, confinement_tangent, view(cache), simulation::view(force_tangent), view(workspace));
    }

    void VorticityConfinement::jvp(const Domain& domain, const simulation::VectorField<float>& velocity_tangent, const float confinement, const Cache& cache, simulation::VectorField<float>& force_tangent, TangentWorkspace& workspace) const {
        kernels::vorticity_jvp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity_tangent), confinement, view(cache), simulation::view(force_tangent), view(workspace));
    }

    void VorticityConfinement::vjp(const Domain& domain, const float* confinement, const Cache& cache, const simulation::VectorField<double>& force_adjoint, simulation::VectorField<double>& velocity_adjoint, double* confinement_adjoint, AdjointWorkspace& workspace) const {
        kernels::vorticity_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), confinement, view(cache), simulation::view(force_adjoint), simulation::view(velocity_adjoint), confinement_adjoint, view(workspace));
    }

    void VorticityConfinement::vjp(const Domain& domain, const float confinement, const Cache& cache, const simulation::VectorField<double>& force_adjoint, simulation::VectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const {
        kernels::vorticity_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), confinement, view(cache), simulation::view(force_adjoint), simulation::view(velocity_adjoint), view(workspace));
    }

    ThermalBuoyancyVorticity::ThermalBuoyancyVorticity(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ThermalBuoyancyVorticity::Parameters ThermalBuoyancyVorticity::allocate_parameters(const Domain& domain) const {
        Parameters result{.values = ::cuda::device_buffer<float>(domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.grid.stream, result.values, 0u);
        return result;
    }

    ThermalBuoyancyVorticity::ParameterTangent ThermalBuoyancyVorticity::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent result{.values = ::cuda::device_buffer<float>(domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.grid.stream, result.values, 0u);
        return result;
    }

    ThermalBuoyancyVorticity::ParameterAdjoint ThermalBuoyancyVorticity::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint result{.values = ::cuda::device_buffer<double>(domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.grid.stream, result.values, 0u);
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

    void ThermalBuoyancyVorticity::forward(const Domain& domain, const simulation::ScalarField<float>& density, const simulation::ScalarField<float>& temperature, const simulation::VectorField<float>& velocity, const simulation::VectorField<float>& external_acceleration, const Parameters& parameters, Cache& cache) const {
        kernels::buoyancy_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(density), simulation::scalar_view(temperature), simulation::view(external_acceleration), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, simulation::view(cache.force));
        if (configuration.vorticity_confinement_enabled) vorticity.forward(domain, velocity, parameters.values.data() + vorticity_confinement, cache.force, cache.vorticity);
    }

    void ThermalBuoyancyVorticity::jvp(const Domain& domain, const simulation::ScalarField<float>& density, const simulation::ScalarField<float>& temperature, const simulation::ScalarField<float>& density_tangent, const simulation::ScalarField<float>& temperature_tangent, const simulation::VectorField<float>& velocity_tangent, const simulation::VectorField<float>& external_acceleration_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Cache& cache, TangentWorkspace& workspace) const {
        kernels::buoyancy_jvp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(density), simulation::scalar_view(temperature), simulation::scalar_view(density_tangent), simulation::scalar_view(temperature_tangent), simulation::view(external_acceleration_tangent), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, parameter_tangent.values.data() + ambient_temperature, parameter_tangent.values.data() + density_buoyancy, parameter_tangent.values.data() + temperature_buoyancy, simulation::view(workspace.force));
        if (configuration.vorticity_confinement_enabled) vorticity.jvp(domain, velocity_tangent, parameters.values.data() + vorticity_confinement, parameter_tangent.values.data() + vorticity_confinement, cache.vorticity, workspace.force, workspace.vorticity);
    }

    void ThermalBuoyancyVorticity::vjp(const Domain& domain, const simulation::ScalarField<float>& density, const simulation::ScalarField<float>& temperature, const Parameters& parameters, const Cache& cache, const simulation::VectorField<double>& force_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::ScalarField<double>& density_adjoint, simulation::ScalarField<double>& temperature_adjoint, simulation::VectorField<double>& external_acceleration_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
        kernels::buoyancy_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(density), simulation::scalar_view(temperature), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, simulation::view(force_adjoint), simulation::scalar_view(density_adjoint), simulation::scalar_view(temperature_adjoint), simulation::view(external_acceleration_adjoint), parameter_adjoint.values.data() + ambient_temperature, parameter_adjoint.values.data() + density_buoyancy, parameter_adjoint.values.data() + temperature_buoyancy);
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

    void ControlledDensityBuoyancyVorticity::forward(const Domain& domain, const simulation::ScalarField<float>& density, const simulation::VectorField<float>& velocity, const simulation::VectorField<float>& control, Cache& cache) const {
        buoyancy.forward(domain, density, cache.buoyancy);
        vorticity.forward(domain, velocity, vorticity_confinement, cache.buoyancy.force, cache.vorticity);
        kernels::combine_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(cache.buoyancy.force), simulation::view(control), simulation::view(cache.total));
    }

    void ControlledDensityBuoyancyVorticity::jvp(const Domain& domain, const simulation::ScalarField<float>& density_tangent, const simulation::VectorField<float>& velocity_tangent, const simulation::VectorField<float>& control_tangent, const Cache& cache, TangentWorkspace& workspace) const {
        buoyancy.jvp(domain, density_tangent, cache.buoyancy, workspace.buoyancy);
        vorticity.jvp(domain, velocity_tangent, vorticity_confinement, cache.vorticity, workspace.buoyancy.force, workspace.vorticity);
        kernels::combine_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(workspace.buoyancy.force), simulation::view(control_tangent), simulation::view(workspace.total));
    }

    void ControlledDensityBuoyancyVorticity::vjp(const Domain& domain, const Cache& cache, const simulation::VectorField<double>& total_adjoint, simulation::ScalarField<double>& density_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& control_adjoint, AdjointWorkspace& workspace) const {
        kernels::combine_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(total_adjoint), simulation::view(workspace.physical), simulation::view(control_adjoint));
        buoyancy.vjp(domain, cache.buoyancy, workspace.physical, density_adjoint, workspace.buoyancy);
        vorticity.vjp(domain, vorticity_confinement, cache.vorticity, workspace.physical, velocity_adjoint, workspace.vorticity);
    }
} // namespace physica::fluids::gas::operators
