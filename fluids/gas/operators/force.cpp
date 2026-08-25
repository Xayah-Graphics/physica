module;

#include "../detail/cuda/interop.h"
#include "force-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.force;

import std;

namespace physica::fluids::gas::operators {
    DensityBuoyancy::DensityBuoyancy(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    DensityBuoyancy::Cache DensityBuoyancy::allocate_cache(const Domain& domain) const {
        return {.force = domain.allocate_centered_vector_field<float>()};
    }

    DensityBuoyancy::TangentWorkspace DensityBuoyancy::allocate_tangent_workspace(const Domain& domain) const {
        return {.force = domain.allocate_centered_vector_field<float>()};
    }

    DensityBuoyancy::AdjointWorkspace DensityBuoyancy::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void DensityBuoyancy::forward(const Domain& domain, const CellField<float>& density, Cache& cache) const {
        cuda_backend::density_buoyancy_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, detail::cuda::scalar(density), detail::cuda::centered(cache.force));
    }

    void DensityBuoyancy::jvp(const Domain& domain, const CellField<float>& density_tangent, const Cache&, TangentWorkspace& workspace) const {
        cuda_backend::density_buoyancy_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, detail::cuda::scalar(density_tangent), detail::cuda::centered(workspace.force));
    }

    void DensityBuoyancy::vjp(const Domain& domain, const Cache&, const CenteredVectorField<double>& force_adjoint, CellField<double>& density_adjoint, AdjointWorkspace&) const {
        cuda_backend::density_buoyancy_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), configuration.buoyancy, detail::cuda::centered_adjoint(force_adjoint), detail::cuda::scalar_adjoint(density_adjoint));
    }

    namespace {
        cuda_backend::VorticityView view(VorticityConfinement::Cache& cache) {
            return {.centered_velocity = detail::cuda::centered(cache.centered_velocity), .vorticity = detail::cuda::centered(cache.vorticity), .magnitude = detail::cuda::scalar(cache.magnitude), .normal = detail::cuda::centered(cache.normal), .normalizer = detail::cuda::scalar(cache.normalizer)};
        }

        cuda_backend::ConstVorticityView view(const VorticityConfinement::Cache& cache) {
            return {.centered_velocity = detail::cuda::centered(cache.centered_velocity), .vorticity = detail::cuda::centered(cache.vorticity), .magnitude = detail::cuda::scalar(cache.magnitude), .normal = detail::cuda::centered(cache.normal), .normalizer = detail::cuda::scalar(cache.normalizer)};
        }

        cuda_backend::VorticityTangentScratch view(VorticityConfinement::TangentWorkspace& workspace) {
            return {.centered_velocity = detail::cuda::centered(workspace.centered_velocity), .vorticity = detail::cuda::centered(workspace.vorticity), .magnitude = detail::cuda::scalar(workspace.magnitude), .normal = detail::cuda::centered(workspace.normal)};
        }

        cuda_backend::VorticityAdjointScratch view(VorticityConfinement::AdjointWorkspace& workspace) {
            return {.centered_velocity = detail::cuda::centered_adjoint(workspace.centered_velocity), .vorticity = detail::cuda::centered_adjoint(workspace.vorticity), .magnitude = detail::cuda::scalar_adjoint(workspace.magnitude), .normal = detail::cuda::centered_adjoint(workspace.normal)};
        }
    } // namespace

    VorticityConfinement::Cache VorticityConfinement::allocate_cache(const Domain& domain) const {
        return {
            .centered_velocity = domain.allocate_centered_vector_field<float>(),
            .vorticity         = domain.allocate_centered_vector_field<float>(),
            .magnitude         = domain.allocate_cell_field<float>(),
            .normal            = domain.allocate_centered_vector_field<float>(),
            .normalizer        = domain.allocate_cell_field<float>(),
        };
    }

    VorticityConfinement::TangentWorkspace VorticityConfinement::allocate_tangent_workspace(const Domain& domain) const {
        return {
            .centered_velocity = domain.allocate_centered_vector_field<float>(),
            .vorticity         = domain.allocate_centered_vector_field<float>(),
            .magnitude         = domain.allocate_cell_field<float>(),
            .normal            = domain.allocate_centered_vector_field<float>(),
        };
    }

    VorticityConfinement::AdjointWorkspace VorticityConfinement::allocate_adjoint_workspace(const Domain& domain) const {
        return {
            .centered_velocity = domain.allocate_centered_vector_field<double>(),
            .vorticity         = domain.allocate_centered_vector_field<double>(),
            .magnitude         = domain.allocate_cell_field<double>(),
            .normal            = domain.allocate_centered_vector_field<double>(),
        };
    }

    void VorticityConfinement::forward(const Domain& domain, const StaggeredVectorField<float>& velocity, const float* confinement, CenteredVectorField<float>& force, Cache& cache) const {
        cuda_backend::vorticity_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity), confinement, view(cache), detail::cuda::centered(force));
    }

    void VorticityConfinement::forward(const Domain& domain, const StaggeredVectorField<float>& velocity, const float confinement, CenteredVectorField<float>& force, Cache& cache) const {
        cuda_backend::vorticity_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity), confinement, view(cache), detail::cuda::centered(force));
    }

    void VorticityConfinement::jvp(const Domain& domain, const StaggeredVectorField<float>& velocity_tangent, const float* confinement, const float* confinement_tangent, const Cache& cache, CenteredVectorField<float>& force_tangent, TangentWorkspace& workspace) const {
        cuda_backend::vorticity_jvp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity_tangent), confinement, confinement_tangent, view(cache), detail::cuda::centered(force_tangent), view(workspace));
    }

    void VorticityConfinement::jvp(const Domain& domain, const StaggeredVectorField<float>& velocity_tangent, const float confinement, const Cache& cache, CenteredVectorField<float>& force_tangent, TangentWorkspace& workspace) const {
        cuda_backend::vorticity_jvp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity_tangent), confinement, view(cache), detail::cuda::centered(force_tangent), view(workspace));
    }

    void VorticityConfinement::vjp(const Domain& domain, const float* confinement, const Cache& cache, const CenteredVectorField<double>& force_adjoint, StaggeredVectorField<double>& velocity_adjoint, double* confinement_adjoint, AdjointWorkspace& workspace) const {
        cuda_backend::vorticity_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), confinement, view(cache), detail::cuda::centered_adjoint(force_adjoint), detail::cuda::staggered_adjoint(velocity_adjoint), confinement_adjoint, view(workspace));
    }

    void VorticityConfinement::vjp(const Domain& domain, const float confinement, const Cache& cache, const CenteredVectorField<double>& force_adjoint, StaggeredVectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const {
        cuda_backend::vorticity_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), confinement, view(cache), detail::cuda::centered_adjoint(force_adjoint), detail::cuda::staggered_adjoint(velocity_adjoint), view(workspace));
    }

    ThermalBuoyancyVorticity::ThermalBuoyancyVorticity(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ThermalBuoyancyVorticity::Parameters ThermalBuoyancyVorticity::allocate_parameters(const Domain& domain) const {
        Parameters result{.values = ::cuda::device_buffer<float>(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.stream, result.values, 0u);
        return result;
    }

    ThermalBuoyancyVorticity::ParameterTangent ThermalBuoyancyVorticity::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent result{.values = ::cuda::device_buffer<float>(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.stream, result.values, 0u);
        return result;
    }

    ThermalBuoyancyVorticity::ParameterAdjoint ThermalBuoyancyVorticity::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint result{.values = ::cuda::device_buffer<double>(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 4u, ::cuda::no_init)};
        ::cuda::fill_bytes(domain.stream, result.values, 0u);
        return result;
    }

    ThermalBuoyancyVorticity::Cache ThermalBuoyancyVorticity::allocate_cache(const Domain& domain) const {
        return {.force = domain.allocate_centered_vector_field<float>(), .vorticity = vorticity.allocate_cache(domain)};
    }

    ThermalBuoyancyVorticity::TangentWorkspace ThermalBuoyancyVorticity::allocate_tangent_workspace(const Domain& domain) const {
        return {.force = domain.allocate_centered_vector_field<float>(), .vorticity = vorticity.allocate_tangent_workspace(domain)};
    }

    ThermalBuoyancyVorticity::AdjointWorkspace ThermalBuoyancyVorticity::allocate_adjoint_workspace(const Domain& domain) const {
        return {.vorticity = vorticity.allocate_adjoint_workspace(domain)};
    }

    void ThermalBuoyancyVorticity::forward(const Domain& domain, const CellField<float>& density, const CellField<float>& temperature, const StaggeredVectorField<float>& velocity, const CenteredVectorField<float>& external_acceleration, const Parameters& parameters, Cache& cache) const {
        cuda_backend::buoyancy_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar(density), detail::cuda::scalar(temperature), detail::cuda::centered(external_acceleration), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, detail::cuda::centered(cache.force));
        if (configuration.vorticity_confinement_enabled) vorticity.forward(domain, velocity, parameters.values.data() + vorticity_confinement, cache.force, cache.vorticity);
    }

    void ThermalBuoyancyVorticity::jvp(const Domain& domain, const CellField<float>& density, const CellField<float>& temperature, const CellField<float>& density_tangent, const CellField<float>& temperature_tangent, const StaggeredVectorField<float>& velocity_tangent, const CenteredVectorField<float>& external_acceleration_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Cache& cache, TangentWorkspace& workspace) const {
        cuda_backend::buoyancy_jvp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar(density), detail::cuda::scalar(temperature), detail::cuda::scalar(density_tangent), detail::cuda::scalar(temperature_tangent), detail::cuda::centered(external_acceleration_tangent), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, parameter_tangent.values.data() + ambient_temperature, parameter_tangent.values.data() + density_buoyancy, parameter_tangent.values.data() + temperature_buoyancy, detail::cuda::centered(workspace.force));
        if (configuration.vorticity_confinement_enabled) vorticity.jvp(domain, velocity_tangent, parameters.values.data() + vorticity_confinement, parameter_tangent.values.data() + vorticity_confinement, cache.vorticity, workspace.force, workspace.vorticity);
    }

    void ThermalBuoyancyVorticity::vjp(const Domain& domain, const CellField<float>& density, const CellField<float>& temperature, const Parameters& parameters, const Cache& cache, const CenteredVectorField<double>& force_adjoint, StaggeredVectorField<double>& velocity_adjoint, CellField<double>& density_adjoint, CellField<double>& temperature_adjoint, CenteredVectorField<double>& external_acceleration_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
        cuda_backend::buoyancy_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar(density), detail::cuda::scalar(temperature), parameters.values.data() + ambient_temperature, parameters.values.data() + density_buoyancy, parameters.values.data() + temperature_buoyancy, detail::cuda::centered_adjoint(force_adjoint), detail::cuda::scalar_adjoint(density_adjoint), detail::cuda::scalar_adjoint(temperature_adjoint), detail::cuda::centered_adjoint(external_acceleration_adjoint), parameter_adjoint.values.data() + ambient_temperature, parameter_adjoint.values.data() + density_buoyancy, parameter_adjoint.values.data() + temperature_buoyancy);
        if (configuration.vorticity_confinement_enabled) vorticity.vjp(domain, parameters.values.data() + vorticity_confinement, cache.vorticity, force_adjoint, velocity_adjoint, parameter_adjoint.values.data() + vorticity_confinement, workspace.vorticity);
    }

    ControlledDensityBuoyancyVorticity::ControlledDensityBuoyancyVorticity(Configuration configuration) : vorticity_confinement(configuration.vorticity_confinement), buoyancy({.buoyancy = configuration.density_buoyancy}) {}

    ControlledDensityBuoyancyVorticity::Cache ControlledDensityBuoyancyVorticity::allocate_cache(const Domain& domain) const {
        return {
            .buoyancy  = buoyancy.allocate_cache(domain),
            .total     = domain.allocate_centered_vector_field<float>(),
            .vorticity = vorticity.allocate_cache(domain),
        };
    }

    ControlledDensityBuoyancyVorticity::TangentWorkspace ControlledDensityBuoyancyVorticity::allocate_tangent_workspace(const Domain& domain) const {
        return {
            .buoyancy  = buoyancy.allocate_tangent_workspace(domain),
            .total     = domain.allocate_centered_vector_field<float>(),
            .vorticity = vorticity.allocate_tangent_workspace(domain),
        };
    }

    ControlledDensityBuoyancyVorticity::AdjointWorkspace ControlledDensityBuoyancyVorticity::allocate_adjoint_workspace(const Domain& domain) const {
        return {
            .physical  = domain.allocate_centered_vector_field<double>(),
            .buoyancy  = buoyancy.allocate_adjoint_workspace(domain),
            .vorticity = vorticity.allocate_adjoint_workspace(domain),
        };
    }

    void ControlledDensityBuoyancyVorticity::forward(const Domain& domain, const CellField<float>& density, const StaggeredVectorField<float>& velocity, const CenteredVectorField<float>& control, Cache& cache) const {
        buoyancy.forward(domain, density, cache.buoyancy);
        vorticity.forward(domain, velocity, vorticity_confinement, cache.buoyancy.force, cache.vorticity);
        cuda_backend::combine_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::centered(cache.buoyancy.force), detail::cuda::centered(control), detail::cuda::centered(cache.total));
    }

    void ControlledDensityBuoyancyVorticity::jvp(const Domain& domain, const CellField<float>& density_tangent, const StaggeredVectorField<float>& velocity_tangent, const CenteredVectorField<float>& control_tangent, const Cache& cache, TangentWorkspace& workspace) const {
        buoyancy.jvp(domain, density_tangent, cache.buoyancy, workspace.buoyancy);
        vorticity.jvp(domain, velocity_tangent, vorticity_confinement, cache.vorticity, workspace.buoyancy.force, workspace.vorticity);
        cuda_backend::combine_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::centered(workspace.buoyancy.force), detail::cuda::centered(control_tangent), detail::cuda::centered(workspace.total));
    }

    void ControlledDensityBuoyancyVorticity::vjp(const Domain& domain, const Cache& cache, const CenteredVectorField<double>& total_adjoint, CellField<double>& density_adjoint, StaggeredVectorField<double>& velocity_adjoint, CenteredVectorField<double>& control_adjoint, AdjointWorkspace& workspace) const {
        cuda_backend::combine_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::centered_adjoint(total_adjoint), detail::cuda::centered_adjoint(workspace.physical), detail::cuda::centered_adjoint(control_adjoint));
        buoyancy.vjp(domain, cache.buoyancy, workspace.physical, density_adjoint, workspace.buoyancy);
        vorticity.vjp(domain, vorticity_confinement, cache.vorticity, workspace.physical, velocity_adjoint, workspace.vorticity);
    }
} // namespace physica::fluids::gas::operators
