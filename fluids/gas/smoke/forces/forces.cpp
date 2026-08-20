module;

#include "../domain/interop.h"
#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>

module physica.fluids.gas.smoke.forces;

import std;

namespace physica::fluids::gas::smoke {
    namespace {
        cuda_detail::VorticityView vorticity(BuoyancyVorticityForces::VorticityCache& cache) {
            return {.centered_velocity = cuda_detail::centered(cache.centered_velocity), .vorticity = cuda_detail::centered(cache.vorticity), .magnitude = cuda_detail::scalar(cache.magnitude), .normal = cuda_detail::centered(cache.normal), .normalizer = cuda_detail::scalar(cache.normalizer)};
        }

        cuda_detail::ConstVorticityView vorticity(const BuoyancyVorticityForces::VorticityCache& cache) {
            return {.centered_velocity = cuda_detail::centered(cache.centered_velocity), .vorticity = cuda_detail::centered(cache.vorticity), .magnitude = cuda_detail::scalar(cache.magnitude), .normal = cuda_detail::centered(cache.normal), .normalizer = cuda_detail::scalar(cache.normalizer)};
        }

        cuda_detail::VorticityTangentScratch vorticity_tangent(BuoyancyVorticityForces::VorticityCache& cache) {
            return {.centered_velocity = cuda_detail::centered(cache.centered_velocity), .vorticity = cuda_detail::centered(cache.vorticity), .magnitude = cuda_detail::scalar(cache.magnitude), .normal = cuda_detail::centered(cache.normal)};
        }

        cuda_detail::VorticityAdjointScratch vorticity_adjoint(BuoyancyVorticityForces::VorticityAdjointCache& cache) {
            return {.centered_velocity = cuda_detail::centered_adjoint(cache.centered_velocity), .vorticity = cuda_detail::centered_adjoint(cache.vorticity), .magnitude = cuda_detail::scalar_adjoint(cache.magnitude), .normal = cuda_detail::centered_adjoint(cache.normal)};
        }

        BuoyancyVorticityForces::VorticityCache allocate_vorticity_cache(const Domain& domain) {
            return {
                .centered_velocity = domain.allocate_centered_vector_field(),
                .vorticity         = domain.allocate_centered_vector_field(),
                .magnitude         = domain.allocate_scalar_field(),
                .normal            = domain.allocate_centered_vector_field(),
                .normalizer        = domain.allocate_scalar_field(),
            };
        }

        BuoyancyVorticityForces::VorticityAdjointCache allocate_vorticity_adjoint_cache(const Domain& domain) {
            return {
                .centered_velocity = domain.allocate_centered_vector_adjoint_field(),
                .vorticity         = domain.allocate_centered_vector_adjoint_field(),
                .magnitude         = domain.allocate_scalar_adjoint_field(),
                .normal            = domain.allocate_centered_vector_adjoint_field(),
            };
        }
    } // namespace

    BuoyancyVorticityForces::BuoyancyVorticityForces(const Domain& domain, Configuration next_configuration, const ExecutionMode mode)
        : configuration(std::move(next_configuration)), differentiation{} {
        if (mode == ExecutionMode::differentiable) {
            differentiation.emplace(Differentiation{
                .force_tangent     = domain.allocate_centered_vector_field(),
                .vorticity_tangent = allocate_vorticity_cache(domain),
                .vorticity_adjoint = allocate_vorticity_adjoint_cache(domain),
                .force_adjoint     = domain.allocate_centered_vector_adjoint_field(),
            });
        }
    }

    BuoyancyVorticityForces::Parameters BuoyancyVorticityForces::allocate_parameters(const Domain& domain) const {
        Parameters value{
            .ambient_temperature   = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .density_buoyancy      = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .temperature_buoyancy  = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .vorticity_confinement = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
        };
        ::cuda::fill_bytes(domain.stream, value.ambient_temperature, 0u);
        ::cuda::fill_bytes(domain.stream, value.density_buoyancy, 0u);
        ::cuda::fill_bytes(domain.stream, value.temperature_buoyancy, 0u);
        ::cuda::fill_bytes(domain.stream, value.vorticity_confinement, 0u);
        return value;
    }

    BuoyancyVorticityForces::ParameterTangent BuoyancyVorticityForces::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent value{
            .ambient_temperature   = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .density_buoyancy      = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .temperature_buoyancy  = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .vorticity_confinement = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
        };
        ::cuda::fill_bytes(domain.stream, value.ambient_temperature, 0u);
        ::cuda::fill_bytes(domain.stream, value.density_buoyancy, 0u);
        ::cuda::fill_bytes(domain.stream, value.temperature_buoyancy, 0u);
        ::cuda::fill_bytes(domain.stream, value.vorticity_confinement, 0u);
        return value;
    }

    BuoyancyVorticityForces::ParameterAdjoint BuoyancyVorticityForces::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint value{
            .ambient_temperature   = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .density_buoyancy      = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .temperature_buoyancy  = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .vorticity_confinement = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
        };
        ::cuda::fill_bytes(domain.stream, value.ambient_temperature, 0u);
        ::cuda::fill_bytes(domain.stream, value.density_buoyancy, 0u);
        ::cuda::fill_bytes(domain.stream, value.temperature_buoyancy, 0u);
        ::cuda::fill_bytes(domain.stream, value.vorticity_confinement, 0u);
        return value;
    }

    BuoyancyVorticityForces::Cache BuoyancyVorticityForces::allocate_cache(const Domain& domain) const {
        return {.force = domain.allocate_centered_vector_field(), .vorticity = allocate_vorticity_cache(domain)};
    }

    void BuoyancyVorticityForces::forward(const Domain& domain, const ScalarField& density, const ScalarField& temperature, const StaggeredVectorField& velocity, const CenteredVectorField& external_acceleration, const Parameters& parameters, Cache& cache) const {
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        cuda_detail::buoyancy_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(density), cuda_detail::scalar(temperature), cuda_detail::centered(external_acceleration), parameters.ambient_temperature.data(), parameters.density_buoyancy.data(), parameters.temperature_buoyancy.data(), cuda_detail::centered(cache.force));
        if (configuration.vorticity_confinement_enabled) cuda_detail::vorticity_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(velocity), parameters.vorticity_confinement.data(), vorticity(cache.vorticity), cuda_detail::centered(cache.force));
    }

    void BuoyancyVorticityForces::jvp(const Domain& domain, const ScalarField& density, const ScalarField& temperature, const ScalarField& density_tangent, const ScalarField& temperature_tangent, const StaggeredVectorField& velocity_tangent, const CenteredVectorField& external_acceleration_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Cache& cache) {
        Differentiation& workspace = *differentiation;
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        cuda_detail::buoyancy_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(density), cuda_detail::scalar(temperature), cuda_detail::scalar(density_tangent), cuda_detail::scalar(temperature_tangent), cuda_detail::centered(external_acceleration_tangent), parameters.ambient_temperature.data(), parameters.density_buoyancy.data(), parameters.temperature_buoyancy.data(), parameter_tangent.ambient_temperature.data(), parameter_tangent.density_buoyancy.data(), parameter_tangent.temperature_buoyancy.data(), cuda_detail::centered(workspace.force_tangent));
        if (configuration.vorticity_confinement_enabled) cuda_detail::vorticity_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(velocity_tangent), parameters.vorticity_confinement.data(), parameter_tangent.vorticity_confinement.data(), vorticity(cache.vorticity), cuda_detail::centered(workspace.force_tangent), vorticity_tangent(workspace.vorticity_tangent));
    }

    void BuoyancyVorticityForces::vjp(const Domain& domain, const ScalarField& density, const ScalarField& temperature, const Parameters& parameters, const Cache& cache, StaggeredVectorAdjointField& velocity_adjoint, ScalarAdjointField& density_adjoint, ScalarAdjointField& temperature_adjoint, CenteredVectorAdjointField& external_acceleration_adjoint, ParameterAdjoint& parameter_adjoint) {
        Differentiation& workspace = *differentiation;
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        cuda_detail::buoyancy_vjp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(density), cuda_detail::scalar(temperature), parameters.ambient_temperature.data(), parameters.density_buoyancy.data(), parameters.temperature_buoyancy.data(), cuda_detail::centered_adjoint(workspace.force_adjoint), cuda_detail::scalar_adjoint(density_adjoint), cuda_detail::scalar_adjoint(temperature_adjoint), cuda_detail::centered_adjoint(external_acceleration_adjoint), parameter_adjoint.ambient_temperature.data(), parameter_adjoint.density_buoyancy.data(), parameter_adjoint.temperature_buoyancy.data());
        if (configuration.vorticity_confinement_enabled) cuda_detail::vorticity_vjp(domain.stream, grid, domain.cell_mask.data(), parameters.vorticity_confinement.data(), vorticity(cache.vorticity), cuda_detail::centered_adjoint(workspace.force_adjoint), cuda_detail::staggered_adjoint(velocity_adjoint), parameter_adjoint.vorticity_confinement.data(), vorticity_adjoint(workspace.vorticity_adjoint));
    }
} // namespace physica::fluids::gas::smoke
