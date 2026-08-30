module;

#include <fluids/gas/interop.h>
#include "conservation-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.conservation;

import std;

namespace physica::fluids::gas::operators {
    MassConservation::MassConservation(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    MassConservation::Cache MassConservation::allocate_cache(const Domain& domain) const {
        return {
            .input_mass    = ::cuda::device_buffer<double>{domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 1u, ::cuda::no_init},
            .advected_mass = ::cuda::device_buffer<double>{domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 1u, ::cuda::no_init},
        };
    }

    MassConservation::TangentWorkspace MassConservation::allocate_tangent_workspace(const Domain& domain) const {
        return {
            .input_mass    = ::cuda::device_buffer<double>{domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 1u, ::cuda::no_init},
            .advected_mass = ::cuda::device_buffer<double>{domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 1u, ::cuda::no_init},
        };
    }

    MassConservation::AdjointWorkspace MassConservation::allocate_adjoint_workspace(const Domain& domain) const {
        return {.density_dot = ::cuda::device_buffer<double>{domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 1u, ::cuda::no_init}};
    }

    void MassConservation::forward(const Domain& domain, const simulation::ScalarField<float>& input, const simulation::ScalarField<float>& advected, simulation::ScalarField<float>& output, Cache& cache) const {
        const float retention = std::exp(-configuration.dissipation * domain.configuration.time_step);
        kernels::mass_forward(domain.grid.stream, device::discretization(domain.configuration), retention, simulation::scalar_view(input), simulation::scalar_view(advected), cache.input_mass.data(), cache.advected_mass.data(), simulation::scalar_view(output));
    }

    void MassConservation::jvp(const Domain& domain, const simulation::ScalarField<float>& input, const simulation::ScalarField<float>& advected, const simulation::ScalarField<float>& input_tangent, const simulation::ScalarField<float>& advected_tangent, const Cache& cache, simulation::ScalarField<float>& output_tangent, TangentWorkspace& workspace) const {
        const float retention = std::exp(-configuration.dissipation * domain.configuration.time_step);
        kernels::mass_jvp(domain.grid.stream, device::discretization(domain.configuration), retention, simulation::scalar_view(input), simulation::scalar_view(advected), simulation::scalar_view(input_tangent), simulation::scalar_view(advected_tangent), cache.input_mass.data(), cache.advected_mass.data(), workspace.input_mass.data(), workspace.advected_mass.data(), simulation::scalar_view(output_tangent));
    }

    void MassConservation::vjp(const Domain& domain, const simulation::ScalarField<float>& advected, const Cache& cache, const simulation::ScalarField<double>& output_adjoint, simulation::ScalarField<double>& input_adjoint, simulation::ScalarField<double>& advected_adjoint, AdjointWorkspace& workspace) const {
        const float retention = std::exp(-configuration.dissipation * domain.configuration.time_step);
        kernels::mass_vjp(domain.grid.stream, device::discretization(domain.configuration), retention, simulation::scalar_view(advected), cache.input_mass.data(), cache.advected_mass.data(), simulation::scalar_view(output_adjoint), workspace.density_dot.data(), simulation::scalar_view(input_adjoint), simulation::scalar_view(advected_adjoint));
    }
} // namespace physica::fluids::gas::operators
