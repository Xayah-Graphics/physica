module;

#include "../detail/cuda/interop.h"
#include "conservation-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.conservation;

import std;

namespace physica::fluids::gas::operators {
    MassConservation::MassConservation(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    MassConservation::Cache MassConservation::allocate_cache(const Domain& domain) const {
        return {
            .input_mass    = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .advected_mass = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
        };
    }

    MassConservation::TangentWorkspace MassConservation::allocate_tangent_workspace(const Domain& domain) const {
        return {
            .input_mass    = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
            .advected_mass = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init},
        };
    }

    MassConservation::AdjointWorkspace MassConservation::allocate_adjoint_workspace(const Domain& domain) const {
        return {.density_dot = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init}};
    }

    void MassConservation::forward(const Domain& domain, const CellField<float>& input, const CellField<float>& advected, CellField<float>& output, Cache& cache) const {
        const float retention = std::exp(-configuration.dissipation * domain.configuration.time_step);
        cuda_backend::mass_forward(domain.stream, detail::cuda::grid(domain.configuration), retention, detail::cuda::scalar(input), detail::cuda::scalar(advected), cache.input_mass.data(), cache.advected_mass.data(), detail::cuda::scalar(output));
    }

    void MassConservation::jvp(const Domain& domain, const CellField<float>& input, const CellField<float>& advected, const CellField<float>& input_tangent, const CellField<float>& advected_tangent, const Cache& cache, CellField<float>& output_tangent, TangentWorkspace& workspace) const {
        const float retention = std::exp(-configuration.dissipation * domain.configuration.time_step);
        cuda_backend::mass_jvp(domain.stream, detail::cuda::grid(domain.configuration), retention, detail::cuda::scalar(input), detail::cuda::scalar(advected), detail::cuda::scalar(input_tangent), detail::cuda::scalar(advected_tangent), cache.input_mass.data(), cache.advected_mass.data(), workspace.input_mass.data(), workspace.advected_mass.data(), detail::cuda::scalar(output_tangent));
    }

    void MassConservation::vjp(const Domain& domain, const CellField<float>& advected, const Cache& cache, const CellField<double>& output_adjoint, CellField<double>& input_adjoint, CellField<double>& advected_adjoint, AdjointWorkspace& workspace) const {
        const float retention = std::exp(-configuration.dissipation * domain.configuration.time_step);
        cuda_backend::mass_vjp(domain.stream, detail::cuda::grid(domain.configuration), retention, detail::cuda::scalar(advected), cache.input_mass.data(), cache.advected_mass.data(), detail::cuda::scalar_adjoint(output_adjoint), workspace.density_dot.data(), detail::cuda::scalar_adjoint(input_adjoint), detail::cuda::scalar_adjoint(advected_adjoint));
    }
} // namespace physica::fluids::gas::operators
