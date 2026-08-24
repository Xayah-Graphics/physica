module;

#include "../domain/interop.h"
#include "kernels.h"
#include <physica/cuda.h>

module physica.deformables.cloth.integration;

import std;

namespace physica::deformables::cloth {
    SemiImplicitEuler::SemiImplicitEuler(const Domain& domain, Configuration next_configuration, const ExecutionMode mode) : configuration(std::move(next_configuration)), differentiation{} {
        if (mode == ExecutionMode::differentiable)
            differentiation.emplace(Differentiation{
                .tangent = {.positions = domain.allocate_vector_field(), .velocities = domain.allocate_vector_field()},
                .adjoint = {.positions = domain.allocate_vector_adjoint_field(), .velocities = domain.allocate_vector_adjoint_field()},
            });
    }

    SemiImplicitEuler::Cache SemiImplicitEuler::allocate_cache(const Domain& domain) const {
        return {.state = {.positions = domain.allocate_vector_field(), .velocities = domain.allocate_vector_field()}};
    }

    void SemiImplicitEuler::forward(const Domain& domain, const State& state, const ScalarField& masses, const VectorField& forces, Cache& cache) const {
        cuda_detail::semi_implicit_euler_forward(domain.stream, static_cast<std::uint32_t>(domain.particle_count), configuration.time_step, cuda_detail::field(state.positions), cuda_detail::field(state.velocities), cuda_detail::field(forces), masses.values.data(), cuda_detail::field(cache.state.positions), cuda_detail::field(cache.state.velocities));
    }

    void SemiImplicitEuler::jvp(const Domain& domain, const ScalarField& masses, const VectorField& forces, const StateTangent& state_tangent, const ScalarField& mass_tangent, const VectorField& force_tangent) {
        Differentiation& workspace = *differentiation;
        cuda_detail::semi_implicit_euler_jvp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), configuration.time_step, cuda_detail::field(forces), masses.values.data(), cuda_detail::field(state_tangent.positions), cuda_detail::field(state_tangent.velocities), cuda_detail::field(force_tangent), mass_tangent.values.data(), cuda_detail::field(workspace.tangent.positions), cuda_detail::field(workspace.tangent.velocities));
    }

    void SemiImplicitEuler::vjp(const Domain& domain, const ScalarField& masses, const VectorField& forces, const StateAdjoint& integrated_state_adjoint, StateAdjoint& state_adjoint, VectorAdjointField& force_adjoint, ScalarAdjointField& mass_adjoint) const {
        cuda_detail::semi_implicit_euler_vjp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), configuration.time_step, cuda_detail::field(forces), masses.values.data(), cuda_detail::adjoint_field(integrated_state_adjoint.positions), cuda_detail::adjoint_field(integrated_state_adjoint.velocities), cuda_detail::adjoint_field(state_adjoint.positions), cuda_detail::adjoint_field(state_adjoint.velocities), cuda_detail::adjoint_field(force_adjoint), mass_adjoint.values.data());
    }
} // namespace physica::deformables::cloth
