module;

#include "../detail/cuda/interop.h"
#include "semi-implicit-euler-kernels.h"
#include <physica/cuda.h>

module physica.deformables.cloth.operators.semi_implicit_euler;

import std;

namespace physica::deformables::cloth::operators {
    SemiImplicitEuler::SemiImplicitEuler(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    SemiImplicitEuler::Cache SemiImplicitEuler::allocate_cache(const Domain&) const {
        return {};
    }

    SemiImplicitEuler::Workspace SemiImplicitEuler::allocate_workspace(const Domain&) const {
        return {};
    }

    SemiImplicitEuler::TangentWorkspace SemiImplicitEuler::allocate_tangent_workspace(const Domain&) const {
        return {};
    }

    SemiImplicitEuler::AdjointWorkspace SemiImplicitEuler::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void SemiImplicitEuler::forward(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const ScalarField<float>& masses, const VectorField<float>& forces, VectorField<float>& integrated_positions, VectorField<float>& integrated_velocities, Cache&, Workspace&) const {
        cuda_detail::semi_implicit_euler_forward(domain.stream, static_cast<std::uint32_t>(domain.particle_count), configuration.time_step, cuda_detail::field<float>(positions), cuda_detail::field<float>(velocities), cuda_detail::field<float>(forces), masses.values.data(), cuda_detail::field<float>(integrated_positions), cuda_detail::field<float>(integrated_velocities));
    }

    void SemiImplicitEuler::jvp(const Domain& domain, const ScalarField<float>& masses, const VectorField<float>& forces, const Cache&, const VectorField<float>& position_tangent, const VectorField<float>& velocity_tangent, const ScalarField<float>& mass_tangent, const VectorField<float>& force_tangent, VectorField<float>& integrated_position_tangent, VectorField<float>& integrated_velocity_tangent, TangentWorkspace&) const {
        cuda_detail::semi_implicit_euler_jvp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), configuration.time_step, cuda_detail::field<float>(forces), masses.values.data(), cuda_detail::field<float>(position_tangent), cuda_detail::field<float>(velocity_tangent), cuda_detail::field<float>(force_tangent), mass_tangent.values.data(), cuda_detail::field<float>(integrated_position_tangent), cuda_detail::field<float>(integrated_velocity_tangent));
    }

    void SemiImplicitEuler::vjp(const Domain& domain, const ScalarField<float>& masses, const VectorField<float>& forces, const Cache&, const VectorField<double>& integrated_position_adjoint, const VectorField<double>& integrated_velocity_adjoint, VectorField<double>& position_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& force_adjoint, ScalarField<double>& mass_adjoint, AdjointWorkspace&) const {
        cuda_detail::semi_implicit_euler_vjp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), configuration.time_step, cuda_detail::field<float>(forces), masses.values.data(), cuda_detail::field<double>(integrated_position_adjoint), cuda_detail::field<double>(integrated_velocity_adjoint), cuda_detail::field<double>(position_adjoint), cuda_detail::field<double>(velocity_adjoint), cuda_detail::field<double>(force_adjoint), mass_adjoint.values.data());
    }
} // namespace physica::deformables::cloth::operators
