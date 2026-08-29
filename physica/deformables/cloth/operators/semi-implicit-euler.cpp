module;

#include <physica/field/device.cuh>
#include "semi-implicit-euler-kernels.h"
#include <physica/cuda.h>

module physica.deformables.cloth.operators.semi_implicit_euler;

import std;

namespace physica::deformables::cloth::operators {
    SemiImplicitEuler::SemiImplicitEuler(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    SemiImplicitEuler::Cache SemiImplicitEuler::allocate_cache(const Model&) const {
        return {};
    }

    SemiImplicitEuler::Workspace SemiImplicitEuler::allocate_workspace(const Model&) const {
        return {};
    }

    SemiImplicitEuler::TangentWorkspace SemiImplicitEuler::allocate_tangent_workspace(const Model&) const {
        return {};
    }

    SemiImplicitEuler::AdjointWorkspace SemiImplicitEuler::allocate_adjoint_workspace(const Model&) const {
        return {};
    }

    void SemiImplicitEuler::forward(const Model& model, const VectorField<float>& positions, const VectorField<float>& velocities, const ScalarField<float>& masses, const VectorField<float>& forces, VectorField<float>& integrated_positions, VectorField<float>& integrated_velocities, Cache&, Workspace&) const {
        kernels::semi_implicit_euler_forward(model.fields.stream, static_cast<std::uint32_t>(model.particle_count), configuration.time_step, field::view(positions), field::view(velocities), field::view(forces), masses.values.data(), field::view(integrated_positions), field::view(integrated_velocities));
    }

    void SemiImplicitEuler::jvp(const Model& model, const ScalarField<float>& masses, const VectorField<float>& forces, const Cache&, const VectorField<float>& position_tangent, const VectorField<float>& velocity_tangent, const ScalarField<float>& mass_tangent, const VectorField<float>& force_tangent, VectorField<float>& integrated_position_tangent, VectorField<float>& integrated_velocity_tangent, TangentWorkspace&) const {
        kernels::semi_implicit_euler_jvp(model.fields.stream, static_cast<std::uint32_t>(model.particle_count), configuration.time_step, field::view(forces), masses.values.data(), field::view(position_tangent), field::view(velocity_tangent), field::view(force_tangent), mass_tangent.values.data(), field::view(integrated_position_tangent), field::view(integrated_velocity_tangent));
    }

    void SemiImplicitEuler::vjp(const Model& model, const ScalarField<float>& masses, const VectorField<float>& forces, const Cache&, const VectorField<double>& integrated_position_adjoint, const VectorField<double>& integrated_velocity_adjoint, VectorField<double>& position_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& force_adjoint, ScalarField<double>& mass_adjoint, AdjointWorkspace&) const {
        kernels::semi_implicit_euler_vjp(model.fields.stream, static_cast<std::uint32_t>(model.particle_count), configuration.time_step, field::view(forces), masses.values.data(), field::view(integrated_position_adjoint), field::view(integrated_velocity_adjoint), field::view(position_adjoint), field::view(velocity_adjoint), field::view(force_adjoint), mass_adjoint.values.data());
    }
} // namespace physica::deformables::cloth::operators
