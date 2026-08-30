module;

#include "semi-implicit-euler-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

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

    void SemiImplicitEuler::forward(const Model& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::ScalarField<float>& masses, const simulation::VectorField<float>& forces, simulation::VectorField<float>& integrated_positions, simulation::VectorField<float>& integrated_velocities, Cache&, Workspace&) const {
        kernels::semi_implicit_euler_forward(model.stream, static_cast<std::uint32_t>(model.particle_count), configuration.time_step, simulation::view(positions), simulation::view(velocities), simulation::view(forces), masses.values.data(), simulation::view(integrated_positions), simulation::view(integrated_velocities));
    }

    void SemiImplicitEuler::jvp(const Model& model, const simulation::ScalarField<float>& masses, const simulation::VectorField<float>& forces, const Cache&, const simulation::VectorField<float>& position_tangent, const simulation::VectorField<float>& velocity_tangent, const simulation::ScalarField<float>& mass_tangent, const simulation::VectorField<float>& force_tangent, simulation::VectorField<float>& integrated_position_tangent, simulation::VectorField<float>& integrated_velocity_tangent, TangentWorkspace&) const {
        kernels::semi_implicit_euler_jvp(model.stream, static_cast<std::uint32_t>(model.particle_count), configuration.time_step, simulation::view(forces), masses.values.data(), simulation::view(position_tangent), simulation::view(velocity_tangent), simulation::view(force_tangent), mass_tangent.values.data(), simulation::view(integrated_position_tangent), simulation::view(integrated_velocity_tangent));
    }

    void SemiImplicitEuler::vjp(const Model& model, const simulation::ScalarField<float>& masses, const simulation::VectorField<float>& forces, const Cache&, const simulation::VectorField<double>& integrated_position_adjoint, const simulation::VectorField<double>& integrated_velocity_adjoint, simulation::VectorField<double>& position_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& force_adjoint, simulation::ScalarField<double>& mass_adjoint, AdjointWorkspace&) const {
        kernels::semi_implicit_euler_vjp(model.stream, static_cast<std::uint32_t>(model.particle_count), configuration.time_step, simulation::view(forces), masses.values.data(), simulation::view(integrated_position_adjoint), simulation::view(integrated_velocity_adjoint), simulation::view(position_adjoint), simulation::view(velocity_adjoint), simulation::view(force_adjoint), mass_adjoint.values.data());
    }
} // namespace physica::deformables::cloth::operators
