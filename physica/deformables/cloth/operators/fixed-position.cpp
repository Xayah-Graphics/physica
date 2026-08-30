module;

#include <simulation/field/device.cuh>
#include "fixed-position-kernels.h"
#include <physica/cuda.h>

module physica.deformables.cloth.operators.fixed_position;

import std;

namespace physica::deformables::cloth::operators {
    FixedPositionConstraint::FixedPositionConstraint(const Model& model, Configuration configuration) : anchor_mask(simulation::ScalarField<std::uint32_t>(model.stream, model.particle_count)), anchor_positions(simulation::VectorField<float>(model.stream, model.particle_count)) {
        std::vector<std::uint32_t> host_mask(model.particle_count);
        std::vector<Vector3<float>> host_positions = model.configuration.rest_positions;
        for (const Anchor anchor : configuration.anchors) {
            host_mask[anchor.particle]      = 1u;
            host_positions[anchor.particle] = anchor.position;
        }
        ::cuda::copy_bytes(model.stream, host_mask, anchor_mask.values);
        simulation::upload(model.stream, host_positions, anchor_positions);
    }

    FixedPositionConstraint::Cache FixedPositionConstraint::allocate_cache(const Model&) const {
        return {};
    }

    FixedPositionConstraint::Workspace FixedPositionConstraint::allocate_workspace(const Model&) const {
        return {};
    }

    FixedPositionConstraint::TangentWorkspace FixedPositionConstraint::allocate_tangent_workspace(const Model&) const {
        return {};
    }

    FixedPositionConstraint::AdjointWorkspace FixedPositionConstraint::allocate_adjoint_workspace(const Model&) const {
        return {};
    }

    void FixedPositionConstraint::forward(const Model& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, simulation::VectorField<float>& constrained_positions, simulation::VectorField<float>& constrained_velocities, Cache&, Workspace&) const {
        kernels::fixed_position_forward(model.stream, static_cast<std::uint32_t>(model.particle_count), anchor_mask.values.data(), simulation::view(anchor_positions), simulation::view(positions), simulation::view(velocities), simulation::view(constrained_positions), simulation::view(constrained_velocities));
    }

    void FixedPositionConstraint::jvp(const Model& model, const simulation::VectorField<float>&, const simulation::VectorField<float>&, const simulation::VectorField<float>&, const simulation::VectorField<float>&, const Cache&, const simulation::VectorField<float>& position_tangent, const simulation::VectorField<float>& velocity_tangent, simulation::VectorField<float>& constrained_position_tangent, simulation::VectorField<float>& constrained_velocity_tangent, TangentWorkspace&) const {
        kernels::fixed_position_jvp(model.stream, static_cast<std::uint32_t>(model.particle_count), anchor_mask.values.data(), simulation::view(position_tangent), simulation::view(velocity_tangent), simulation::view(constrained_position_tangent), simulation::view(constrained_velocity_tangent));
    }

    void FixedPositionConstraint::vjp(const Model& model, const simulation::VectorField<float>&, const simulation::VectorField<float>&, const simulation::VectorField<float>&, const simulation::VectorField<float>&, const Cache&, const simulation::VectorField<double>& constrained_position_adjoint, const simulation::VectorField<double>& constrained_velocity_adjoint, simulation::VectorField<double>& position_adjoint, simulation::VectorField<double>& velocity_adjoint, AdjointWorkspace&) const {
        kernels::fixed_position_vjp(model.stream, static_cast<std::uint32_t>(model.particle_count), anchor_mask.values.data(), simulation::view(constrained_position_adjoint), simulation::view(constrained_velocity_adjoint), simulation::view(position_adjoint), simulation::view(velocity_adjoint));
    }
} // namespace physica::deformables::cloth::operators
