module;

#include <physica/field/device.cuh>
#include "fixed-position-kernels.h"
#include <physica/cuda.h>

module physica.deformables.cloth.operators.fixed_position;

import std;

namespace physica::deformables::cloth::operators {
    FixedPositionConstraint::FixedPositionConstraint(const Model& model, Configuration configuration) : anchor_mask(model.fields.allocate_scalar_field<std::uint32_t>(model.particle_count)), anchor_positions(model.fields.allocate_vector_field<float>(model.particle_count)) {
        std::vector<std::uint32_t> host_mask(model.particle_count);
        std::vector<Vector3<float>> host_positions = model.configuration.rest_positions;
        for (const Anchor anchor : configuration.anchors) {
            host_mask[anchor.particle]      = 1u;
            host_positions[anchor.particle] = anchor.position;
        }
        ::cuda::copy_bytes(model.fields.stream, host_mask, anchor_mask.values);
        model.fields.upload(host_positions, anchor_positions);
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

    void FixedPositionConstraint::forward(const Model& model, const VectorField<float>& positions, const VectorField<float>& velocities, VectorField<float>& constrained_positions, VectorField<float>& constrained_velocities, Cache&, Workspace&) const {
        kernels::fixed_position_forward(model.fields.stream, static_cast<std::uint32_t>(model.particle_count), anchor_mask.values.data(), field::view(anchor_positions), field::view(positions), field::view(velocities), field::view(constrained_positions), field::view(constrained_velocities));
    }

    void FixedPositionConstraint::jvp(const Model& model, const VectorField<float>&, const VectorField<float>&, const VectorField<float>&, const VectorField<float>&, const Cache&, const VectorField<float>& position_tangent, const VectorField<float>& velocity_tangent, VectorField<float>& constrained_position_tangent, VectorField<float>& constrained_velocity_tangent, TangentWorkspace&) const {
        kernels::fixed_position_jvp(model.fields.stream, static_cast<std::uint32_t>(model.particle_count), anchor_mask.values.data(), field::view(position_tangent), field::view(velocity_tangent), field::view(constrained_position_tangent), field::view(constrained_velocity_tangent));
    }

    void FixedPositionConstraint::vjp(const Model& model, const VectorField<float>&, const VectorField<float>&, const VectorField<float>&, const VectorField<float>&, const Cache&, const VectorField<double>& constrained_position_adjoint, const VectorField<double>& constrained_velocity_adjoint, VectorField<double>& position_adjoint, VectorField<double>& velocity_adjoint, AdjointWorkspace&) const {
        kernels::fixed_position_vjp(model.fields.stream, static_cast<std::uint32_t>(model.particle_count), anchor_mask.values.data(), field::view(constrained_position_adjoint), field::view(constrained_velocity_adjoint), field::view(position_adjoint), field::view(velocity_adjoint));
    }
} // namespace physica::deformables::cloth::operators
