module;

#include "../detail/cuda/interop.h"
#include "fixed-position-kernels.h"
#include <physica/cuda.h>

module physica.deformables.cloth.operators.fixed_position;

import std;

namespace physica::deformables::cloth::operators {
    FixedPositionConstraint::FixedPositionConstraint(const Domain& domain, Configuration configuration) : anchor_mask(domain.allocate_scalar_field<std::uint32_t>(domain.particle_count)), anchor_positions(domain.allocate_vector_field<float>(domain.particle_count)) {
        std::vector<std::uint32_t> host_mask(domain.particle_count);
        std::vector<Vector3> host_positions = domain.configuration.rest_positions;
        for (const Anchor anchor : configuration.anchors) {
            host_mask[anchor.particle]      = 1u;
            host_positions[anchor.particle] = anchor.position;
        }
        ::cuda::copy_bytes(domain.stream, host_mask, anchor_mask.values);
        domain.upload(host_positions, anchor_positions);
    }

    FixedPositionConstraint::Cache FixedPositionConstraint::allocate_cache(const Domain&) const {
        return {};
    }

    FixedPositionConstraint::Workspace FixedPositionConstraint::allocate_workspace(const Domain&) const {
        return {};
    }

    FixedPositionConstraint::TangentWorkspace FixedPositionConstraint::allocate_tangent_workspace(const Domain&) const {
        return {};
    }

    FixedPositionConstraint::AdjointWorkspace FixedPositionConstraint::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void FixedPositionConstraint::forward(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, VectorField<float>& constrained_positions, VectorField<float>& constrained_velocities, Cache&, Workspace&) const {
        cuda_detail::fixed_position_forward(domain.stream, static_cast<std::uint32_t>(domain.particle_count), anchor_mask.values.data(), cuda_detail::field<float>(anchor_positions), cuda_detail::field<float>(positions), cuda_detail::field<float>(velocities), cuda_detail::field<float>(constrained_positions), cuda_detail::field<float>(constrained_velocities));
    }

    void FixedPositionConstraint::jvp(const Domain& domain, const VectorField<float>&, const VectorField<float>&, const VectorField<float>&, const VectorField<float>&, const Cache&, const VectorField<float>& position_tangent, const VectorField<float>& velocity_tangent, VectorField<float>& constrained_position_tangent, VectorField<float>& constrained_velocity_tangent, TangentWorkspace&) const {
        cuda_detail::fixed_position_jvp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), anchor_mask.values.data(), cuda_detail::field<float>(position_tangent), cuda_detail::field<float>(velocity_tangent), cuda_detail::field<float>(constrained_position_tangent), cuda_detail::field<float>(constrained_velocity_tangent));
    }

    void FixedPositionConstraint::vjp(const Domain& domain, const VectorField<float>&, const VectorField<float>&, const VectorField<float>&, const VectorField<float>&, const Cache&, const VectorField<double>& constrained_position_adjoint, const VectorField<double>& constrained_velocity_adjoint, VectorField<double>& position_adjoint, VectorField<double>& velocity_adjoint, AdjointWorkspace&) const {
        cuda_detail::fixed_position_vjp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), anchor_mask.values.data(), cuda_detail::field<double>(constrained_position_adjoint), cuda_detail::field<double>(constrained_velocity_adjoint), cuda_detail::field<double>(position_adjoint), cuda_detail::field<double>(velocity_adjoint));
    }
} // namespace physica::deformables::cloth::operators
