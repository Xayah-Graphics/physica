module;

#include "../domain/interop.h"
#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>

module physica.deformables.cloth.constraints;

import std;

namespace physica::deformables::cloth {
    FixedPositionConstraints::FixedPositionConstraints(const Domain& domain) : anchor_mask(domain.allocate_index_field(domain.particle_count)), anchor_positions(domain.allocate_vector_field()) {
        std::vector<std::uint32_t> host_mask(domain.particle_count);
        std::vector<Vector3> host_positions(domain.particle_count);
        for (std::size_t particle = 0uz; particle < domain.particle_count; ++particle) {
            host_mask[particle] = domain.configuration.anchors[particle].has_value() ? 1u : 0u;
            host_positions[particle] = domain.configuration.anchors[particle].value_or(domain.configuration.rest_positions[particle]);
        }
        ::cuda::copy_bytes(domain.stream, host_mask, anchor_mask.values);
        domain.upload(host_positions, anchor_positions);
    }

    void FixedPositionConstraints::forward(const Domain& domain, const State& state, State& constrained_state) const {
        cuda_detail::fixed_position_forward(domain.stream, static_cast<std::uint32_t>(domain.particle_count), anchor_mask.values.data(), cuda_detail::field(anchor_positions), cuda_detail::field(state.positions), cuda_detail::field(state.velocities), cuda_detail::field(constrained_state.positions), cuda_detail::field(constrained_state.velocities));
    }

    void FixedPositionConstraints::jvp(const Domain& domain, const StateTangent& state_tangent, StateTangent& constrained_state_tangent) const {
        cuda_detail::fixed_position_jvp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), anchor_mask.values.data(), cuda_detail::field(state_tangent.positions), cuda_detail::field(state_tangent.velocities), cuda_detail::field(constrained_state_tangent.positions), cuda_detail::field(constrained_state_tangent.velocities));
    }

    void FixedPositionConstraints::vjp(const Domain& domain, const StateAdjoint& constrained_state_adjoint, StateAdjoint& state_adjoint) const {
        cuda_detail::fixed_position_vjp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), anchor_mask.values.data(), cuda_detail::adjoint_field(constrained_state_adjoint.positions), cuda_detail::adjoint_field(constrained_state_adjoint.velocities), cuda_detail::adjoint_field(state_adjoint.positions), cuda_detail::adjoint_field(state_adjoint.velocities));
    }
} // namespace physica::deformables::cloth
