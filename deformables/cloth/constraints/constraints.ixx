module;

#include <physica/cuda.h>

export module physica.deformables.cloth.constraints;

import std;
import physica.deformables.cloth.domain;

export namespace physica::deformables::cloth {
    struct FixedPositionConstraints final {
        IndexField anchor_mask;
        VectorField anchor_positions;

        explicit FixedPositionConstraints(const Domain& domain);

        void forward(const Domain& domain, const State& state, State& constrained_state) const;
        void jvp(const Domain& domain, const StateTangent& state_tangent, StateTangent& constrained_state_tangent) const;
        void vjp(const Domain& domain, const StateAdjoint& constrained_state_adjoint, StateAdjoint& state_adjoint) const;
    };

    template<class Algorithm>
    concept ConstraintAlgorithm = std::constructible_from<Algorithm, const Domain&> && requires(const Algorithm algorithm, const Domain& domain, const State& state, State& state_output, const StateTangent& state_tangent, StateTangent& tangent_output, const StateAdjoint& state_adjoint, StateAdjoint& adjoint_output) {
        algorithm.forward(domain, state, state_output);
        algorithm.jvp(domain, state_tangent, tangent_output);
        algorithm.vjp(domain, state_adjoint, adjoint_output);
    };
} // namespace physica::deformables::cloth
