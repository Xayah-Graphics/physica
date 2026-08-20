module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

export module physica.deformables.cloth.integration;

import std;
import physica.deformables.cloth.domain;

export namespace physica::deformables::cloth {
    struct SemiImplicitEuler final {
        struct Configuration final {
            float time_step;
        };

        struct Cache final {
            State state;
        };

        struct Differentiation final {
            StateTangent tangent;
            StateAdjoint adjoint;
        };

        const Configuration configuration;
        std::optional<Differentiation> differentiation;

        SemiImplicitEuler(const Domain& domain, Configuration configuration, ExecutionMode mode);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;

        void forward(const Domain& domain, const State& state, const ScalarField& masses, const VectorField& forces, Cache& cache) const;
        void jvp(const Domain& domain, const ScalarField& masses, const VectorField& forces, const StateTangent& state_tangent, const ScalarField& mass_tangent, const VectorField& force_tangent);
        void vjp(const Domain& domain, const ScalarField& masses, const VectorField& forces, const StateAdjoint& integrated_state_adjoint, StateAdjoint& state_adjoint, VectorAdjointField& force_adjoint, ScalarAdjointField& mass_adjoint) const;
    };

    template<class Algorithm>
    concept IntegrationAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration, ExecutionMode> && requires(Algorithm algorithm, const Algorithm const_algorithm, const Domain& domain, const State& state, const ScalarField& scalar, const VectorField& vector, typename Algorithm::Cache& cache, const StateTangent& state_tangent, const StateAdjoint& integrated_adjoint, StateAdjoint& state_adjoint, VectorAdjointField& vector_adjoint, ScalarAdjointField& scalar_adjoint) {
        { const_algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        const_algorithm.forward(domain, state, scalar, vector, cache);
        algorithm.jvp(domain, scalar, vector, state_tangent, scalar, vector);
        const_algorithm.vjp(domain, scalar, vector, integrated_adjoint, state_adjoint, vector_adjoint, scalar_adjoint);
    };
} // namespace physica::deformables::cloth
