module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

export module physica.fluids.gas.smoke.transport;

import std;
import physica.fluids.gas.smoke.domain;

export namespace physica::fluids::gas::smoke {
    struct SemiLagrangianRK2 final {
        struct Configuration final {};

        struct Cache final {
            ScalarField sourced_density;
            ScalarField sourced_temperature;
            StaggeredVectorField forced_velocity;
            StaggeredVectorField advected_velocity;
        };

        struct Differentiation final {
            ScalarField sourced_density_tangent;
            ScalarField sourced_temperature_tangent;
            StaggeredVectorField forced_velocity_tangent;
            StaggeredVectorField raw_advected_velocity_tangent;
            StaggeredVectorField advected_velocity_tangent;
            ScalarAdjointField sourced_density_adjoint;
            ScalarAdjointField sourced_temperature_adjoint;
            StaggeredVectorAdjointField projected_velocity_adjoint;
            StaggeredVectorAdjointField advected_velocity_adjoint;
            StaggeredVectorAdjointField raw_advected_velocity_adjoint;
            StaggeredVectorAdjointField forced_velocity_adjoint;
        };

        StaggeredVectorField raw_advected_velocity;
        std::optional<Differentiation> differentiation;

        SemiLagrangianRK2(const Domain& domain, Configuration configuration, ExecutionMode mode);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;

        void source_forward(const Domain& domain, const ScalarField& density, const ScalarField& temperature, const ScalarField& density_source, const ScalarField& temperature_source, Cache& cache) const;
        void velocity_forward(const Domain& domain, const StaggeredVectorField& velocity, const CenteredVectorField& force, Cache& cache);
        void scalar_forward(const Domain& domain, const Cache& cache, const StaggeredVectorField& projected_velocity, ScalarField& density, ScalarField& temperature) const;

        void source_jvp(const Domain& domain, const ScalarField& density_tangent, const ScalarField& temperature_tangent, const ScalarField& density_source_tangent, const ScalarField& temperature_source_tangent);
        void velocity_jvp(const Domain& domain, const Cache& cache, const StaggeredVectorField& velocity_tangent, const CenteredVectorField& force_tangent);
        void scalar_jvp(const Domain& domain, const Cache& cache, const StaggeredVectorField& projected_velocity, const StaggeredVectorField& projected_velocity_tangent, ScalarField& density_tangent, ScalarField& temperature_tangent) const;

        void scalar_vjp(const Domain& domain, const Cache& cache, const StaggeredVectorField& projected_velocity, const ScalarAdjointField& density_adjoint, const ScalarAdjointField& temperature_adjoint, const StaggeredVectorAdjointField& velocity_adjoint);
        void velocity_vjp(const Domain& domain, const Cache& cache, StaggeredVectorAdjointField& velocity_adjoint, CenteredVectorAdjointField& force_adjoint);
        void source_vjp(const Domain& domain, ScalarAdjointField& density_adjoint, ScalarAdjointField& temperature_adjoint, ScalarAdjointField& density_source_adjoint, ScalarAdjointField& temperature_source_adjoint) const;
    };

    template<class Algorithm>
    concept TransportAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration, ExecutionMode> && requires(Algorithm algorithm, const Domain& domain, typename Algorithm::Cache& cache, const typename Algorithm::Cache& const_cache, const ScalarField& scalar, ScalarField& scalar_output, const StaggeredVectorField& staggered, const CenteredVectorField& centered, CenteredVectorAdjointField& centered_adjoint, const ScalarAdjointField& scalar_adjoint, ScalarAdjointField& scalar_adjoint_output, const StaggeredVectorAdjointField& staggered_adjoint, StaggeredVectorAdjointField& staggered_adjoint_output) {
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        algorithm.source_forward(domain, scalar, scalar, scalar, scalar, cache);
        algorithm.velocity_forward(domain, staggered, centered, cache);
        algorithm.scalar_forward(domain, const_cache, staggered, scalar_output, scalar_output);
        algorithm.source_jvp(domain, scalar, scalar, scalar, scalar);
        algorithm.velocity_jvp(domain, const_cache, staggered, centered);
        algorithm.scalar_jvp(domain, const_cache, staggered, staggered, scalar_output, scalar_output);
        algorithm.scalar_vjp(domain, const_cache, staggered, scalar_adjoint, scalar_adjoint, staggered_adjoint);
        algorithm.velocity_vjp(domain, const_cache, staggered_adjoint_output, centered_adjoint);
        algorithm.source_vjp(domain, scalar_adjoint_output, scalar_adjoint_output, scalar_adjoint_output, scalar_adjoint_output);
    };
} // namespace physica::fluids::gas::smoke
