module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

export module physica.fluids.gas.smoke.forces;

import std;
import physica.fluids.gas.smoke.domain;

export namespace physica::fluids::gas::smoke {
    struct BuoyancyVorticityForces final {
        struct Configuration final {
            bool vorticity_confinement_enabled = true;
        };

        struct VorticityCache final {
            CenteredVectorField centered_velocity;
            CenteredVectorField vorticity;
            ScalarField magnitude;
            CenteredVectorField normal;
            ScalarField normalizer;
        };

        struct VorticityAdjointCache final {
            CenteredVectorAdjointField centered_velocity;
            CenteredVectorAdjointField vorticity;
            ScalarAdjointField magnitude;
            CenteredVectorAdjointField normal;
        };

        struct Cache final {
            CenteredVectorField force;
            VorticityCache vorticity;
        };

        struct Differentiation final {
            CenteredVectorField force_tangent;
            VorticityCache vorticity_tangent;
            VorticityAdjointCache vorticity_adjoint;
            CenteredVectorAdjointField force_adjoint;
        };

        struct Parameters final {
            ::cuda::device_buffer<float> ambient_temperature;
            ::cuda::device_buffer<float> density_buoyancy;
            ::cuda::device_buffer<float> temperature_buoyancy;
            ::cuda::device_buffer<float> vorticity_confinement;
        };

        struct ParameterTangent final {
            ::cuda::device_buffer<float> ambient_temperature;
            ::cuda::device_buffer<float> density_buoyancy;
            ::cuda::device_buffer<float> temperature_buoyancy;
            ::cuda::device_buffer<float> vorticity_confinement;
        };

        struct ParameterAdjoint final {
            ::cuda::device_buffer<double> ambient_temperature;
            ::cuda::device_buffer<double> density_buoyancy;
            ::cuda::device_buffer<double> temperature_buoyancy;
            ::cuda::device_buffer<double> vorticity_confinement;
        };

        const Configuration configuration;
        std::optional<Differentiation> differentiation;

        BuoyancyVorticityForces(const Domain& domain, Configuration configuration, ExecutionMode mode);

        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;

        void forward(const Domain& domain, const ScalarField& density, const ScalarField& temperature, const StaggeredVectorField& velocity, const CenteredVectorField& external_acceleration, const Parameters& parameters, Cache& cache) const;
        void jvp(const Domain& domain, const ScalarField& density, const ScalarField& temperature, const ScalarField& density_tangent, const ScalarField& temperature_tangent, const StaggeredVectorField& velocity_tangent, const CenteredVectorField& external_acceleration_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Cache& cache);
        void vjp(const Domain& domain, const ScalarField& density, const ScalarField& temperature, const Parameters& parameters, const Cache& cache, StaggeredVectorAdjointField& velocity_adjoint, ScalarAdjointField& density_adjoint, ScalarAdjointField& temperature_adjoint, CenteredVectorAdjointField& external_acceleration_adjoint, ParameterAdjoint& parameter_adjoint);
    };

    template<class Algorithm>
    concept ForceAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration, ExecutionMode> && requires(Algorithm algorithm, const Domain& domain, typename Algorithm::Cache& cache, const typename Algorithm::Cache& const_cache, const ScalarField& scalar, const StaggeredVectorField& staggered, const CenteredVectorField& centered, const typename Algorithm::Parameters& parameters, const typename Algorithm::ParameterTangent& parameter_tangent, typename Algorithm::ParameterAdjoint& parameter_adjoint, ScalarAdjointField& scalar_adjoint, StaggeredVectorAdjointField& staggered_adjoint, CenteredVectorAdjointField& centered_adjoint) {
        { algorithm.allocate_parameters(domain) } -> std::same_as<typename Algorithm::Parameters>;
        { algorithm.allocate_parameter_tangent(domain) } -> std::same_as<typename Algorithm::ParameterTangent>;
        { algorithm.allocate_parameter_adjoint(domain) } -> std::same_as<typename Algorithm::ParameterAdjoint>;
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        algorithm.forward(domain, scalar, scalar, staggered, centered, parameters, cache);
        algorithm.jvp(domain, scalar, scalar, scalar, scalar, staggered, centered, parameters, parameter_tangent, const_cache);
        algorithm.vjp(domain, scalar, scalar, parameters, const_cache, staggered_adjoint, scalar_adjoint, scalar_adjoint, centered_adjoint, parameter_adjoint);
    };
} // namespace physica::fluids::gas::smoke
