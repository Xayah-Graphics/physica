module;

#include <physica/cuda.h>

export module physica.deformables.cloth.forces;

import std;
import physica.deformables.cloth.domain;

export namespace physica::deformables::cloth {
    struct Spring final {
        std::uint32_t first;
        std::uint32_t second;
        float rest_length;
    };

    struct SpringTopology final {
        std::vector<Spring> springs;
        std::vector<std::uint32_t> offsets;
        std::vector<std::uint32_t> indices;
    };

    struct MassSpringTopology final {
        SpringTopology stretch;
        SpringTopology bending;
    };

    struct DeviceSpringTopology final {
        IndexField first;
        IndexField second;
        IndexField offsets;
        IndexField indices;
    };

    struct DeviceMassSpringTopology final {
        DeviceSpringTopology stretch;
        DeviceSpringTopology bending;
    };

    struct MassSpringForces final {
        struct Configuration final {
            Vector3 gravity{};
        };

        struct SpringParameters final {
            ScalarField stiffnesses;
            ScalarField dampings;
            ScalarField rest_lengths;
        };

        struct SpringParameterAdjoint final {
            ScalarAdjointField stiffnesses;
            ScalarAdjointField dampings;
            ScalarAdjointField rest_lengths;
        };

        struct Parameters final {
            SpringParameters stretch;
            SpringParameters bending;
        };

        struct ParameterTangent final {
            SpringParameters stretch;
            SpringParameters bending;
        };

        struct ParameterAdjoint final {
            SpringParameterAdjoint stretch;
            SpringParameterAdjoint bending;
        };

        struct Cache final {
            VectorField values;
        };

        struct Differentiation final {
            VectorField tangent;
            VectorAdjointField adjoint;
        };

        const Configuration configuration;
        const MassSpringTopology topology;
        DeviceMassSpringTopology device_topology;
        std::optional<Differentiation> differentiation;

        MassSpringForces(const Domain& domain, Configuration configuration, ExecutionMode mode);

        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;

        void forward(const Domain& domain, const State& state, const Control& control, const ScalarField& masses, const Parameters& parameters, Cache& cache) const;
        void jvp(const Domain& domain, const State& state, const ScalarField& masses, const Parameters& parameters, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ScalarField& mass_tangent, const ParameterTangent& parameter_tangent, const Cache& cache);
        void vjp(const Domain& domain, const State& state, const ScalarField& masses, const Parameters& parameters, const VectorAdjointField& force_adjoint, StateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, ScalarAdjointField& mass_adjoint, ParameterAdjoint& parameter_adjoint) const;
    };

    template <class Algorithm>
    concept ForceAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration, ExecutionMode> && requires(Algorithm algorithm, const Algorithm const_algorithm, const Domain& domain, const State& state, const Control& control, const ScalarField& scalar, const typename Algorithm::Parameters& parameters, const StateTangent& state_tangent, const ControlTangent& control_tangent, const typename Algorithm::ParameterTangent& parameter_tangent, typename Algorithm::Cache& cache, const typename Algorithm::Cache& const_cache, StateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, ScalarAdjointField& scalar_adjoint, typename Algorithm::ParameterAdjoint& parameter_adjoint, const VectorAdjointField& vector_adjoint) {
        { const_algorithm.allocate_parameters(domain) } -> std::same_as<typename Algorithm::Parameters>;
        { const_algorithm.allocate_parameter_tangent(domain) } -> std::same_as<typename Algorithm::ParameterTangent>;
        { const_algorithm.allocate_parameter_adjoint(domain) } -> std::same_as<typename Algorithm::ParameterAdjoint>;
        { const_algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        const_algorithm.forward(domain, state, control, scalar, parameters, cache);
        algorithm.jvp(domain, state, scalar, parameters, state_tangent, control_tangent, scalar, parameter_tangent, const_cache);
        const_algorithm.vjp(domain, state, scalar, parameters, vector_adjoint, state_adjoint, control_adjoint, scalar_adjoint, parameter_adjoint);
    };
} // namespace physica::deformables::cloth
