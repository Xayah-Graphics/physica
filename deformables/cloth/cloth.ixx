module;

#include <physica/cuda.h>

export module physica.deformables.cloth;

import std;
export import physica.deformables.cloth.constraints;
export import physica.deformables.cloth.domain;
export import physica.deformables.cloth.forces;
export import physica.deformables.cloth.integration;

export namespace physica::deformables::cloth {
    template<ForceAlgorithm Force, IntegrationAlgorithm Integration, ConstraintAlgorithm Constraint>
    struct Solver final {
        struct Parameters final {
            ScalarField masses;
            typename Force::Parameters forces;
        };

        struct ParameterTangent final {
            ScalarField masses;
            typename Force::ParameterTangent forces;
        };

        struct ParameterAdjoint final {
            ScalarAdjointField masses;
            typename Force::ParameterAdjoint forces;
        };

        struct StepCache final {
            typename Force::Cache forces;
            typename Integration::Cache integration;
        };

        Solver(DomainConfiguration domain_configuration, typename Force::Configuration force_configuration, typename Integration::Configuration integration_configuration, const ExecutionMode mode, const ::cuda::stream_ref stream)
            : domain(std::move(domain_configuration), stream), force(domain, std::move(force_configuration), mode), integration(domain, std::move(integration_configuration), mode), constraint(domain) {}

        Solver(const Solver&) = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&) = delete;
        Solver& operator=(Solver&&) = delete;

        [[nodiscard]] State allocate_state() const {
            State result{.positions = domain.allocate_vector_field(), .velocities = domain.allocate_vector_field()};
            domain.clear(result.positions);
            domain.clear(result.velocities);
            return result;
        }

        [[nodiscard]] Control allocate_control() const {
            Control result{.external_forces = domain.allocate_vector_field()};
            domain.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] Parameters allocate_parameters() const {
            Parameters result{.masses = domain.allocate_scalar_field(domain.particle_count), .forces = force.allocate_parameters(domain)};
            domain.clear(result.masses);
            return result;
        }

        [[nodiscard]] StepCache allocate_step_cache() const {
            return {.forces = force.allocate_cache(domain), .integration = integration.allocate_cache(domain)};
        }

        [[nodiscard]] StateTangent allocate_state_tangent() const {
            StateTangent result{.positions = domain.allocate_vector_field(), .velocities = domain.allocate_vector_field()};
            domain.clear(result.positions);
            domain.clear(result.velocities);
            return result;
        }

        [[nodiscard]] ControlTangent allocate_control_tangent() const {
            ControlTangent result{.external_forces = domain.allocate_vector_field()};
            domain.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] ParameterTangent allocate_parameter_tangent() const {
            ParameterTangent result{.masses = domain.allocate_scalar_field(domain.particle_count), .forces = force.allocate_parameter_tangent(domain)};
            domain.clear(result.masses);
            return result;
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint() const {
            StateAdjoint result{.positions = domain.allocate_vector_adjoint_field(), .velocities = domain.allocate_vector_adjoint_field()};
            domain.clear(result.positions);
            domain.clear(result.velocities);
            return result;
        }

        [[nodiscard]] ControlAdjoint allocate_control_adjoint() const {
            ControlAdjoint result{.external_forces = domain.allocate_vector_adjoint_field()};
            domain.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint() const {
            ParameterAdjoint result{.masses = domain.allocate_scalar_adjoint_field(domain.particle_count), .forces = force.allocate_parameter_adjoint(domain)};
            domain.clear(result.masses);
            return result;
        }

        void reset_state(State& state) const {
            domain.upload(domain.configuration.rest_positions, state.positions);
            domain.clear(state.velocities);
        }

        void copy_state(const State& source, State& destination) const {
            domain.copy(source.positions, destination.positions);
            domain.copy(source.velocities, destination.velocities);
        }

        void copy_state_tangent(const StateTangent& source, StateTangent& destination) const {
            domain.copy(source.positions, destination.positions);
            domain.copy(source.velocities, destination.velocities);
        }

        void copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const {
            domain.copy(source.positions, destination.positions);
            domain.copy(source.velocities, destination.velocities);
        }

        void accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const {
            domain.accumulate(source.positions, destination.positions);
            domain.accumulate(source.velocities, destination.velocities);
        }

        void forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache) {
            force.forward(domain, state, control, parameters.masses, parameters.forces, cache.forces);
            integration.forward(domain, state, parameters.masses, cache.forces.values, cache.integration);
            constraint.forward(domain, cache.integration.state, next_state);
        }

        void jvp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent) {
            force.jvp(domain, state, parameters.masses, parameters.forces, state_tangent, control_tangent, parameter_tangent.masses, parameter_tangent.forces, cache.forces);
            typename Force::Differentiation& force_workspace = *force.differentiation;
            integration.jvp(domain, parameters.masses, cache.forces.values, state_tangent, parameter_tangent.masses, force_workspace.tangent);
            typename Integration::Differentiation& integration_workspace = *integration.differentiation;
            constraint.jvp(domain, integration_workspace.tangent, next_state_tangent);
        }

        void vjp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint) {
            typename Force::Differentiation& force_workspace = *force.differentiation;
            typename Integration::Differentiation& integration_workspace = *integration.differentiation;
            domain.clear(force_workspace.adjoint);
            domain.clear(integration_workspace.adjoint.positions);
            domain.clear(integration_workspace.adjoint.velocities);
            constraint.vjp(domain, next_state_adjoint, integration_workspace.adjoint);
            integration.vjp(domain, parameters.masses, cache.forces.values, integration_workspace.adjoint, previous_state_adjoint, force_workspace.adjoint, parameter_adjoint.masses);
            force.vjp(domain, state, parameters.masses, parameters.forces, force_workspace.adjoint, previous_state_adjoint, control_adjoint, parameter_adjoint.masses, parameter_adjoint.forces);
        }

    private:
        Domain domain;
        Force force;
        Integration integration;
        Constraint constraint;
    };

    template struct Solver<MassSpringForces, SemiImplicitEuler, FixedPositionConstraints>;
} // namespace physica::deformables::cloth
