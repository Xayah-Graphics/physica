module;

#include <physica/cuda.h>

export module physica.fluids.gas.smoke;

import std;
export import physica.fluids.gas.smoke.domain;
export import physica.fluids.gas.smoke.transport;
export import physica.fluids.gas.smoke.forces;
export import physica.fluids.gas.smoke.projection;

export namespace physica::fluids::gas::smoke {
    struct State final {
        ScalarField density;
        ScalarField temperature;
        StaggeredVectorField velocity;
    };

    struct Control final {
        ScalarField density_source;
        ScalarField temperature_source;
        CenteredVectorField external_acceleration;
    };

    struct StateTangent final {
        ScalarField density;
        ScalarField temperature;
        StaggeredVectorField velocity;
    };

    struct ControlTangent final {
        ScalarField density_source;
        ScalarField temperature_source;
        CenteredVectorField external_acceleration;
    };

    struct StateAdjoint final {
        ScalarAdjointField density;
        ScalarAdjointField temperature;
        StaggeredVectorAdjointField velocity;
    };

    struct ControlAdjoint final {
        ScalarAdjointField density_source;
        ScalarAdjointField temperature_source;
        CenteredVectorAdjointField external_acceleration;
    };

    template<TransportAlgorithm Transport, ForceAlgorithm Force, ProjectionAlgorithm Projection>
    struct Solver final {
        struct StepCache final {
            typename Transport::Cache transport;
            typename Force::Cache force;
        };

        Solver(DomainConfiguration domain_configuration, typename Transport::Configuration transport_configuration, typename Force::Configuration force_configuration, typename Projection::Configuration projection_configuration, const ExecutionMode mode, const ::cuda::stream_ref stream)
            : domain(std::move(domain_configuration), stream), transport(domain, std::move(transport_configuration), mode), force(domain, std::move(force_configuration), mode), projection(domain, std::move(projection_configuration), mode) {}

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State allocate_state() const {
            State value{.density = domain.allocate_scalar_field(), .temperature = domain.allocate_scalar_field(), .velocity = domain.allocate_staggered_vector_field()};
            domain.clear(value.density);
            domain.clear(value.temperature);
            domain.clear(value.velocity);
            return value;
        }

        [[nodiscard]] Control allocate_control() const {
            Control value{.density_source = domain.allocate_scalar_field(), .temperature_source = domain.allocate_scalar_field(), .external_acceleration = domain.allocate_centered_vector_field()};
            domain.clear(value.density_source);
            domain.clear(value.temperature_source);
            domain.clear(value.external_acceleration);
            return value;
        }

        [[nodiscard]] typename Force::Parameters allocate_parameters() const {
            return force.allocate_parameters(domain);
        }

        [[nodiscard]] StepCache allocate_step_cache() const {
            return {.transport = transport.allocate_cache(domain), .force = force.allocate_cache(domain)};
        }

        [[nodiscard]] StateTangent allocate_state_tangent() const {
            StateTangent value{.density = domain.allocate_scalar_field(), .temperature = domain.allocate_scalar_field(), .velocity = domain.allocate_staggered_vector_field()};
            domain.clear(value.density);
            domain.clear(value.temperature);
            domain.clear(value.velocity);
            return value;
        }

        [[nodiscard]] ControlTangent allocate_control_tangent() const {
            ControlTangent value{.density_source = domain.allocate_scalar_field(), .temperature_source = domain.allocate_scalar_field(), .external_acceleration = domain.allocate_centered_vector_field()};
            domain.clear(value.density_source);
            domain.clear(value.temperature_source);
            domain.clear(value.external_acceleration);
            return value;
        }

        [[nodiscard]] typename Force::ParameterTangent allocate_parameter_tangent() const {
            return force.allocate_parameter_tangent(domain);
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint() const {
            StateAdjoint value{.density = domain.allocate_scalar_adjoint_field(), .temperature = domain.allocate_scalar_adjoint_field(), .velocity = domain.allocate_staggered_vector_adjoint_field()};
            domain.clear(value.density);
            domain.clear(value.temperature);
            domain.clear(value.velocity);
            return value;
        }

        [[nodiscard]] ControlAdjoint allocate_control_adjoint() const {
            ControlAdjoint value{.density_source = domain.allocate_scalar_adjoint_field(), .temperature_source = domain.allocate_scalar_adjoint_field(), .external_acceleration = domain.allocate_centered_vector_adjoint_field()};
            domain.clear(value.density_source);
            domain.clear(value.temperature_source);
            domain.clear(value.external_acceleration);
            return value;
        }

        [[nodiscard]] typename Force::ParameterAdjoint allocate_parameter_adjoint() const {
            return force.allocate_parameter_adjoint(domain);
        }

        void copy_state(const State& source, State& destination) const {
            domain.copy(source.density, destination.density);
            domain.copy(source.temperature, destination.temperature);
            domain.copy(source.velocity, destination.velocity);
        }

        void copy_state_tangent(const StateTangent& source, StateTangent& destination) const {
            domain.copy(source.density, destination.density);
            domain.copy(source.temperature, destination.temperature);
            domain.copy(source.velocity, destination.velocity);
        }

        void copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const {
            domain.copy(source.density, destination.density);
            domain.copy(source.temperature, destination.temperature);
            domain.copy(source.velocity, destination.velocity);
        }

        void accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const {
            domain.accumulate(source.density, destination.density);
            domain.accumulate(source.temperature, destination.temperature);
            domain.accumulate(source.velocity, destination.velocity);
        }

        void forward_step(const State& state, const Control& control, const typename Force::Parameters& parameters, State& next_state, StepCache& cache) {
            transport.source_forward(domain, state.density, state.temperature, control.density_source, control.temperature_source, cache.transport);
            force.forward(domain, cache.transport.sourced_density, cache.transport.sourced_temperature, state.velocity, control.external_acceleration, parameters, cache.force);
            transport.velocity_forward(domain, state.velocity, cache.force.force, cache.transport);
            projection.forward(domain, cache.transport.advected_velocity, next_state.velocity);
            transport.scalar_forward(domain, cache.transport, next_state.velocity, next_state.density, next_state.temperature);
        }

        void jvp_step(const State&, const Control&, const typename Force::Parameters& parameters, const State& next_state, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const typename Force::ParameterTangent& parameter_tangent, StateTangent& next_state_tangent) {
            transport.source_jvp(domain, state_tangent.density, state_tangent.temperature, control_tangent.density_source, control_tangent.temperature_source);
            typename Transport::Differentiation& transport_workspace = *transport.differentiation;
            force.jvp(domain, cache.transport.sourced_density, cache.transport.sourced_temperature, transport_workspace.sourced_density_tangent, transport_workspace.sourced_temperature_tangent, state_tangent.velocity, control_tangent.external_acceleration, parameters, parameter_tangent, cache.force);
            typename Force::Differentiation& force_workspace = *force.differentiation;
            transport.velocity_jvp(domain, cache.transport, state_tangent.velocity, force_workspace.force_tangent);
            projection.jvp(domain, transport_workspace.advected_velocity_tangent, next_state_tangent.velocity);
            transport.scalar_jvp(domain, cache.transport, next_state.velocity, next_state_tangent.velocity, next_state_tangent.density, next_state_tangent.temperature);
        }

        void vjp_step(const State&, const Control&, const typename Force::Parameters& parameters, const State& next_state, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, typename Force::ParameterAdjoint& parameter_adjoint) {
            transport.scalar_vjp(domain, cache.transport, next_state.velocity, next_state_adjoint.density, next_state_adjoint.temperature, next_state_adjoint.velocity);
            typename Transport::Differentiation& transport_workspace = *transport.differentiation;
            projection.vjp(domain, transport_workspace.projected_velocity_adjoint, transport_workspace.advected_velocity_adjoint);
            typename Force::Differentiation& force_workspace = *force.differentiation;
            transport.velocity_vjp(domain, cache.transport, previous_state_adjoint.velocity, force_workspace.force_adjoint);
            force.vjp(domain, cache.transport.sourced_density, cache.transport.sourced_temperature, parameters, cache.force, previous_state_adjoint.velocity, transport_workspace.sourced_density_adjoint, transport_workspace.sourced_temperature_adjoint, control_adjoint.external_acceleration, parameter_adjoint);
            transport.source_vjp(domain, previous_state_adjoint.density, previous_state_adjoint.temperature, control_adjoint.density_source, control_adjoint.temperature_source);
        }

    private:
        Domain domain;
        Transport transport;
        Force force;
        Projection projection;
    };

    template struct Solver<SemiLagrangianRK2, BuoyancyVorticityForces, MacProjection<RedBlackGaussSeidel>>;
} // namespace physica::fluids::gas::smoke
