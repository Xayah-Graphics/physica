module;

#include <physica/cuda.h>

export module physica.fluids.gas.solvers.smoke;

import std;
import physica.fluids.gas.domain;
import physica.fluids.gas.operators.pointwise;
import physica.fluids.gas.operators.advection;
import physica.fluids.gas.operators.diffusion;
import physica.fluids.gas.operators.projection;

export namespace physica::fluids::gas::solvers::smoke {
    struct State final {
        simulation::ScalarField<float> density;
        simulation::ScalarField<float> temperature;
        simulation::VectorField<float> velocity;
    };

    struct Control final {
        simulation::ScalarField<float> density_source;
        simulation::ScalarField<float> temperature_source;
        simulation::VectorField<float> external_acceleration;
    };

    struct StateTangent final {
        simulation::ScalarField<float> density;
        simulation::ScalarField<float> temperature;
        simulation::VectorField<float> velocity;
    };

    struct ControlTangent final {
        simulation::ScalarField<float> density_source;
        simulation::ScalarField<float> temperature_source;
        simulation::VectorField<float> external_acceleration;
    };

    struct StateAdjoint final {
        simulation::ScalarField<double> density;
        simulation::ScalarField<double> temperature;
        simulation::VectorField<double> velocity;
    };

    struct ControlAdjoint final {
        simulation::ScalarField<double> density_source;
        simulation::ScalarField<double> temperature_source;
        simulation::VectorField<double> external_acceleration;
    };

    template <class Algorithm>
    concept ForceAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace, const simulation::ScalarField<float>& scalar, const simulation::VectorField<float>& velocity, const simulation::VectorField<float>& centered, const typename Algorithm::Parameters& parameters, const typename Algorithm::ParameterTangent& parameter_tangent, typename Algorithm::ParameterAdjoint& parameter_adjoint, simulation::ScalarField<double>& scalar_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& centered_adjoint) {
        { algorithm.allocate_parameters(domain) } -> std::same_as<typename Algorithm::Parameters>;
        { algorithm.allocate_parameter_tangent(domain) } -> std::same_as<typename Algorithm::ParameterTangent>;
        { algorithm.allocate_parameter_adjoint(domain) } -> std::same_as<typename Algorithm::ParameterAdjoint>;
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, scalar, scalar, velocity, centered, parameters, cache);
        algorithm.jvp(domain, scalar, scalar, scalar, scalar, velocity, centered, parameters, parameter_tangent, constant_cache, tangent_workspace);
        algorithm.vjp(domain, scalar, scalar, parameters, constant_cache, centered_adjoint, velocity_adjoint, scalar_adjoint, scalar_adjoint, centered_adjoint, parameter_adjoint, adjoint_workspace);
    };

    template <operators::AdvectionAlgorithm Advection, operators::DiffusionAlgorithm Diffusion, ForceAlgorithm Force, operators::ProjectionAlgorithm Projection>
    struct Solver final {
        struct Configuration final {
            typename Advection::Configuration advection{};
            typename Diffusion::Configuration diffusion{};
            typename Force::Configuration force{};
            typename Projection::Configuration projection{};
            ScalarBoundary density_boundary{};
            ScalarBoundary temperature_boundary{};
            std::vector<float> collider_density{};
            std::vector<float> collider_temperature{};
        };

        struct StepCache final {
            simulation::ScalarField<float> sourced_density;
            simulation::ScalarField<float> sourced_temperature;
            typename Force::Cache force;
            simulation::VectorField<float> forced_velocity;
        };

        struct Workspace final {
            typename Advection::Workspace advection;
            simulation::VectorField<float> advected_velocity;
            typename Diffusion::Workspace diffusion;
            simulation::VectorField<float> diffused_velocity;
            typename Projection::Workspace projection;
        };

        struct TangentWorkspace final {
            simulation::ScalarField<float> sourced_density;
            simulation::ScalarField<float> sourced_temperature;
            typename Force::TangentWorkspace force;
            simulation::VectorField<float> forced_velocity;
            typename Advection::TangentWorkspace advection;
            simulation::VectorField<float> advected_velocity;
            typename Diffusion::TangentWorkspace diffusion;
            simulation::VectorField<float> diffused_velocity;
            simulation::VectorField<float> constrained_velocity;
            typename Projection::TangentWorkspace projection;
        };

        struct AdjointWorkspace final {
            simulation::ScalarField<double> sourced_density;
            simulation::ScalarField<double> sourced_temperature;
            simulation::VectorField<double> projected_velocity;
            typename Projection::AdjointWorkspace projection;
            simulation::VectorField<double> constrained_velocity;
            simulation::VectorField<double> diffused_velocity;
            typename Diffusion::AdjointWorkspace diffusion;
            simulation::VectorField<double> advected_velocity;
            typename Advection::AdjointWorkspace advection;
            simulation::VectorField<double> forced_velocity;
            simulation::VectorField<double> force_adjoint;
            typename Force::AdjointWorkspace force;
        };

        Solver(const Domain& domain, Configuration configuration) : advection(std::move(configuration.advection)), diffusion(std::move(configuration.diffusion)), force(std::move(configuration.force)), projection(domain, std::move(configuration.projection)), density_boundary(std::move(configuration.density_boundary)), temperature_boundary(std::move(configuration.temperature_boundary)), collider_density(domain.allocate_collider_field(configuration.collider_density)), collider_temperature(domain.allocate_collider_field(configuration.collider_temperature)) {}

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State allocate_state(const Domain& domain) const {
            State value{.density = domain.grid.allocate_cell_field<float>(), .temperature = domain.grid.allocate_cell_field<float>(), .velocity = domain.grid.allocate_mac_field<float>()};
            domain.grid.clear(value.density);
            domain.grid.clear(value.temperature);
            domain.grid.clear(value.velocity);
            return value;
        }

        [[nodiscard]] Control allocate_control(const Domain& domain) const {
            Control value{.density_source = domain.grid.allocate_cell_field<float>(), .temperature_source = domain.grid.allocate_cell_field<float>(), .external_acceleration = domain.grid.allocate_cell_vector_field<float>()};
            domain.grid.clear(value.density_source);
            domain.grid.clear(value.temperature_source);
            domain.grid.clear(value.external_acceleration);
            return value;
        }

        [[nodiscard]] typename Force::Parameters allocate_parameters(const Domain& domain) const {
            return force.allocate_parameters(domain);
        }

        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const {
            return {
                .sourced_density     = domain.grid.allocate_cell_field<float>(),
                .sourced_temperature = domain.grid.allocate_cell_field<float>(),
                .force               = force.allocate_cache(domain),
                .forced_velocity     = domain.grid.allocate_mac_field<float>(),
            };
        }

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const {
            return {
                .advection         = advection.allocate_workspace(domain),
                .advected_velocity = domain.grid.allocate_mac_field<float>(),
                .diffusion         = diffusion.allocate_workspace(domain),
                .diffused_velocity = domain.grid.allocate_mac_field<float>(),
                .projection        = projection.allocate_workspace(domain),
            };
        }

        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const {
            StateTangent value{.density = domain.grid.allocate_cell_field<float>(), .temperature = domain.grid.allocate_cell_field<float>(), .velocity = domain.grid.allocate_mac_field<float>()};
            domain.grid.clear(value.density);
            domain.grid.clear(value.temperature);
            domain.grid.clear(value.velocity);
            return value;
        }

        [[nodiscard]] ControlTangent allocate_control_tangent(const Domain& domain) const {
            ControlTangent value{.density_source = domain.grid.allocate_cell_field<float>(), .temperature_source = domain.grid.allocate_cell_field<float>(), .external_acceleration = domain.grid.allocate_cell_vector_field<float>()};
            domain.grid.clear(value.density_source);
            domain.grid.clear(value.temperature_source);
            domain.grid.clear(value.external_acceleration);
            return value;
        }

        [[nodiscard]] typename Force::ParameterTangent allocate_parameter_tangent(const Domain& domain) const {
            return force.allocate_parameter_tangent(domain);
        }

        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const {
            return {
                .sourced_density      = domain.grid.allocate_cell_field<float>(),
                .sourced_temperature  = domain.grid.allocate_cell_field<float>(),
                .force                = force.allocate_tangent_workspace(domain),
                .forced_velocity      = domain.grid.allocate_mac_field<float>(),
                .advection            = advection.allocate_tangent_workspace(domain),
                .advected_velocity    = domain.grid.allocate_mac_field<float>(),
                .diffusion            = diffusion.allocate_tangent_workspace(domain),
                .diffused_velocity    = domain.grid.allocate_mac_field<float>(),
                .constrained_velocity = domain.grid.allocate_mac_field<float>(),
                .projection           = projection.allocate_tangent_workspace(domain),
            };
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const {
            StateAdjoint value{.density = domain.grid.allocate_cell_field<double>(), .temperature = domain.grid.allocate_cell_field<double>(), .velocity = domain.grid.allocate_mac_field<double>()};
            domain.grid.clear(value.density);
            domain.grid.clear(value.temperature);
            domain.grid.clear(value.velocity);
            return value;
        }

        [[nodiscard]] ControlAdjoint allocate_control_adjoint(const Domain& domain) const {
            ControlAdjoint value{.density_source = domain.grid.allocate_cell_field<double>(), .temperature_source = domain.grid.allocate_cell_field<double>(), .external_acceleration = domain.grid.allocate_cell_vector_field<double>()};
            domain.grid.clear(value.density_source);
            domain.grid.clear(value.temperature_source);
            domain.grid.clear(value.external_acceleration);
            return value;
        }

        [[nodiscard]] typename Force::ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const {
            return force.allocate_parameter_adjoint(domain);
        }

        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const {
            return {
                .sourced_density      = domain.grid.allocate_cell_field<double>(),
                .sourced_temperature  = domain.grid.allocate_cell_field<double>(),
                .projected_velocity   = domain.grid.allocate_mac_field<double>(),
                .projection           = projection.allocate_adjoint_workspace(domain),
                .constrained_velocity = domain.grid.allocate_mac_field<double>(),
                .diffused_velocity    = domain.grid.allocate_mac_field<double>(),
                .diffusion            = diffusion.allocate_adjoint_workspace(domain),
                .advected_velocity    = domain.grid.allocate_mac_field<double>(),
                .advection            = advection.allocate_adjoint_workspace(domain),
                .forced_velocity      = domain.grid.allocate_mac_field<double>(),
                .force_adjoint        = domain.grid.allocate_cell_vector_field<double>(),
                .force                = force.allocate_adjoint_workspace(domain),
            };
        }

        void forward(const Domain& domain, const State& state, const Control& control, const typename Force::Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
            operators::add_source_forward(domain, state.density, control.density_source, cache.sourced_density);
            operators::add_source_forward(domain, state.temperature, control.temperature_source, cache.sourced_temperature);
            force.forward(domain, cache.sourced_density, cache.sourced_temperature, state.velocity, control.external_acceleration, parameters, cache.force);
            operators::integrate_velocity_forward(domain, state.velocity, cache.force.force, cache.forced_velocity);
            advection.velocity_forward(domain, cache.forced_velocity, workspace.advected_velocity, workspace.advection);
            diffusion.forward(domain, workspace.advected_velocity, workspace.diffused_velocity, workspace.diffusion);
            operators::constrain_velocity_forward(domain, workspace.diffused_velocity, workspace.advected_velocity);
            projection.forward(domain, workspace.advected_velocity, next_state.velocity, workspace.projection);
            advection.scalar_forward(domain, cache.sourced_density, next_state.velocity, density_boundary, collider_density, next_state.density);
            advection.scalar_forward(domain, cache.sourced_temperature, next_state.velocity, temperature_boundary, collider_temperature, next_state.temperature);
        }

        void jvp(const Domain& domain, const typename Force::Parameters& parameters, const State& next_state, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const typename Force::ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const {
            operators::add_source_jvp(domain, state_tangent.density, control_tangent.density_source, workspace.sourced_density);
            operators::add_source_jvp(domain, state_tangent.temperature, control_tangent.temperature_source, workspace.sourced_temperature);
            force.jvp(domain, cache.sourced_density, cache.sourced_temperature, workspace.sourced_density, workspace.sourced_temperature, state_tangent.velocity, control_tangent.external_acceleration, parameters, parameter_tangent, cache.force, workspace.force);
            operators::integrate_velocity_jvp(domain, state_tangent.velocity, workspace.force.force, workspace.forced_velocity);
            advection.velocity_jvp(domain, cache.forced_velocity, workspace.forced_velocity, workspace.advected_velocity, workspace.advection);
            diffusion.jvp(domain, workspace.advected_velocity, workspace.diffused_velocity, workspace.diffusion);
            operators::constrain_velocity_jvp(domain, workspace.diffused_velocity, workspace.constrained_velocity);
            projection.jvp(domain, workspace.constrained_velocity, next_state_tangent.velocity, workspace.projection);
            advection.scalar_jvp(domain, cache.sourced_density, workspace.sourced_density, next_state.velocity, next_state_tangent.velocity, density_boundary, next_state_tangent.density);
            advection.scalar_jvp(domain, cache.sourced_temperature, workspace.sourced_temperature, next_state.velocity, next_state_tangent.velocity, temperature_boundary, next_state_tangent.temperature);
        }

        void vjp(const Domain& domain, const typename Force::Parameters& parameters, const State& next_state, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, typename Force::ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
            domain.grid.clear(previous_state_adjoint.density);
            domain.grid.clear(previous_state_adjoint.temperature);
            domain.grid.clear(previous_state_adjoint.velocity);
            domain.grid.clear(control_adjoint.density_source);
            domain.grid.clear(control_adjoint.temperature_source);
            domain.grid.clear(control_adjoint.external_acceleration);
            domain.grid.clear(workspace.sourced_density);
            domain.grid.clear(workspace.sourced_temperature);
            domain.grid.clear(workspace.projected_velocity);
            domain.grid.copy(next_state_adjoint.velocity, workspace.projected_velocity);
            advection.scalar_vjp(domain, cache.sourced_density, next_state.velocity, density_boundary, next_state_adjoint.density, workspace.sourced_density, workspace.projected_velocity);
            advection.scalar_vjp(domain, cache.sourced_temperature, next_state.velocity, temperature_boundary, next_state_adjoint.temperature, workspace.sourced_temperature, workspace.projected_velocity);
            domain.grid.clear(workspace.constrained_velocity);
            projection.vjp(domain, workspace.projected_velocity, workspace.constrained_velocity, workspace.projection);
            domain.grid.clear(workspace.diffused_velocity);
            operators::constrain_velocity_vjp(domain, workspace.constrained_velocity, workspace.diffused_velocity);
            domain.grid.clear(workspace.advected_velocity);
            domain.grid.clear(workspace.forced_velocity);
            domain.grid.clear(workspace.force_adjoint);
            diffusion.vjp(domain, workspace.diffused_velocity, workspace.advected_velocity, workspace.diffusion);
            advection.velocity_vjp(domain, cache.forced_velocity, workspace.advected_velocity, workspace.forced_velocity, workspace.advection);
            operators::integrate_velocity_vjp(domain, workspace.forced_velocity, previous_state_adjoint.velocity, workspace.force_adjoint);
            force.vjp(domain, cache.sourced_density, cache.sourced_temperature, parameters, cache.force, workspace.force_adjoint, previous_state_adjoint.velocity, workspace.sourced_density, workspace.sourced_temperature, control_adjoint.external_acceleration, parameter_adjoint, workspace.force);
            operators::add_source_vjp(domain, workspace.sourced_density, previous_state_adjoint.density, control_adjoint.density_source);
            operators::add_source_vjp(domain, workspace.sourced_temperature, previous_state_adjoint.temperature, control_adjoint.temperature_source);
        }

    private:
        Advection advection;
        Diffusion diffusion;
        Force force;
        Projection projection;
        const ScalarBoundary density_boundary;
        const ScalarBoundary temperature_boundary;
        simulation::ScalarField<float> collider_density;
        simulation::ScalarField<float> collider_temperature;
    };
} // namespace physica::fluids::gas::solvers::smoke
