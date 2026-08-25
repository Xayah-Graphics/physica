module;

#include <physica/cuda.h>

export module physica.deformables.cloth.explicit_dynamics;

import std;
import physica.deformables.cloth.domain;

export namespace physica::deformables::cloth::explicit_dynamics {
    struct State final {
        VectorField<float> positions;
        VectorField<float> velocities;
    };

    struct Control final {
        VectorField<float> external_forces;
    };

    struct StateTangent final {
        VectorField<float> positions;
        VectorField<float> velocities;
    };

    struct ControlTangent final {
        VectorField<float> external_forces;
    };

    struct StateAdjoint final {
        VectorField<double> positions;
        VectorField<double> velocities;
    };

    struct ControlAdjoint final {
        VectorField<double> external_forces;
    };

    template <class Algorithm>
    concept ForceAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, const VectorField<float>& vector, VectorField<float>& vector_output, const VectorField<double>& vector_adjoint, VectorField<double>& vector_adjoint_output, const ScalarField<float>& scalar, ScalarField<double>& scalar_adjoint_output, const typename Algorithm::Parameters& parameters, const typename Algorithm::ParameterTangent& parameter_tangent, typename Algorithm::ParameterAdjoint& parameter_adjoint, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace) {
        { algorithm.allocate_parameters(domain) } -> std::same_as<typename Algorithm::Parameters>;
        { algorithm.allocate_parameter_tangent(domain) } -> std::same_as<typename Algorithm::ParameterTangent>;
        { algorithm.allocate_parameter_adjoint(domain) } -> std::same_as<typename Algorithm::ParameterAdjoint>;
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, vector, vector, vector, scalar, parameters, vector_output, cache, workspace);
        algorithm.jvp(domain, vector, vector, vector, scalar, parameters, vector, constant_cache, vector, vector, vector, scalar, parameter_tangent, vector_output, tangent_workspace);
        algorithm.vjp(domain, vector, vector, vector, scalar, parameters, vector, constant_cache, vector_adjoint, vector_adjoint_output, vector_adjoint_output, vector_adjoint_output, scalar_adjoint_output, parameter_adjoint, adjoint_workspace);
    };

    template <class Algorithm>
    concept IntegratorAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, const VectorField<float>& vector, VectorField<float>& vector_output, const VectorField<double>& vector_adjoint, VectorField<double>& vector_adjoint_output, const ScalarField<float>& scalar, ScalarField<double>& scalar_adjoint, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace) {
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, vector, vector, scalar, vector, vector_output, vector_output, cache, workspace);
        algorithm.jvp(domain, scalar, vector, constant_cache, vector, vector, scalar, vector, vector_output, vector_output, tangent_workspace);
        algorithm.vjp(domain, scalar, vector, constant_cache, vector_adjoint, vector_adjoint, vector_adjoint_output, vector_adjoint_output, vector_adjoint_output, scalar_adjoint, adjoint_workspace);
    };

    template <class Algorithm>
    concept ConstraintAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, const VectorField<float>& vector, VectorField<float>& vector_output, const VectorField<double>& vector_adjoint, VectorField<double>& vector_adjoint_output, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace) {
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, vector, vector, vector_output, vector_output, cache, workspace);
        algorithm.jvp(domain, vector, vector, vector, vector, constant_cache, vector, vector, vector_output, vector_output, tangent_workspace);
        algorithm.vjp(domain, vector, vector, vector, vector, constant_cache, vector_adjoint, vector_adjoint, vector_adjoint_output, vector_adjoint_output, adjoint_workspace);
    };

    template <ForceAlgorithm Force, IntegratorAlgorithm Integrator, ConstraintAlgorithm Constraint>
    struct Solver final {
        struct Configuration final {
            typename Force::Configuration force;
            typename Integrator::Configuration integrator;
            typename Constraint::Configuration constraint;
        };

        struct Parameters final {
            ScalarField<float> masses;
            typename Force::Parameters force;
        };

        struct ParameterTangent final {
            ScalarField<float> masses;
            typename Force::ParameterTangent force;
        };

        struct ParameterAdjoint final {
            ScalarField<double> masses;
            typename Force::ParameterAdjoint force;
        };

        struct StepCache final {
            VectorField<float> forces;
            [[no_unique_address]] typename Force::Cache force;
            VectorField<float> integrated_positions;
            VectorField<float> integrated_velocities;
            [[no_unique_address]] typename Integrator::Cache integrator;
            [[no_unique_address]] typename Constraint::Cache constraint;
        };

        struct Workspace final {
            [[no_unique_address]] typename Force::Workspace force;
            [[no_unique_address]] typename Integrator::Workspace integrator;
            [[no_unique_address]] typename Constraint::Workspace constraint;
        };

        struct TangentWorkspace final {
            VectorField<float> forces;
            [[no_unique_address]] typename Force::TangentWorkspace force;
            VectorField<float> integrated_positions;
            VectorField<float> integrated_velocities;
            [[no_unique_address]] typename Integrator::TangentWorkspace integrator;
            [[no_unique_address]] typename Constraint::TangentWorkspace constraint;
        };

        struct AdjointWorkspace final {
            VectorField<double> integrated_positions;
            VectorField<double> integrated_velocities;
            [[no_unique_address]] typename Constraint::AdjointWorkspace constraint;
            VectorField<double> forces;
            [[no_unique_address]] typename Integrator::AdjointWorkspace integrator;
            [[no_unique_address]] typename Force::AdjointWorkspace force;
        };

        Solver(const Domain& domain, Configuration configuration) : force(domain, std::move(configuration.force)), integrator(std::move(configuration.integrator)), constraint(domain, std::move(configuration.constraint)) {}

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State allocate_state(const Domain& domain) const {
            State result{
                .positions  = domain.allocate_vector_field<float>(domain.particle_count),
                .velocities = domain.allocate_vector_field<float>(domain.particle_count),
            };
            domain.clear(result.positions);
            domain.clear(result.velocities);
            return result;
        }

        [[nodiscard]] Control allocate_control(const Domain& domain) const {
            Control result{.external_forces = domain.allocate_vector_field<float>(domain.particle_count)};
            domain.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const {
            Parameters result{.masses = domain.allocate_scalar_field<float>(domain.particle_count), .force = force.allocate_parameters(domain)};
            domain.clear(result.masses);
            return result;
        }

        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const {
            return {
                .forces                = domain.allocate_vector_field<float>(domain.particle_count),
                .force                 = force.allocate_cache(domain),
                .integrated_positions  = domain.allocate_vector_field<float>(domain.particle_count),
                .integrated_velocities = domain.allocate_vector_field<float>(domain.particle_count),
                .integrator            = integrator.allocate_cache(domain),
                .constraint            = constraint.allocate_cache(domain),
            };
        }

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const {
            return {.force = force.allocate_workspace(domain), .integrator = integrator.allocate_workspace(domain), .constraint = constraint.allocate_workspace(domain)};
        }

        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const {
            StateTangent result{
                .positions  = domain.allocate_vector_field<float>(domain.particle_count),
                .velocities = domain.allocate_vector_field<float>(domain.particle_count),
            };
            domain.clear(result.positions);
            domain.clear(result.velocities);
            return result;
        }

        [[nodiscard]] ControlTangent allocate_control_tangent(const Domain& domain) const {
            ControlTangent result{.external_forces = domain.allocate_vector_field<float>(domain.particle_count)};
            domain.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const {
            ParameterTangent result{.masses = domain.allocate_scalar_field<float>(domain.particle_count), .force = force.allocate_parameter_tangent(domain)};
            domain.clear(result.masses);
            return result;
        }

        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const {
            return {
                .forces                = domain.allocate_vector_field<float>(domain.particle_count),
                .force                 = force.allocate_tangent_workspace(domain),
                .integrated_positions  = domain.allocate_vector_field<float>(domain.particle_count),
                .integrated_velocities = domain.allocate_vector_field<float>(domain.particle_count),
                .integrator            = integrator.allocate_tangent_workspace(domain),
                .constraint            = constraint.allocate_tangent_workspace(domain),
            };
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const {
            StateAdjoint result{
                .positions  = domain.allocate_vector_field<double>(domain.particle_count),
                .velocities = domain.allocate_vector_field<double>(domain.particle_count),
            };
            domain.clear(result.positions);
            domain.clear(result.velocities);
            return result;
        }

        [[nodiscard]] ControlAdjoint allocate_control_adjoint(const Domain& domain) const {
            ControlAdjoint result{.external_forces = domain.allocate_vector_field<double>(domain.particle_count)};
            domain.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const {
            ParameterAdjoint result{.masses = domain.allocate_scalar_field<double>(domain.particle_count), .force = force.allocate_parameter_adjoint(domain)};
            domain.clear(result.masses);
            return result;
        }

        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const {
            return {
                .integrated_positions  = domain.allocate_vector_field<double>(domain.particle_count),
                .integrated_velocities = domain.allocate_vector_field<double>(domain.particle_count),
                .constraint            = constraint.allocate_adjoint_workspace(domain),
                .forces                = domain.allocate_vector_field<double>(domain.particle_count),
                .integrator            = integrator.allocate_adjoint_workspace(domain),
                .force                 = force.allocate_adjoint_workspace(domain),
            };
        }

        void forward(const Domain& domain, const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
            force.forward(domain, state.positions, state.velocities, control.external_forces, parameters.masses, parameters.force, cache.forces, cache.force, workspace.force);
            integrator.forward(domain, state.positions, state.velocities, parameters.masses, cache.forces, cache.integrated_positions, cache.integrated_velocities, cache.integrator, workspace.integrator);
            constraint.forward(domain, cache.integrated_positions, cache.integrated_velocities, next_state.positions, next_state.velocities, cache.constraint, workspace.constraint);
        }

        void jvp(const Domain& domain, const State& state, const Control& control, const Parameters& parameters, const State& next_state, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const {
            force.jvp(domain, state.positions, state.velocities, control.external_forces, parameters.masses, parameters.force, cache.forces, cache.force, state_tangent.positions, state_tangent.velocities, control_tangent.external_forces, parameter_tangent.masses, parameter_tangent.force, workspace.forces, workspace.force);
            integrator.jvp(domain, parameters.masses, cache.forces, cache.integrator, state_tangent.positions, state_tangent.velocities, parameter_tangent.masses, workspace.forces, workspace.integrated_positions, workspace.integrated_velocities, workspace.integrator);
            constraint.jvp(domain, cache.integrated_positions, cache.integrated_velocities, next_state.positions, next_state.velocities, cache.constraint, workspace.integrated_positions, workspace.integrated_velocities, next_state_tangent.positions, next_state_tangent.velocities, workspace.constraint);
        }

        void vjp(const Domain& domain, const State& state, const Control& control, const Parameters& parameters, const State& next_state, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
            domain.clear(previous_state_adjoint.positions);
            domain.clear(previous_state_adjoint.velocities);
            domain.clear(control_adjoint.external_forces);
            domain.clear(workspace.integrated_positions);
            domain.clear(workspace.integrated_velocities);
            domain.clear(workspace.forces);
            constraint.vjp(domain, cache.integrated_positions, cache.integrated_velocities, next_state.positions, next_state.velocities, cache.constraint, next_state_adjoint.positions, next_state_adjoint.velocities, workspace.integrated_positions, workspace.integrated_velocities, workspace.constraint);
            integrator.vjp(domain, parameters.masses, cache.forces, cache.integrator, workspace.integrated_positions, workspace.integrated_velocities, previous_state_adjoint.positions, previous_state_adjoint.velocities, workspace.forces, parameter_adjoint.masses, workspace.integrator);
            force.vjp(domain, state.positions, state.velocities, control.external_forces, parameters.masses, parameters.force, cache.forces, cache.force, workspace.forces, previous_state_adjoint.positions, previous_state_adjoint.velocities, control_adjoint.external_forces, parameter_adjoint.masses, parameter_adjoint.force, workspace.force);
        }

    private:
        Force force;
        Integrator integrator;
        Constraint constraint;
    };
} // namespace physica::deformables::cloth::explicit_dynamics
