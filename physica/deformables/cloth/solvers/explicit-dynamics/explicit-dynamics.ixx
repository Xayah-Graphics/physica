module;

#include <physica/cuda.h>

export module physica.deformables.cloth.explicit_dynamics;

import std;
import physica.deformables.cloth.model;

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
    concept ForceAlgorithm = std::constructible_from<Algorithm, const Model&, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Model& model, const VectorField<float>& vector, VectorField<float>& vector_output, const VectorField<double>& vector_adjoint, VectorField<double>& vector_adjoint_output, const ScalarField<float>& scalar, ScalarField<double>& scalar_adjoint_output, const typename Algorithm::Parameters& parameters, const typename Algorithm::ParameterTangent& parameter_tangent, typename Algorithm::ParameterAdjoint& parameter_adjoint, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace) {
        { algorithm.allocate_parameters(model) } -> std::same_as<typename Algorithm::Parameters>;
        { algorithm.allocate_parameter_tangent(model) } -> std::same_as<typename Algorithm::ParameterTangent>;
        { algorithm.allocate_parameter_adjoint(model) } -> std::same_as<typename Algorithm::ParameterAdjoint>;
        { algorithm.allocate_cache(model) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(model) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(model) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(model) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(model, vector, vector, vector, scalar, parameters, vector_output, cache, workspace);
        algorithm.jvp(model, vector, vector, vector, scalar, parameters, vector, constant_cache, vector, vector, vector, scalar, parameter_tangent, vector_output, tangent_workspace);
        algorithm.vjp(model, vector, vector, vector, scalar, parameters, vector, constant_cache, vector_adjoint, vector_adjoint_output, vector_adjoint_output, vector_adjoint_output, scalar_adjoint_output, parameter_adjoint, adjoint_workspace);
    };

    template <class Algorithm>
    concept IntegratorAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Model& model, const VectorField<float>& vector, VectorField<float>& vector_output, const VectorField<double>& vector_adjoint, VectorField<double>& vector_adjoint_output, const ScalarField<float>& scalar, ScalarField<double>& scalar_adjoint, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace) {
        { algorithm.allocate_cache(model) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(model) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(model) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(model) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(model, vector, vector, scalar, vector, vector_output, vector_output, cache, workspace);
        algorithm.jvp(model, scalar, vector, constant_cache, vector, vector, scalar, vector, vector_output, vector_output, tangent_workspace);
        algorithm.vjp(model, scalar, vector, constant_cache, vector_adjoint, vector_adjoint, vector_adjoint_output, vector_adjoint_output, vector_adjoint_output, scalar_adjoint, adjoint_workspace);
    };

    template <class Algorithm>
    concept ConstraintAlgorithm = std::constructible_from<Algorithm, const Model&, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Model& model, const VectorField<float>& vector, VectorField<float>& vector_output, const VectorField<double>& vector_adjoint, VectorField<double>& vector_adjoint_output, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace) {
        { algorithm.allocate_cache(model) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(model) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(model) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(model) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(model, vector, vector, vector_output, vector_output, cache, workspace);
        algorithm.jvp(model, vector, vector, vector, vector, constant_cache, vector, vector, vector_output, vector_output, tangent_workspace);
        algorithm.vjp(model, vector, vector, vector, vector, constant_cache, vector_adjoint, vector_adjoint, vector_adjoint_output, vector_adjoint_output, adjoint_workspace);
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

        Solver(const Model& model, Configuration configuration) : force(model, std::move(configuration.force)), integrator(std::move(configuration.integrator)), constraint(model, std::move(configuration.constraint)) {}

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State allocate_state(const Model& model) const {
            State result{
                .positions  = model.fields.allocate_vector_field<float>(model.particle_count),
                .velocities = model.fields.allocate_vector_field<float>(model.particle_count),
            };
            model.fields.clear(result.positions);
            model.fields.clear(result.velocities);
            return result;
        }

        [[nodiscard]] Control allocate_control(const Model& model) const {
            Control result{.external_forces = model.fields.allocate_vector_field<float>(model.particle_count)};
            model.fields.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] Parameters allocate_parameters(const Model& model) const {
            Parameters result{.masses = model.fields.allocate_scalar_field<float>(model.particle_count), .force = force.allocate_parameters(model)};
            model.fields.clear(result.masses);
            return result;
        }

        [[nodiscard]] StepCache allocate_step_cache(const Model& model) const {
            return {
                .forces                = model.fields.allocate_vector_field<float>(model.particle_count),
                .force                 = force.allocate_cache(model),
                .integrated_positions  = model.fields.allocate_vector_field<float>(model.particle_count),
                .integrated_velocities = model.fields.allocate_vector_field<float>(model.particle_count),
                .integrator            = integrator.allocate_cache(model),
                .constraint            = constraint.allocate_cache(model),
            };
        }

        [[nodiscard]] Workspace allocate_workspace(const Model& model) const {
            return {.force = force.allocate_workspace(model), .integrator = integrator.allocate_workspace(model), .constraint = constraint.allocate_workspace(model)};
        }

        [[nodiscard]] StateTangent allocate_state_tangent(const Model& model) const {
            StateTangent result{
                .positions  = model.fields.allocate_vector_field<float>(model.particle_count),
                .velocities = model.fields.allocate_vector_field<float>(model.particle_count),
            };
            model.fields.clear(result.positions);
            model.fields.clear(result.velocities);
            return result;
        }

        [[nodiscard]] ControlTangent allocate_control_tangent(const Model& model) const {
            ControlTangent result{.external_forces = model.fields.allocate_vector_field<float>(model.particle_count)};
            model.fields.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Model& model) const {
            ParameterTangent result{.masses = model.fields.allocate_scalar_field<float>(model.particle_count), .force = force.allocate_parameter_tangent(model)};
            model.fields.clear(result.masses);
            return result;
        }

        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Model& model) const {
            return {
                .forces                = model.fields.allocate_vector_field<float>(model.particle_count),
                .force                 = force.allocate_tangent_workspace(model),
                .integrated_positions  = model.fields.allocate_vector_field<float>(model.particle_count),
                .integrated_velocities = model.fields.allocate_vector_field<float>(model.particle_count),
                .integrator            = integrator.allocate_tangent_workspace(model),
                .constraint            = constraint.allocate_tangent_workspace(model),
            };
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Model& model) const {
            StateAdjoint result{
                .positions  = model.fields.allocate_vector_field<double>(model.particle_count),
                .velocities = model.fields.allocate_vector_field<double>(model.particle_count),
            };
            model.fields.clear(result.positions);
            model.fields.clear(result.velocities);
            return result;
        }

        [[nodiscard]] ControlAdjoint allocate_control_adjoint(const Model& model) const {
            ControlAdjoint result{.external_forces = model.fields.allocate_vector_field<double>(model.particle_count)};
            model.fields.clear(result.external_forces);
            return result;
        }

        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Model& model) const {
            ParameterAdjoint result{.masses = model.fields.allocate_scalar_field<double>(model.particle_count), .force = force.allocate_parameter_adjoint(model)};
            model.fields.clear(result.masses);
            return result;
        }

        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Model& model) const {
            return {
                .integrated_positions  = model.fields.allocate_vector_field<double>(model.particle_count),
                .integrated_velocities = model.fields.allocate_vector_field<double>(model.particle_count),
                .constraint            = constraint.allocate_adjoint_workspace(model),
                .forces                = model.fields.allocate_vector_field<double>(model.particle_count),
                .integrator            = integrator.allocate_adjoint_workspace(model),
                .force                 = force.allocate_adjoint_workspace(model),
            };
        }

        void forward(const Model& model, const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
            force.forward(model, state.positions, state.velocities, control.external_forces, parameters.masses, parameters.force, cache.forces, cache.force, workspace.force);
            integrator.forward(model, state.positions, state.velocities, parameters.masses, cache.forces, cache.integrated_positions, cache.integrated_velocities, cache.integrator, workspace.integrator);
            constraint.forward(model, cache.integrated_positions, cache.integrated_velocities, next_state.positions, next_state.velocities, cache.constraint, workspace.constraint);
        }

        void jvp(const Model& model, const State& state, const Control& control, const Parameters& parameters, const State& next_state, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const {
            force.jvp(model, state.positions, state.velocities, control.external_forces, parameters.masses, parameters.force, cache.forces, cache.force, state_tangent.positions, state_tangent.velocities, control_tangent.external_forces, parameter_tangent.masses, parameter_tangent.force, workspace.forces, workspace.force);
            integrator.jvp(model, parameters.masses, cache.forces, cache.integrator, state_tangent.positions, state_tangent.velocities, parameter_tangent.masses, workspace.forces, workspace.integrated_positions, workspace.integrated_velocities, workspace.integrator);
            constraint.jvp(model, cache.integrated_positions, cache.integrated_velocities, next_state.positions, next_state.velocities, cache.constraint, workspace.integrated_positions, workspace.integrated_velocities, next_state_tangent.positions, next_state_tangent.velocities, workspace.constraint);
        }

        void vjp(const Model& model, const State& state, const Control& control, const Parameters& parameters, const State& next_state, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
            model.fields.clear(previous_state_adjoint.positions);
            model.fields.clear(previous_state_adjoint.velocities);
            model.fields.clear(control_adjoint.external_forces);
            model.fields.clear(workspace.integrated_positions);
            model.fields.clear(workspace.integrated_velocities);
            model.fields.clear(workspace.forces);
            constraint.vjp(model, cache.integrated_positions, cache.integrated_velocities, next_state.positions, next_state.velocities, cache.constraint, next_state_adjoint.positions, next_state_adjoint.velocities, workspace.integrated_positions, workspace.integrated_velocities, workspace.constraint);
            integrator.vjp(model, parameters.masses, cache.forces, cache.integrator, workspace.integrated_positions, workspace.integrated_velocities, previous_state_adjoint.positions, previous_state_adjoint.velocities, workspace.forces, parameter_adjoint.masses, workspace.integrator);
            force.vjp(model, state.positions, state.velocities, control.external_forces, parameters.masses, parameters.force, cache.forces, cache.force, workspace.forces, previous_state_adjoint.positions, previous_state_adjoint.velocities, control_adjoint.external_forces, parameter_adjoint.masses, parameter_adjoint.force, workspace.force);
        }

    private:
        Force force;
        Integrator integrator;
        Constraint constraint;
    };
} // namespace physica::deformables::cloth::explicit_dynamics
