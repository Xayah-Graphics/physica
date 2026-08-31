module;

#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.explicit_dynamics;

import std;
import physica.deformables.cloth.forward;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::explicit_dynamics {
    template <class Value, class Force, class Integrator, class Constraint>
        requires ForwardForceAlgorithm<Force, Value> && ForwardIntegratorAlgorithm<Integrator, Value> && ForwardConstraintAlgorithm<Constraint, Value>
    struct Solver final {
        struct Configuration final {
            typename Force::Configuration force;
            typename Integrator::Configuration integrator;
            typename Constraint::Configuration constraint;
        };

        struct Parameters final {
            simulation::ScalarField<Value> masses;
            typename Force::Parameters force;
        };

        struct StepCache final {
            simulation::VectorField<Value> forces;
            [[no_unique_address]] typename Force::Cache force;
            simulation::VectorField<Value> integrated_positions;
            simulation::VectorField<Value> integrated_velocities;
            [[no_unique_address]] typename Integrator::Cache integrator;
            [[no_unique_address]] typename Constraint::Cache constraint;
        };

        struct Workspace final {
            [[no_unique_address]] typename Force::Workspace force;
            [[no_unique_address]] typename Integrator::Workspace integrator;
            [[no_unique_address]] typename Constraint::Workspace constraint;
        };

        Solver(const Model<Value>& model, Configuration configuration) : force(model, std::move(configuration.force)), integrator(std::move(configuration.integrator)), constraint(model, std::move(configuration.constraint)) {}

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State<Value> allocate_state(const Model<Value>& model) const {
            State<Value> result(model.stream, model.particle_count);
            simulation::clear(model.stream, result.positions);
            simulation::clear(model.stream, result.velocities);
            return result;
        }

        [[nodiscard]] Control<Value> allocate_control(const Model<Value>& model) const {
            Control<Value> result(model.stream, model.particle_count);
            simulation::clear(model.stream, result.external_forces);
            return result;
        }

        [[nodiscard]] Parameters allocate_parameters(const Model<Value>& model) const {
            Parameters result{.masses = simulation::ScalarField<Value>(model.stream, model.particle_count), .force = force.allocate_parameters(model)};
            simulation::clear(model.stream, result.masses);
            return result;
        }

        [[nodiscard]] StepCache allocate_step_cache(const Model<Value>& model) const {
            return {
                .forces                = simulation::VectorField<Value>(model.stream, model.particle_count),
                .force                 = force.allocate_cache(model),
                .integrated_positions  = simulation::VectorField<Value>(model.stream, model.particle_count),
                .integrated_velocities = simulation::VectorField<Value>(model.stream, model.particle_count),
                .integrator            = integrator.allocate_cache(model),
                .constraint            = constraint.allocate_cache(model),
            };
        }

        [[nodiscard]] Workspace allocate_workspace(const Model<Value>& model) const {
            return {.force = force.allocate_workspace(model), .integrator = integrator.allocate_workspace(model), .constraint = constraint.allocate_workspace(model)};
        }

        void forward(const Model<Value>& model, const State<Value>& state, const Control<Value>& control, const Parameters& parameters, State<Value>& next_state, StepCache& cache, Workspace& workspace) const {
            force.forward(model, state.positions, state.velocities, control.external_forces, parameters.masses, parameters.force, cache.forces, cache.force, workspace.force);
            integrator.forward(model, state.positions, state.velocities, parameters.masses, cache.forces, cache.integrated_positions, cache.integrated_velocities, cache.integrator, workspace.integrator);
            constraint.forward(model, state.positions, state.velocities, cache.integrated_positions, cache.integrated_velocities, parameters.masses, static_cast<Value>(integrator.configuration.time_step), next_state.positions, next_state.velocities, cache.constraint, workspace.constraint);
        }

    private:
        Force force;
        Integrator integrator;
        Constraint constraint;
    };
} // namespace physica::deformables::cloth::solvers::explicit_dynamics
