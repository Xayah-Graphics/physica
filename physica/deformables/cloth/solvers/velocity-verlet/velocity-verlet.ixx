module;

#include "velocity-verlet-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

export module physica.deformables.cloth.solvers.velocity_verlet;

import std;
import physica.deformables.cloth.forward;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::velocity_verlet {
    template <class Value, class Force, class Constraint>
        requires ForwardForceAlgorithm<Force, Value> && ForwardConstraintAlgorithm<Constraint, Value>
    struct Solver final {
        struct Configuration final {
            float time_step;
            typename Force::Configuration force;
            typename Constraint::Configuration constraint;
        };

        struct Parameters final {
            simulation::ScalarField<Value> masses;
            typename Force::Parameters force;
        };

        struct StepCache final {
            simulation::VectorField<Value> first_forces;
            [[no_unique_address]] typename Force::Cache first_force;
            simulation::VectorField<Value> predicted_positions;
            simulation::VectorField<Value> predicted_velocities;
            [[no_unique_address]] typename Constraint::Cache prediction_constraint;
            simulation::VectorField<Value> second_forces;
            [[no_unique_address]] typename Force::Cache second_force;
            [[no_unique_address]] typename Constraint::Cache final_constraint;
        };

        struct Workspace final {
            [[no_unique_address]] typename Force::Workspace first_force;
            simulation::VectorField<Value> unconstrained_predicted_positions;
            simulation::VectorField<Value> unconstrained_predicted_velocities;
            [[no_unique_address]] typename Constraint::Workspace prediction_constraint;
            [[no_unique_address]] typename Force::Workspace second_force;
            simulation::VectorField<Value> unconstrained_final_positions;
            simulation::VectorField<Value> unconstrained_final_velocities;
            [[no_unique_address]] typename Constraint::Workspace final_constraint;
        };

        Solver(const Model<Value>& model, Configuration configuration) : time_step(configuration.time_step), force(model, std::move(configuration.force)), constraint(model, std::move(configuration.constraint)) {}

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
                .first_forces          = simulation::VectorField<Value>(model.stream, model.particle_count),
                .first_force           = force.allocate_cache(model),
                .predicted_positions   = simulation::VectorField<Value>(model.stream, model.particle_count),
                .predicted_velocities  = simulation::VectorField<Value>(model.stream, model.particle_count),
                .prediction_constraint = constraint.allocate_cache(model),
                .second_forces         = simulation::VectorField<Value>(model.stream, model.particle_count),
                .second_force          = force.allocate_cache(model),
                .final_constraint      = constraint.allocate_cache(model),
            };
        }

        [[nodiscard]] Workspace allocate_workspace(const Model<Value>& model) const {
            return {
                .first_force                        = force.allocate_workspace(model),
                .unconstrained_predicted_positions  = simulation::VectorField<Value>(model.stream, model.particle_count),
                .unconstrained_predicted_velocities = simulation::VectorField<Value>(model.stream, model.particle_count),
                .prediction_constraint              = constraint.allocate_workspace(model),
                .second_force                       = force.allocate_workspace(model),
                .unconstrained_final_positions      = simulation::VectorField<Value>(model.stream, model.particle_count),
                .unconstrained_final_velocities     = simulation::VectorField<Value>(model.stream, model.particle_count),
                .final_constraint                   = constraint.allocate_workspace(model),
            };
        }

        void forward(const Model<Value>& model, const State<Value>& state, const Control<Value>& control, const Parameters& parameters, State<Value>& next_state, StepCache& cache, Workspace& workspace) const {
            force.forward(model, state.positions, state.velocities, control.external_forces, parameters.masses, parameters.force, cache.first_forces, cache.first_force, workspace.first_force);
            kernels::velocity_verlet_predict(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(state.positions), simulation::view(state.velocities), parameters.masses.values.data(), simulation::view(cache.first_forces), simulation::view(workspace.unconstrained_predicted_positions), simulation::view(workspace.unconstrained_predicted_velocities));
            constraint.forward(model, state.positions, state.velocities, workspace.unconstrained_predicted_positions, workspace.unconstrained_predicted_velocities, parameters.masses, static_cast<Value>(time_step), cache.predicted_positions, cache.predicted_velocities, cache.prediction_constraint, workspace.prediction_constraint);
            force.forward(model, cache.predicted_positions, cache.predicted_velocities, control.external_forces, parameters.masses, parameters.force, cache.second_forces, cache.second_force, workspace.second_force);
            kernels::velocity_verlet_second_half_kick(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(cache.predicted_positions), simulation::view(cache.predicted_velocities), parameters.masses.values.data(), simulation::view(cache.second_forces), simulation::view(workspace.unconstrained_final_positions), simulation::view(workspace.unconstrained_final_velocities));
            constraint.forward(model, state.positions, state.velocities, workspace.unconstrained_final_positions, workspace.unconstrained_final_velocities, parameters.masses, static_cast<Value>(time_step), next_state.positions, next_state.velocities, cache.final_constraint, workspace.final_constraint);
        }

    private:
        const float time_step;
        Force force;
        Constraint constraint;
    };
} // namespace physica::deformables::cloth::solvers::velocity_verlet
