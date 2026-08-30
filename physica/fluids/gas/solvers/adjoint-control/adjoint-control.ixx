module;

#include <physica/cuda.h>

export module physica.fluids.gas.solvers.adjoint_control;

import std;
import physica.fluids.gas.operators.conservation;
import physica.fluids.gas.domain;
import physica.fluids.gas.operators.pointwise;
import physica.fluids.gas.operators.advection;
import physica.fluids.gas.operators.diffusion;
import physica.fluids.gas.operators.projection;

export namespace physica::fluids::gas::solvers::adjoint_control {
    struct State final {
        simulation::ScalarField<float> density;
        simulation::VectorField<float> velocity;
    };

    struct StateTangent final {
        simulation::ScalarField<float> density;
        simulation::VectorField<float> velocity;
    };

    struct StateAdjoint final {
        simulation::ScalarField<double> density;
        simulation::VectorField<double> velocity;
    };

    struct DenseControl final {
        simulation::VectorField<float> force;
    };

    struct DenseControlTangent final {
        simulation::VectorField<float> force;
    };

    struct DenseControlAdjoint final {
        simulation::VectorField<double> force;
    };

    struct Keyframe final {
        std::uint32_t step{};
        State target;
        double density_weight{1.0};
        double velocity_weight{};
    };

    struct Problem final {
        std::uint32_t begin_step{};
        std::uint32_t step_count{};
        State initial_state;
        std::vector<Keyframe> keyframes;
    };

    template <class Algorithm>
    concept ForceAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace, const simulation::ScalarField<float>& density, const simulation::VectorField<double>& force_adjoint, simulation::ScalarField<double>& density_adjoint) {
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, density, cache);
        algorithm.jvp(domain, density, constant_cache, tangent_workspace);
        algorithm.vjp(domain, constant_cache, force_adjoint, density_adjoint, adjoint_workspace);
    };

    template <operators::AdvectionAlgorithm Advection, operators::DiffusionAlgorithm Diffusion, ForceAlgorithm Force, operators::ProjectionAlgorithm Projection>
    struct Solver final {
        struct Configuration final {
            typename Advection::Configuration advection{};
            typename Diffusion::Configuration diffusion{};
            typename Force::Configuration force{};
            operators::MassConservation::Configuration conservation{};
            typename Projection::Configuration projection{};
            ScalarBoundary density_boundary{};
            std::vector<float> collider_density{};
        };

        struct StepCache final {
            simulation::VectorField<float> controlled_velocity;
            simulation::ScalarField<float> advected_density;
            operators::MassConservation::Cache conservation;
            typename Force::Cache force;
        };

        struct Workspace final {
            typename Advection::Workspace advection;
            simulation::VectorField<float> advected_velocity;
            typename Diffusion::Workspace diffusion;
            simulation::VectorField<float> diffused_velocity;
            simulation::VectorField<float> forced_velocity;
            simulation::VectorField<float> constrained_velocity;
            typename Projection::Workspace projection;
        };

        struct TangentWorkspace final {
            simulation::VectorField<float> controlled_velocity;
            simulation::VectorField<float> advected_velocity;
            typename Advection::TangentWorkspace advection;
            simulation::ScalarField<float> advected_density;
            operators::MassConservation::TangentWorkspace conservation;
            simulation::ScalarField<float> transported_density;
            simulation::VectorField<float> diffused_velocity;
            typename Diffusion::TangentWorkspace diffusion;
            typename Force::TangentWorkspace force;
            simulation::VectorField<float> forced_velocity;
            simulation::VectorField<float> constrained_velocity;
            typename Projection::TangentWorkspace projection;
        };

        struct AdjointWorkspace final {
            simulation::VectorField<double> controlled_velocity;
            simulation::VectorField<double> advected_velocity;
            typename Advection::AdjointWorkspace advection;
            simulation::ScalarField<double> advected_density;
            operators::MassConservation::AdjointWorkspace conservation;
            simulation::ScalarField<double> transported_density;
            simulation::VectorField<double> diffused_velocity;
            typename Diffusion::AdjointWorkspace diffusion;
            typename Force::AdjointWorkspace force;
            simulation::VectorField<double> force_adjoint;
            simulation::VectorField<double> forced_velocity;
            simulation::VectorField<double> constrained_velocity;
            typename Projection::AdjointWorkspace projection;
        };

        Solver(const Domain& domain, Configuration configuration) : advection(std::move(configuration.advection)), diffusion(std::move(configuration.diffusion)), force(std::move(configuration.force)), conservation(std::move(configuration.conservation)), projection(domain, std::move(configuration.projection)), density_boundary(std::move(configuration.density_boundary)), collider_density(domain.allocate_collider_field(configuration.collider_density)) {}

        [[nodiscard]] State allocate_state(const Domain& domain) const {
            State value{.density = domain.grid.allocate_cell_field<float>(), .velocity = domain.grid.allocate_mac_field<float>()};
            domain.grid.clear(value.density);
            domain.grid.clear(value.velocity);
            return value;
        }

        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const {
            StateTangent value{.density = domain.grid.allocate_cell_field<float>(), .velocity = domain.grid.allocate_mac_field<float>()};
            domain.grid.clear(value.density);
            domain.grid.clear(value.velocity);
            return value;
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const {
            StateAdjoint value{.density = domain.grid.allocate_cell_field<double>(), .velocity = domain.grid.allocate_mac_field<double>()};
            domain.grid.clear(value.density);
            domain.grid.clear(value.velocity);
            return value;
        }

        [[nodiscard]] DenseControl allocate_control(const Domain& domain) const {
            DenseControl value{.force = domain.grid.allocate_cell_vector_field<float>()};
            domain.grid.clear(value.force);
            return value;
        }

        [[nodiscard]] DenseControlTangent allocate_control_tangent(const Domain& domain) const {
            DenseControlTangent value{.force = domain.grid.allocate_cell_vector_field<float>()};
            domain.grid.clear(value.force);
            return value;
        }

        [[nodiscard]] DenseControlAdjoint allocate_control_adjoint(const Domain& domain) const {
            DenseControlAdjoint value{.force = domain.grid.allocate_cell_vector_field<double>()};
            domain.grid.clear(value.force);
            return value;
        }

        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const {
            return {
                .controlled_velocity = domain.grid.allocate_mac_field<float>(),
                .advected_density    = domain.grid.allocate_cell_field<float>(),
                .conservation        = conservation.allocate_cache(domain),
                .force               = force.allocate_cache(domain),
            };
        }

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const {
            return {
                .advection            = advection.allocate_workspace(domain),
                .advected_velocity    = domain.grid.allocate_mac_field<float>(),
                .diffusion            = diffusion.allocate_workspace(domain),
                .diffused_velocity    = domain.grid.allocate_mac_field<float>(),
                .forced_velocity      = domain.grid.allocate_mac_field<float>(),
                .constrained_velocity = domain.grid.allocate_mac_field<float>(),
                .projection           = projection.allocate_workspace(domain),
            };
        }

        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const {
            return {
                .controlled_velocity  = domain.grid.allocate_mac_field<float>(),
                .advected_velocity    = domain.grid.allocate_mac_field<float>(),
                .advection            = advection.allocate_tangent_workspace(domain),
                .advected_density     = domain.grid.allocate_cell_field<float>(),
                .conservation         = conservation.allocate_tangent_workspace(domain),
                .transported_density  = domain.grid.allocate_cell_field<float>(),
                .diffused_velocity    = domain.grid.allocate_mac_field<float>(),
                .diffusion            = diffusion.allocate_tangent_workspace(domain),
                .force                = force.allocate_tangent_workspace(domain),
                .forced_velocity      = domain.grid.allocate_mac_field<float>(),
                .constrained_velocity = domain.grid.allocate_mac_field<float>(),
                .projection           = projection.allocate_tangent_workspace(domain),
            };
        }

        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const {
            return {
                .controlled_velocity  = domain.grid.allocate_mac_field<double>(),
                .advected_velocity    = domain.grid.allocate_mac_field<double>(),
                .advection            = advection.allocate_adjoint_workspace(domain),
                .advected_density     = domain.grid.allocate_cell_field<double>(),
                .conservation         = conservation.allocate_adjoint_workspace(domain),
                .transported_density  = domain.grid.allocate_cell_field<double>(),
                .diffused_velocity    = domain.grid.allocate_mac_field<double>(),
                .diffusion            = diffusion.allocate_adjoint_workspace(domain),
                .force                = force.allocate_adjoint_workspace(domain),
                .force_adjoint        = domain.grid.allocate_cell_vector_field<double>(),
                .forced_velocity      = domain.grid.allocate_mac_field<double>(),
                .constrained_velocity = domain.grid.allocate_mac_field<double>(),
                .projection           = projection.allocate_adjoint_workspace(domain),
            };
        }

        void forward(const Domain& domain, const State& state, const DenseControl& control, State& output, StepCache& cache, Workspace& workspace) const {
            operators::integrate_velocity_forward(domain, state.velocity, control.force, cache.controlled_velocity);
            advection.velocity_forward(domain, cache.controlled_velocity, workspace.advected_velocity, workspace.advection);
            advection.scalar_forward(domain, state.density, cache.controlled_velocity, density_boundary, collider_density, cache.advected_density);
            conservation.forward(domain, state.density, cache.advected_density, output.density, cache.conservation);
            diffusion.forward(domain, workspace.advected_velocity, workspace.diffused_velocity, workspace.diffusion);
            force.forward(domain, output.density, cache.force);
            operators::integrate_velocity_forward(domain, workspace.diffused_velocity, cache.force.force, workspace.forced_velocity);
            operators::constrain_velocity_forward(domain, workspace.forced_velocity, workspace.constrained_velocity);
            projection.forward(domain, workspace.constrained_velocity, output.velocity, workspace.projection);
        }

        void jvp(const Domain& domain, const State& state, const StepCache& cache, const StateTangent& state_tangent, const DenseControlTangent& control_tangent, StateTangent& output_tangent, TangentWorkspace& workspace) const {
            operators::integrate_velocity_jvp(domain, state_tangent.velocity, control_tangent.force, workspace.controlled_velocity);
            advection.velocity_jvp(domain, cache.controlled_velocity, workspace.controlled_velocity, workspace.advected_velocity, workspace.advection);
            advection.scalar_jvp(domain, state.density, state_tangent.density, cache.controlled_velocity, workspace.controlled_velocity, density_boundary, workspace.advected_density);
            conservation.jvp(domain, state.density, cache.advected_density, state_tangent.density, workspace.advected_density, cache.conservation, workspace.transported_density, workspace.conservation);
            diffusion.jvp(domain, workspace.advected_velocity, workspace.diffused_velocity, workspace.diffusion);
            force.jvp(domain, workspace.transported_density, cache.force, workspace.force);
            operators::integrate_velocity_jvp(domain, workspace.diffused_velocity, workspace.force.force, workspace.forced_velocity);
            operators::constrain_velocity_jvp(domain, workspace.forced_velocity, workspace.constrained_velocity);
            projection.jvp(domain, workspace.constrained_velocity, output_tangent.velocity, workspace.projection);
            domain.grid.copy(workspace.transported_density, output_tangent.density);
        }

        void vjp(const Domain& domain, const State& state, const StepCache& cache, const StateAdjoint& output_adjoint, StateAdjoint& state_adjoint, DenseControlAdjoint& control_adjoint, AdjointWorkspace& workspace) const {
            domain.grid.clear(state_adjoint.density);
            domain.grid.clear(state_adjoint.velocity);
            domain.grid.clear(control_adjoint.force);
            domain.grid.copy(output_adjoint.density, workspace.transported_density);
            domain.grid.clear(workspace.constrained_velocity);
            projection.vjp(domain, output_adjoint.velocity, workspace.constrained_velocity, workspace.projection);
            domain.grid.clear(workspace.forced_velocity);
            operators::constrain_velocity_vjp(domain, workspace.constrained_velocity, workspace.forced_velocity);
            domain.grid.clear(workspace.diffused_velocity);
            domain.grid.clear(workspace.force_adjoint);
            operators::integrate_velocity_vjp(domain, workspace.forced_velocity, workspace.diffused_velocity, workspace.force_adjoint);
            force.vjp(domain, cache.force, workspace.force_adjoint, workspace.transported_density, workspace.force);
            domain.grid.clear(workspace.advected_velocity);
            diffusion.vjp(domain, workspace.diffused_velocity, workspace.advected_velocity, workspace.diffusion);
            domain.grid.clear(workspace.controlled_velocity);
            advection.velocity_vjp(domain, cache.controlled_velocity, workspace.advected_velocity, workspace.controlled_velocity, workspace.advection);
            domain.grid.clear(workspace.advected_density);
            conservation.vjp(domain, cache.advected_density, cache.conservation, workspace.transported_density, state_adjoint.density, workspace.advected_density, workspace.conservation);
            advection.scalar_vjp(domain, state.density, cache.controlled_velocity, density_boundary, workspace.advected_density, state_adjoint.density, workspace.controlled_velocity);
            operators::integrate_velocity_vjp(domain, workspace.controlled_velocity, state_adjoint.velocity, control_adjoint.force);
        }

    private:
        Advection advection;
        Diffusion diffusion;
        Force force;
        operators::MassConservation conservation;
        Projection projection;
        const ScalarBoundary density_boundary;
        simulation::ScalarField<float> collider_density;
    };
} // namespace physica::fluids::gas::solvers::adjoint_control
