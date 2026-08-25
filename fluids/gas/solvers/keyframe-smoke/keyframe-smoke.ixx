module;

#include <physica/cuda.h>

export module physica.fluids.gas.keyframe_smoke;

import std;
import physica.fluids.gas.operators.conservation;
import physica.fluids.gas.domain;
import physica.fluids.gas.operators.pointwise;
import physica.fluids.gas.operators.advection;
import physica.fluids.gas.operators.diffusion;
import physica.fluids.gas.operators.projection;

export namespace physica::fluids::gas::keyframe_smoke {
    struct State final {
        CellField<float> density;
        StaggeredVectorField<float> velocity;
    };

    struct StateTangent final {
        CellField<float> density;
        StaggeredVectorField<float> velocity;
    };

    struct StateAdjoint final {
        CellField<double> density;
        StaggeredVectorField<double> velocity;
    };

    struct DenseControl final {
        CenteredVectorField<float> force;
    };

    struct DenseControlTangent final {
        CenteredVectorField<float> force;
    };

    struct DenseControlAdjoint final {
        CenteredVectorField<double> force;
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
    concept ForceAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm algorithm, const Domain& domain, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace, const CellField<float>& density, const StaggeredVectorField<float>& velocity, const CenteredVectorField<float>& control, const CenteredVectorField<double>& force_adjoint, CellField<double>& density_adjoint, StaggeredVectorField<double>& velocity_adjoint, CenteredVectorField<double>& control_adjoint) {
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, density, velocity, control, cache);
        algorithm.jvp(domain, density, velocity, control, constant_cache, tangent_workspace);
        algorithm.vjp(domain, constant_cache, force_adjoint, density_adjoint, velocity_adjoint, control_adjoint, adjoint_workspace);
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
            typename Force::Cache force;
            StaggeredVectorField<float> forced_velocity;
            CellField<float> advected_density;
            operators::MassConservation::Cache conservation;
        };

        struct Workspace final {
            typename Advection::Workspace advection;
            StaggeredVectorField<float> advected_velocity;
            typename Diffusion::Workspace diffusion;
            StaggeredVectorField<float> diffused_velocity;
            StaggeredVectorField<float> constrained_velocity;
            typename Projection::Workspace projection;
        };

        struct TangentWorkspace final {
            typename Force::TangentWorkspace force;
            StaggeredVectorField<float> forced_velocity;
            StaggeredVectorField<float> advected_velocity;
            StaggeredVectorField<float> diffused_velocity;
            typename Advection::TangentWorkspace advection;
            typename Diffusion::TangentWorkspace diffusion;
            StaggeredVectorField<float> constrained_velocity;
            typename Projection::TangentWorkspace projection;
            CellField<float> advected_density;
            operators::MassConservation::TangentWorkspace conservation;
        };

        struct AdjointWorkspace final {
            typename Force::AdjointWorkspace force;
            CenteredVectorField<double> total_force;
            StaggeredVectorField<double> forced_velocity;
            StaggeredVectorField<double> advected_velocity;
            StaggeredVectorField<double> diffused_velocity;
            typename Advection::AdjointWorkspace advection;
            typename Diffusion::AdjointWorkspace diffusion;
            StaggeredVectorField<double> projected_velocity;
            typename Projection::AdjointWorkspace projection;
            StaggeredVectorField<double> constrained_velocity;
            CellField<double> advected_density;
            operators::MassConservation::AdjointWorkspace conservation;
        };

        Solver(const Domain& domain, Configuration configuration) : advection(std::move(configuration.advection)), diffusion(std::move(configuration.diffusion)), force(std::move(configuration.force)), conservation(std::move(configuration.conservation)), projection(domain, std::move(configuration.projection)), density_boundary(std::move(configuration.density_boundary)), collider_density(domain.allocate_collider_field(configuration.collider_density)) {}

        [[nodiscard]] State allocate_state(const Domain& domain) const {
            State value{.density = domain.allocate_cell_field<float>(), .velocity = domain.allocate_staggered_vector_field<float>()};
            domain.clear(value.density);
            domain.clear(value.velocity);
            return value;
        }

        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const {
            StateTangent value{.density = domain.allocate_cell_field<float>(), .velocity = domain.allocate_staggered_vector_field<float>()};
            domain.clear(value.density);
            domain.clear(value.velocity);
            return value;
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const {
            StateAdjoint value{.density = domain.allocate_cell_field<double>(), .velocity = domain.allocate_staggered_vector_field<double>()};
            domain.clear(value.density);
            domain.clear(value.velocity);
            return value;
        }

        [[nodiscard]] DenseControl allocate_control(const Domain& domain) const {
            DenseControl value{.force = domain.allocate_centered_vector_field<float>()};
            domain.clear(value.force);
            return value;
        }

        [[nodiscard]] DenseControlTangent allocate_control_tangent(const Domain& domain) const {
            DenseControlTangent value{.force = domain.allocate_centered_vector_field<float>()};
            domain.clear(value.force);
            return value;
        }

        [[nodiscard]] DenseControlAdjoint allocate_control_adjoint(const Domain& domain) const {
            DenseControlAdjoint value{.force = domain.allocate_centered_vector_field<double>()};
            domain.clear(value.force);
            return value;
        }

        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const {
            return {
                .force            = force.allocate_cache(domain),
                .forced_velocity  = domain.allocate_staggered_vector_field<float>(),
                .advected_density = domain.allocate_cell_field<float>(),
                .conservation     = conservation.allocate_cache(domain),
            };
        }

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const {
            return {
                .advection            = advection.allocate_workspace(domain),
                .advected_velocity    = domain.allocate_staggered_vector_field<float>(),
                .diffusion            = diffusion.allocate_workspace(domain),
                .diffused_velocity    = domain.allocate_staggered_vector_field<float>(),
                .constrained_velocity = domain.allocate_staggered_vector_field<float>(),
                .projection           = projection.allocate_workspace(domain),
            };
        }

        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const {
            return {
                .force                = force.allocate_tangent_workspace(domain),
                .forced_velocity      = domain.allocate_staggered_vector_field<float>(),
                .advected_velocity    = domain.allocate_staggered_vector_field<float>(),
                .diffused_velocity    = domain.allocate_staggered_vector_field<float>(),
                .advection            = advection.allocate_tangent_workspace(domain),
                .diffusion            = diffusion.allocate_tangent_workspace(domain),
                .constrained_velocity = domain.allocate_staggered_vector_field<float>(),
                .projection           = projection.allocate_tangent_workspace(domain),
                .advected_density     = domain.allocate_cell_field<float>(),
                .conservation         = conservation.allocate_tangent_workspace(domain),
            };
        }

        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const {
            return {
                .force                = force.allocate_adjoint_workspace(domain),
                .total_force          = domain.allocate_centered_vector_field<double>(),
                .forced_velocity      = domain.allocate_staggered_vector_field<double>(),
                .advected_velocity    = domain.allocate_staggered_vector_field<double>(),
                .diffused_velocity    = domain.allocate_staggered_vector_field<double>(),
                .advection            = advection.allocate_adjoint_workspace(domain),
                .diffusion            = diffusion.allocate_adjoint_workspace(domain),
                .projected_velocity   = domain.allocate_staggered_vector_field<double>(),
                .projection           = projection.allocate_adjoint_workspace(domain),
                .constrained_velocity = domain.allocate_staggered_vector_field<double>(),
                .advected_density     = domain.allocate_cell_field<double>(),
                .conservation         = conservation.allocate_adjoint_workspace(domain),
            };
        }

        void forward(const Domain& domain, const State& state, const DenseControl& control, State& output, StepCache& cache, Workspace& workspace) const {
            force.forward(domain, state.density, state.velocity, control.force, cache.force);
            operators::integrate_velocity_forward(domain, state.velocity, cache.force.total, cache.forced_velocity);
            advection.velocity_forward(domain, cache.forced_velocity, workspace.advected_velocity, workspace.advection);
            diffusion.forward(domain, workspace.advected_velocity, workspace.diffused_velocity, workspace.diffusion);
            operators::constrain_velocity_forward(domain, workspace.diffused_velocity, workspace.constrained_velocity);
            projection.forward(domain, workspace.constrained_velocity, output.velocity, workspace.projection);
            advection.scalar_forward(domain, state.density, output.velocity, density_boundary, collider_density, cache.advected_density);
            conservation.forward(domain, state.density, cache.advected_density, output.density, cache.conservation);
        }

        void jvp(const Domain& domain, const State& state, const State& output, const StepCache& cache, const StateTangent& state_tangent, const DenseControlTangent& control_tangent, StateTangent& output_tangent, TangentWorkspace& workspace) const {
            force.jvp(domain, state_tangent.density, state_tangent.velocity, control_tangent.force, cache.force, workspace.force);
            operators::integrate_velocity_jvp(domain, state_tangent.velocity, workspace.force.total, workspace.forced_velocity);
            advection.velocity_jvp(domain, cache.forced_velocity, workspace.forced_velocity, workspace.advected_velocity, workspace.advection);
            diffusion.jvp(domain, workspace.advected_velocity, workspace.diffused_velocity, workspace.diffusion);
            operators::constrain_velocity_jvp(domain, workspace.diffused_velocity, workspace.constrained_velocity);
            projection.jvp(domain, workspace.constrained_velocity, output_tangent.velocity, workspace.projection);
            advection.scalar_jvp(domain, state.density, state_tangent.density, output.velocity, output_tangent.velocity, density_boundary, workspace.advected_density);
            conservation.jvp(domain, state.density, cache.advected_density, state_tangent.density, workspace.advected_density, cache.conservation, output_tangent.density, workspace.conservation);
        }

        void vjp(const Domain& domain, const State& state, const State& output, const StepCache& cache, const StateAdjoint& output_adjoint, StateAdjoint& state_adjoint, DenseControlAdjoint& control_adjoint, AdjointWorkspace& workspace) const {
            domain.clear(state_adjoint.density);
            domain.clear(state_adjoint.velocity);
            domain.clear(control_adjoint.force);
            domain.clear(workspace.projected_velocity);
            domain.copy(output_adjoint.velocity, workspace.projected_velocity);
            domain.clear(workspace.advected_density);
            conservation.vjp(domain, cache.advected_density, cache.conservation, output_adjoint.density, state_adjoint.density, workspace.advected_density, workspace.conservation);
            advection.scalar_vjp(domain, state.density, output.velocity, density_boundary, workspace.advected_density, state_adjoint.density, workspace.projected_velocity);
            domain.clear(workspace.constrained_velocity);
            projection.vjp(domain, workspace.projected_velocity, workspace.constrained_velocity, workspace.projection);
            domain.clear(workspace.diffused_velocity);
            operators::constrain_velocity_vjp(domain, workspace.constrained_velocity, workspace.diffused_velocity);
            domain.clear(workspace.advected_velocity);
            diffusion.vjp(domain, workspace.diffused_velocity, workspace.advected_velocity, workspace.diffusion);
            domain.clear(workspace.forced_velocity);
            advection.velocity_vjp(domain, cache.forced_velocity, workspace.advected_velocity, workspace.forced_velocity, workspace.advection);
            domain.clear(workspace.total_force);
            operators::integrate_velocity_vjp(domain, workspace.forced_velocity, state_adjoint.velocity, workspace.total_force);
            force.vjp(domain, cache.force, workspace.total_force, state_adjoint.density, state_adjoint.velocity, control_adjoint.force, workspace.force);
        }

    private:
        Advection advection;
        Diffusion diffusion;
        Force force;
        operators::MassConservation conservation;
        Projection projection;
        const ScalarBoundary density_boundary;
        CellField<float> collider_density;
    };
} // namespace physica::fluids::gas::keyframe_smoke
