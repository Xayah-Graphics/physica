module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

export module physica.fluids.gas.keyframe_smoke.solver;

import std;
export import physica.fluids.gas.keyframe_smoke.domain;

export namespace physica::fluids::gas::keyframe_smoke {
    struct State final {
        ScalarField density;
        StaggeredVectorField velocity;
    };

    struct StateTangent final {
        ScalarField density;
        StaggeredVectorField velocity;
    };

    struct StateAdjoint final {
        ScalarAdjointField density;
        StaggeredVectorAdjointField velocity;
    };

    struct DenseControl final {
        CenteredVectorField force;
    };

    struct DenseControlTangent final {
        CenteredVectorField force;
    };

    struct DenseControlAdjoint final {
        CenteredVectorAdjointField force;
    };

    struct VorticityCache final {
        CenteredVectorField centered_velocity;
        CenteredVectorField vorticity;
        ScalarField magnitude;
        CenteredVectorField normal;
        ScalarField normalizer;
    };

    struct StepCache final {
        CenteredVectorField physical_force;
        CenteredVectorField control_force;
        CenteredVectorField total_force;
        VorticityCache vorticity;
        StaggeredVectorField forced_velocity;
        StaggeredVectorField advected_velocity;
        StaggeredVectorField diffused_velocity;
        ScalarField pressure_rhs;
        ScalarField pressure;
        StaggeredVectorField projected_velocity;
        ScalarField advected_density;
        ::cuda::device_buffer<double> input_mass;
        ::cuda::device_buffer<double> advected_mass;
    };

    struct SolverConfiguration final {
        std::uint32_t diffusion_iterations{20u};
        std::uint32_t pressure_iterations{80u};
        float viscosity{};
        float density_buoyancy{2.0F};
        float vorticity_confinement{2.0F};
        float density_dissipation{};
    };

    struct Solver final {
        const SolverConfiguration configuration;

        Solver(const Domain& domain, SolverConfiguration configuration);

        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] DenseControl allocate_control(const Domain& domain) const;
        [[nodiscard]] DenseControlTangent allocate_control_tangent(const Domain& domain) const;
        [[nodiscard]] DenseControlAdjoint allocate_control_adjoint(const Domain& domain) const;
        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const;

        void clear(const Domain& domain, State& state) const;
        void clear(const Domain& domain, StateTangent& tangent) const;
        void clear(const Domain& domain, StateAdjoint& adjoint) const;
        void clear(const Domain& domain, DenseControl& control) const;
        void clear(const Domain& domain, DenseControlTangent& tangent) const;
        void clear(const Domain& domain, DenseControlAdjoint& adjoint) const;
        void copy(const Domain& domain, const State& source, State& destination) const;
        void copy(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void accumulate(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;

        void forward(const Domain& domain, const State& state, const DenseControl& control, State& output, StepCache& cache);
        void jvp(const Domain& domain, const State& state, const DenseControl& control, const StepCache& cache, const StateTangent& state_tangent, const DenseControlTangent& control_tangent, StateTangent& output_tangent);
        void vjp(const Domain& domain, const State& state, const DenseControl& control, const StepCache& cache, const StateAdjoint& output_adjoint, StateAdjoint& state_adjoint, DenseControlAdjoint& control_adjoint);

    private:
        struct TangentWorkspace final {
            CenteredVectorField physical_force;
            CenteredVectorField total_force;
            VorticityCache vorticity;
            StaggeredVectorField forced_velocity;
            StaggeredVectorField advected_velocity;
            StaggeredVectorField diffusion_first;
            StaggeredVectorField diffusion_second;
            ScalarField pressure_rhs;
            ScalarField pressure;
            StaggeredVectorField projected_velocity;
            ScalarField advected_density;
            ::cuda::device_buffer<double> input_mass;
            ::cuda::device_buffer<double> advected_mass;
        };

        struct VorticityAdjointWorkspace final {
            CenteredVectorAdjointField centered_velocity;
            CenteredVectorAdjointField vorticity;
            ScalarAdjointField magnitude;
            CenteredVectorAdjointField normal;
        };

        struct AdjointWorkspace final {
            ScalarAdjointField advected_density;
            StaggeredVectorAdjointField projected_velocity;
            ScalarAdjointField pressure;
            ScalarAdjointField pressure_rhs;
            StaggeredVectorAdjointField diffused_velocity;
            StaggeredVectorAdjointField advected_velocity;
            StaggeredVectorAdjointField forced_velocity;
            CenteredVectorAdjointField total_force;
            CenteredVectorAdjointField physical_force;
            VorticityAdjointWorkspace vorticity;
            ::cuda::device_buffer<double> density_dot;
        };

        StaggeredVectorField diffusion_first;
        StaggeredVectorField diffusion_second;
        TangentWorkspace tangent;
        AdjointWorkspace adjoint;
    };
} // namespace physica::fluids::gas::keyframe_smoke
