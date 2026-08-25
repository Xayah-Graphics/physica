module;

#include <physica/cuda.h>

export module physica.fluids.liquid.operators.sph_dynamics;

import std;
import physica.fluids.liquid.domain;
import physica.fluids.liquid.operators.density;
import physica.fluids.liquid.operators.neighborhood;

export namespace physica::fluids::liquid::operators {
    struct ParticleState final {
        VectorField<float> positions;
        VectorField<float> velocities;
        std::uint64_t step_index{};
    };

    struct ParticleStateTangent final {
        VectorField<float> positions;
        VectorField<float> velocities;
    };

    struct ParticleStateAdjoint final {
        VectorField<double> positions;
        VectorField<double> velocities;
    };

    struct Control final {
        VectorField<float> external_accelerations;
    };

    struct ControlTangent final {
        VectorField<float> external_accelerations;
    };

    struct ControlAdjoint final {
        VectorField<double> external_accelerations;
    };

    struct ParticleParameters final {
        ::cuda::device_buffer<float> masses;
        ::cuda::device_buffer<float> rest_densities;
        ::cuda::device_buffer<float> viscosities;
        ::cuda::device_buffer<float> surface_tensions;
    };

    struct ParticleParameterTangent final {
        ::cuda::device_buffer<float> masses;
        ::cuda::device_buffer<float> rest_densities;
        ::cuda::device_buffer<float> viscosities;
        ::cuda::device_buffer<float> surface_tensions;
    };

    struct ParticleParameterAdjoint final {
        ::cuda::device_buffer<double> masses;
        ::cuda::device_buffer<double> rest_densities;
        ::cuda::device_buffer<double> viscosities;
        ::cuda::device_buffer<double> surface_tensions;
    };

    struct PressureIterationCache final {
        std::uint32_t iteration;
        ScalarField<float> pressures;
        ScalarField<float> predicted_densities;
        VectorField<float> pressure_accelerations;
        VectorField<float> predicted_positions;
        VectorField<float> predicted_velocities;
    };

    struct PressureIterationTangent final {
        ScalarField<float> pressures;
        ScalarField<float> predicted_densities;
        VectorField<float> pressure_accelerations;
        VectorField<float> predicted_positions;
        VectorField<float> predicted_velocities;
    };

    struct PressureIterationAdjoint final {
        ScalarField<double> pressures;
        ScalarField<double> predicted_densities;
        VectorField<double> pressure_accelerations;
        VectorField<double> predicted_positions;
        VectorField<double> predicted_velocities;
    };

    struct WeaklyCompressible final {
        struct Configuration final {
            Vector3 gravity{.x = 0.0F, .y = -9.81F, .z = 0.0F};
        };
        struct State final {};
        struct StateTangent final {};
        struct StateAdjoint final {};

        struct Parameters final {
            ::cuda::device_buffer<float> speed_of_sound;
            ::cuda::device_buffer<float> tait_exponent;
            ::cuda::device_buffer<float> boundary_surface_tension;
        };

        struct ParameterTangent final {
            ::cuda::device_buffer<float> speed_of_sound;
            ::cuda::device_buffer<float> tait_exponent;
            ::cuda::device_buffer<float> boundary_surface_tension;
        };

        struct ParameterAdjoint final {
            ::cuda::device_buffer<double> speed_of_sound;
            ::cuda::device_buffer<double> tait_exponent;
            ::cuda::device_buffer<double> boundary_surface_tension;
        };

        struct Cache final {
            ScalarField<float> pressures;
            VectorField<float> pressure_accelerations;
            VectorField<float> viscosity_accelerations;
            VectorField<float> surface_accelerations;
            VectorField<float> external_accelerations;
            VectorField<float> total_accelerations;
        };

        struct Workspace final {};

        struct TangentWorkspace final {
            ScalarField<float> pressures;
            VectorField<float> pressure_accelerations;
            VectorField<float> viscosity_accelerations;
            VectorField<float> surface_accelerations;
            VectorField<float> total_accelerations;
        };

        struct AdjointWorkspace final {
            ScalarField<double> pressures;
            VectorField<double> total_accelerations;
        };

        WeaklyCompressible(const Domain& domain, Configuration configuration);

        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, ParticleState& next_state, State& next_method_state, Cache& cache, Workspace& workspace) const;
        void jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField<float>& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };

    struct PredictiveCorrective final {
        struct Configuration final {
            std::uint32_t pressure_iterations{6u};
            std::uint32_t checkpoint_interval{2u};
            Vector3 gravity{.x = 0.0F, .y = -9.81F, .z = 0.0F};
        };
        struct State final {};
        struct StateTangent final {};
        struct StateAdjoint final {};
        struct Parameters final {
            ::cuda::device_buffer<float> pressure_relaxation;
        };
        struct ParameterTangent final {
            ::cuda::device_buffer<float> pressure_relaxation;
        };
        struct ParameterAdjoint final {
            ::cuda::device_buffer<double> pressure_relaxation;
        };
        struct Cache final {
            VectorField<float> non_pressure_accelerations;
            std::vector<PressureIterationCache> checkpoints;
        };
        struct Workspace final {
            PressureIterationCache primal;
        };
        struct TangentWorkspace final {
            PressureIterationCache primal;
            PressureIterationTangent tangent;
            VectorField<float> non_pressure_accelerations;
        };
        struct AdjointWorkspace final {
            std::vector<PressureIterationCache> recomputed_iterations;
            PressureIterationAdjoint adjoint;
            PressureIterationAdjoint previous_adjoint;
            VectorField<double> non_pressure_accelerations;
        };

        PredictiveCorrective(const Domain& domain, Configuration configuration);
        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, ParticleState& next_state, State& next_method_state, Cache& cache, Workspace& workspace) const;
        void jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField<float>& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
        const float reference_gradient_norm;
        [[no_unique_address]] CubicSplineDensity density;
    };

    struct ImplicitIncompressible final {
        struct Configuration final {
            std::uint32_t pressure_iterations{6u};
            std::uint32_t checkpoint_interval{2u};
            Vector3 gravity{.x = 0.0F, .y = -9.81F, .z = 0.0F};
        };
        struct State final {};
        struct StateTangent final {};
        struct StateAdjoint final {};
        struct Parameters final {
            ::cuda::device_buffer<float> jacobi_relaxation;
        };
        struct ParameterTangent final {
            ::cuda::device_buffer<float> jacobi_relaxation;
        };
        struct ParameterAdjoint final {
            ::cuda::device_buffer<double> jacobi_relaxation;
        };
        struct Cache final {
            VectorField<float> non_pressure_accelerations;
            std::vector<PressureIterationCache> checkpoints;
        };
        struct Workspace final {
            PressureIterationCache primal;
        };
        struct TangentWorkspace final {
            PressureIterationCache primal;
            PressureIterationTangent tangent;
            VectorField<float> non_pressure_accelerations;
        };
        struct AdjointWorkspace final {
            std::vector<PressureIterationCache> recomputed_iterations;
            PressureIterationAdjoint adjoint;
            PressureIterationAdjoint previous_adjoint;
            VectorField<double> non_pressure_accelerations;
        };

        ImplicitIncompressible(const Domain& domain, Configuration configuration);
        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, ParticleState& next_state, State& next_method_state, Cache& cache, Workspace& workspace) const;
        void jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField<float>& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
        const float reference_gradient_norm;
        [[no_unique_address]] CubicSplineDensity density;
    };

    struct DivergenceFree final {
        struct Configuration final {
            std::uint32_t divergence_iterations{4u};
            std::uint32_t density_iterations{6u};
            std::uint32_t checkpoint_interval{2u};
            bool pressure_warm_start{true};
            Vector3 gravity{.x = 0.0F, .y = -9.81F, .z = 0.0F};
        };
        struct State final {
            ScalarField<float> warm_divergence_pressure;
            ScalarField<float> warm_density_pressure;
        };
        struct StateTangent final {
            ScalarField<float> warm_divergence_pressure;
            ScalarField<float> warm_density_pressure;
        };
        struct StateAdjoint final {
            ScalarField<double> warm_divergence_pressure;
            ScalarField<double> warm_density_pressure;
        };
        struct Parameters final {
            ::cuda::device_buffer<float> divergence_relaxation;
            ::cuda::device_buffer<float> density_relaxation;
        };
        struct ParameterTangent final {
            ::cuda::device_buffer<float> divergence_relaxation;
            ::cuda::device_buffer<float> density_relaxation;
        };
        struct ParameterAdjoint final {
            ::cuda::device_buffer<double> divergence_relaxation;
            ::cuda::device_buffer<double> density_relaxation;
        };
        struct Cache final {
            VectorField<float> non_pressure_accelerations;
            std::vector<PressureIterationCache> divergence_checkpoints;
            VectorField<float> divergence_pressure_accelerations;
            std::vector<PressureIterationCache> density_checkpoints;
            VectorField<float> total_pressure_accelerations;
        };
        struct Workspace final {
            PressureIterationCache primal;
            VectorField<float> total_pressure_accelerations;
        };
        struct TangentWorkspace final {
            PressureIterationCache primal;
            PressureIterationTangent tangent;
            VectorField<float> non_pressure_accelerations;
            VectorField<float> divergence_pressure_accelerations;
            VectorField<float> primal_total_pressure_accelerations;
            VectorField<float> tangent_total_pressure_accelerations;
        };
        struct AdjointWorkspace final {
            std::vector<PressureIterationCache> recomputed_iterations;
            PressureIterationAdjoint adjoint;
            PressureIterationAdjoint previous_adjoint;
            VectorField<float> total_pressure_accelerations;
            ScalarField<double> target_densities;
            VectorField<double> non_pressure_accelerations;
            VectorField<double> divergence_pressure_accelerations;
            VectorField<double> total_pressure_accelerations_adjoint;
        };

        DivergenceFree(const Domain& domain, Configuration configuration);
        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, ParticleState& next_state, State& next_method_state, Cache& cache, Workspace& workspace) const;
        void jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField<float>& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<float>& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
        const float reference_gradient_norm;
        [[no_unique_address]] CubicSplineDensity density;
    };

    template <class Algorithm>
    concept SPHDynamicsAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, const ParticleState& state, const ParticleStateTangent& state_tangent, const ParticleStateAdjoint& next_state_adjoint, ParticleState& next_state, ParticleStateTangent& next_state_tangent, ParticleStateAdjoint& previous_state_adjoint, const Control& control, const ControlTangent& control_tangent, ControlAdjoint& control_adjoint, const ParticleParameters& particles, const ParticleParameterTangent& particle_tangent, ParticleParameterAdjoint& particle_adjoint, const Neighborhood& neighborhood, const ScalarField<float>& densities, const ScalarField<float>& density_tangent, ScalarField<double>& density_adjoint, const typename Algorithm::State& method_state, typename Algorithm::State& next_method_state, const typename Algorithm::StateTangent& method_state_tangent, typename Algorithm::StateTangent& next_method_state_tangent, const typename Algorithm::StateAdjoint& next_method_state_adjoint, typename Algorithm::StateAdjoint& previous_method_state_adjoint, const typename Algorithm::Parameters& parameters, const typename Algorithm::ParameterTangent& parameter_tangent, typename Algorithm::ParameterAdjoint& parameter_adjoint, typename Algorithm::Cache& cache, const typename Algorithm::Cache& constant_cache, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace) {
        { algorithm.allocate_state(domain) } -> std::same_as<typename Algorithm::State>;
        { algorithm.allocate_state_tangent(domain) } -> std::same_as<typename Algorithm::StateTangent>;
        { algorithm.allocate_state_adjoint(domain) } -> std::same_as<typename Algorithm::StateAdjoint>;
        { algorithm.allocate_parameters(domain) } -> std::same_as<typename Algorithm::Parameters>;
        { algorithm.allocate_parameter_tangent(domain) } -> std::same_as<typename Algorithm::ParameterTangent>;
        { algorithm.allocate_parameter_adjoint(domain) } -> std::same_as<typename Algorithm::ParameterAdjoint>;
        { algorithm.allocate_cache(domain) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.copy_state(domain, method_state, next_method_state);
        algorithm.copy_state_tangent(domain, method_state_tangent, next_method_state_tangent);
        algorithm.copy_state_adjoint(domain, next_method_state_adjoint, previous_method_state_adjoint);
        algorithm.accumulate_state_adjoint(domain, next_method_state_adjoint, previous_method_state_adjoint);
        algorithm.forward(domain, state, method_state, control, particles, parameters, neighborhood, densities, next_state, next_method_state, cache, workspace);
        algorithm.jvp(domain, state, method_state, particles, parameters, neighborhood, densities, constant_cache, state_tangent, method_state_tangent, control_tangent, particle_tangent, parameter_tangent, density_tangent, next_state_tangent, next_method_state_tangent, tangent_workspace);
        algorithm.vjp(domain, state, method_state, particles, parameters, neighborhood, densities, constant_cache, next_state_adjoint, next_method_state_adjoint, previous_state_adjoint, previous_method_state_adjoint, control_adjoint, particle_adjoint, parameter_adjoint, density_adjoint, adjoint_workspace);
    };
} // namespace physica::fluids::liquid::operators
