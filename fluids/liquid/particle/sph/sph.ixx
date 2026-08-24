module;

#include <physica/cuda.h>

export module physica.fluids.liquid.particle.sph;

import std;
import physica.fluids.liquid.particle.density;
import physica.fluids.liquid.particle.domain;
import physica.fluids.liquid.particle.neighborhood;

export namespace physica::fluids::liquid::particle {
    struct PressureIterationCache final {
        std::uint32_t iteration;
        ScalarField pressures;
        ScalarField predicted_densities;
        VectorField pressure_accelerations;
        VectorField predicted_positions;
        VectorField predicted_velocities;
    };

    struct PressureIterationTangent final {
        ScalarField pressures;
        ScalarField predicted_densities;
        VectorField pressure_accelerations;
        VectorField predicted_positions;
        VectorField predicted_velocities;
    };

    struct PressureIterationAdjoint final {
        ScalarAdjointField pressures;
        ScalarAdjointField predicted_densities;
        VectorAdjointField pressure_accelerations;
        VectorAdjointField predicted_positions;
        VectorAdjointField predicted_velocities;
    };

    struct WCSPH final {
        struct Configuration final {};
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
            ScalarField pressures;
            VectorField pressure_accelerations;
            VectorField viscosity_accelerations;
            VectorField surface_accelerations;
            VectorField external_accelerations;
            VectorField total_accelerations;
        };

        struct Differentiation final {
            ScalarField pressure_tangent;
            VectorField pressure_acceleration_tangent;
            VectorField viscosity_acceleration_tangent;
            VectorField surface_acceleration_tangent;
            VectorField total_acceleration_tangent;
            ScalarAdjointField pressure_adjoint;
            VectorAdjointField total_acceleration_adjoint;
        };

        const Configuration configuration;
        std::optional<Differentiation> differentiation;

        WCSPH(const Domain& domain, Configuration configuration, ExecutionMode mode);

        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, ParticleState& next_state, State& next_method_state, Cache& cache);
        void jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent);
        void vjp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarAdjointField& density_adjoint);
    };

    struct PCISPH final {
        struct Configuration final {
            std::uint32_t pressure_iterations{6u};
            std::uint32_t checkpoint_interval{2u};
        };
        struct State final {};
        struct StateTangent final {};
        struct StateAdjoint final {};
        struct Parameters final { ::cuda::device_buffer<float> pressure_relaxation; };
        struct ParameterTangent final { ::cuda::device_buffer<float> pressure_relaxation; };
        struct ParameterAdjoint final { ::cuda::device_buffer<double> pressure_relaxation; };
        struct Cache final {
            VectorField non_pressure_accelerations;
            std::vector<PressureIterationCache> checkpoints;
        };
        struct Differentiation final {
            PressureIterationTangent tangent;
            PressureIterationAdjoint adjoint;
            PressureIterationAdjoint previous_adjoint;
            VectorField non_pressure_acceleration_tangent;
            VectorAdjointField non_pressure_acceleration_adjoint;
        };

        const Configuration configuration;
        const float reference_gradient_norm;
        PressureIterationCache primal;
        std::vector<PressureIterationCache> recomputed_iterations;
        std::optional<Differentiation> differentiation;

        PCISPH(const Domain& domain, Configuration configuration, ExecutionMode mode);
        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, ParticleState& next_state, State& next_method_state, Cache& cache);
        void jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent);
        void vjp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarAdjointField& density_adjoint);
    };

    struct IISPH final {
        struct Configuration final {
            std::uint32_t pressure_iterations{6u};
            std::uint32_t checkpoint_interval{2u};
        };
        struct State final {};
        struct StateTangent final {};
        struct StateAdjoint final {};
        struct Parameters final { ::cuda::device_buffer<float> jacobi_relaxation; };
        struct ParameterTangent final { ::cuda::device_buffer<float> jacobi_relaxation; };
        struct ParameterAdjoint final { ::cuda::device_buffer<double> jacobi_relaxation; };
        struct Cache final {
            VectorField non_pressure_accelerations;
            std::vector<PressureIterationCache> checkpoints;
        };
        struct Differentiation final {
            PressureIterationTangent tangent;
            PressureIterationAdjoint adjoint;
            PressureIterationAdjoint previous_adjoint;
            VectorField non_pressure_acceleration_tangent;
            VectorAdjointField non_pressure_acceleration_adjoint;
        };

        const Configuration configuration;
        const float reference_gradient_norm;
        PressureIterationCache primal;
        std::vector<PressureIterationCache> recomputed_iterations;
        std::optional<Differentiation> differentiation;

        IISPH(const Domain& domain, Configuration configuration, ExecutionMode mode);
        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, ParticleState& next_state, State& next_method_state, Cache& cache);
        void jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent);
        void vjp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarAdjointField& density_adjoint);
    };

    struct DFSPH final {
        struct Configuration final {
            std::uint32_t divergence_iterations{4u};
            std::uint32_t density_iterations{6u};
            std::uint32_t checkpoint_interval{2u};
            bool pressure_warm_start{true};
        };
        struct State final {
            ScalarField warm_divergence_pressure;
            ScalarField warm_density_pressure;
        };
        struct StateTangent final {
            ScalarField warm_divergence_pressure;
            ScalarField warm_density_pressure;
        };
        struct StateAdjoint final {
            ScalarAdjointField warm_divergence_pressure;
            ScalarAdjointField warm_density_pressure;
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
            VectorField non_pressure_accelerations;
            std::vector<PressureIterationCache> divergence_checkpoints;
            VectorField divergence_pressure_accelerations;
            std::vector<PressureIterationCache> density_checkpoints;
            VectorField total_pressure_accelerations;
        };
        struct Differentiation final {
            PressureIterationTangent tangent;
            PressureIterationAdjoint adjoint;
            PressureIterationAdjoint previous_adjoint;
            VectorField non_pressure_acceleration_tangent;
            VectorField divergence_pressure_acceleration_tangent;
            VectorField total_pressure_acceleration_tangent;
            ScalarAdjointField target_density_adjoint;
            VectorAdjointField non_pressure_acceleration_adjoint;
            VectorAdjointField divergence_pressure_acceleration_adjoint;
            VectorAdjointField total_pressure_acceleration_adjoint;
        };

        const Configuration configuration;
        const float reference_gradient_norm;
        PressureIterationCache primal;
        std::vector<PressureIterationCache> recomputed_iterations;
        VectorField total_pressure_acceleration;
        std::optional<Differentiation> differentiation;

        DFSPH(const Domain& domain, Configuration configuration, ExecutionMode mode);
        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, ParticleState& next_state, State& next_method_state, Cache& cache);
        void jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent);
        void vjp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarAdjointField& density_adjoint);
    };

    template<class Method>
    concept SPHMethod = std::same_as<Method, WCSPH> || std::same_as<Method, PCISPH> || std::same_as<Method, IISPH> || std::same_as<Method, DFSPH>;

    template<SPHMethod Method>
    struct SPH final {
        struct State final {
            ParticleState particles;
            [[no_unique_address]] typename Method::State method;
        };
        struct StateTangent final {
            ParticleStateTangent particles;
            [[no_unique_address]] typename Method::StateTangent method;
        };
        struct StateAdjoint final {
            ParticleStateAdjoint particles;
            [[no_unique_address]] typename Method::StateAdjoint method;
        };
        struct Parameters final {
            ParticleParameters particles;
            typename Method::Parameters method;
        };
        struct ParameterTangent final {
            ParticleParameterTangent particles;
            typename Method::ParameterTangent method;
        };
        struct ParameterAdjoint final {
            ParticleParameterAdjoint particles;
            typename Method::ParameterAdjoint method;
        };
        struct StepCache final {
            Neighborhood neighborhood;
            ScalarField densities;
            typename Method::Cache method;
        };
        struct Differentiation final {
            ScalarField density_tangent;
            ScalarAdjointField density_adjoint;
        };

        SPH(DomainConfiguration domain_configuration, typename Method::Configuration method_configuration, const ExecutionMode mode, const ::cuda::stream_ref stream)
            : domain(std::move(domain_configuration), stream), neighborhood_search(domain), method(domain, std::move(method_configuration), mode) {
            if (mode == ExecutionMode::differentiable) differentiation.emplace(Differentiation{.density_tangent = domain.allocate_scalar_field(domain.configuration.particle_count), .density_adjoint = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count)});
        }

        SPH(const SPH&) = delete;
        SPH& operator=(const SPH&) = delete;
        SPH(SPH&&) = delete;
        SPH& operator=(SPH&&) = delete;

        [[nodiscard]] State allocate_state() const { return {.particles = domain.allocate_particle_state(), .method = method.allocate_state(domain)}; }
        [[nodiscard]] Control allocate_control() const { return domain.allocate_control(); }
        [[nodiscard]] Parameters allocate_parameters() const { return {.particles = domain.allocate_particle_parameters(), .method = method.allocate_parameters(domain)}; }
        [[nodiscard]] StepCache allocate_step_cache() const { return {.neighborhood = neighborhood_search.allocate(), .densities = domain.allocate_scalar_field(domain.configuration.particle_count), .method = method.allocate_cache(domain)}; }
        [[nodiscard]] StateTangent allocate_state_tangent() const { return {.particles = domain.allocate_particle_state_tangent(), .method = method.allocate_state_tangent(domain)}; }
        [[nodiscard]] ControlTangent allocate_control_tangent() const { return domain.allocate_control_tangent(); }
        [[nodiscard]] ParameterTangent allocate_parameter_tangent() const { return {.particles = domain.allocate_particle_parameter_tangent(), .method = method.allocate_parameter_tangent(domain)}; }
        [[nodiscard]] StateAdjoint allocate_state_adjoint() const { return {.particles = domain.allocate_particle_state_adjoint(), .method = method.allocate_state_adjoint(domain)}; }
        [[nodiscard]] ControlAdjoint allocate_control_adjoint() const { return domain.allocate_control_adjoint(); }
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint() const { return {.particles = domain.allocate_particle_parameter_adjoint(), .method = method.allocate_parameter_adjoint(domain)}; }

        void copy_state(const State& source, State& destination) const {
            domain.copy(source.particles, destination.particles);
            method.copy_state(domain, source.method, destination.method);
        }

        void copy_state_tangent(const StateTangent& source, StateTangent& destination) const {
            domain.copy(source.particles, destination.particles);
            method.copy_state_tangent(domain, source.method, destination.method);
        }

        void copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const {
            domain.copy(source.particles, destination.particles);
            method.copy_state_adjoint(domain, source.method, destination.method);
        }

        void accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const {
            domain.accumulate(source.particles, destination.particles);
            method.accumulate_state_adjoint(domain, source.method, destination.method);
        }

        void forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache) {
            neighborhood_search.build(state.particles.step_index, state.particles.positions, cache.neighborhood);
            density::sph_forward(domain, state.particles.positions, state.particles.positions, parameters.particles, cache.neighborhood, cache.densities);
            method.forward(domain, state.particles, state.method, control, parameters.particles, parameters.method, cache.neighborhood, cache.densities, next_state.particles, next_state.method, cache.method);
        }

        void jvp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent) {
            density::sph_jvp(domain, state.particles.positions, state.particles.positions, state_tangent.particles.positions, parameters.particles, parameter_tangent.particles, cache.neighborhood, differentiation->density_tangent);
            method.jvp(domain, state.particles, state.method, parameters.particles, parameters.method, cache.neighborhood, cache.densities, cache.method, state_tangent.particles, state_tangent.method, control_tangent, parameter_tangent.particles, parameter_tangent.method, differentiation->density_tangent, next_state_tangent.particles, next_state_tangent.method);
        }

        void vjp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint) {
            domain.clear(differentiation->density_adjoint);
            method.vjp(domain, state.particles, state.method, parameters.particles, parameters.method, cache.neighborhood, cache.densities, cache.method, next_state_adjoint.particles, next_state_adjoint.method, previous_state_adjoint.particles, previous_state_adjoint.method, control_adjoint, parameter_adjoint.particles, parameter_adjoint.method, differentiation->density_adjoint);
            density::sph_vjp(domain, state.particles.positions, state.particles.positions, parameters.particles, cache.neighborhood, differentiation->density_adjoint, previous_state_adjoint.particles.positions, parameter_adjoint.particles);
        }

    private:
        Domain domain;
        NeighborhoodSearch neighborhood_search;
        Method method;
        std::optional<Differentiation> differentiation;
    };

    template struct SPH<WCSPH>;
    template struct SPH<PCISPH>;
    template struct SPH<IISPH>;
    template struct SPH<DFSPH>;
} // namespace physica::fluids::liquid::particle
