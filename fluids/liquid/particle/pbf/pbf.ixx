module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

export module physica.fluids.liquid.particle.pbf;

import std;
import physica.fluids.liquid.particle.density;
import physica.fluids.liquid.particle.domain;
import physica.fluids.liquid.particle.neighborhood;

export namespace physica::fluids::liquid::particle {
    struct PBF final {
        struct Configuration final {
            std::uint32_t pressure_iterations{5u};
            std::uint32_t checkpoint_interval{2u};
        };

        struct State final { ParticleState particles; };
        struct StateTangent final { ParticleStateTangent particles; };
        struct StateAdjoint final { ParticleStateAdjoint particles; };

        struct Parameters final {
            ParticleParameters particles;
            ::cuda::device_buffer<float> relaxation;
            ::cuda::device_buffer<float> artificial_pressure_strength;
            ::cuda::device_buffer<float> artificial_pressure_exponent;
            ::cuda::device_buffer<float> artificial_pressure_radius;
            ::cuda::device_buffer<float> xsph_viscosity;
            ::cuda::device_buffer<float> vorticity_confinement;
        };

        struct ParameterTangent final {
            ParticleParameterTangent particles;
            ::cuda::device_buffer<float> relaxation;
            ::cuda::device_buffer<float> artificial_pressure_strength;
            ::cuda::device_buffer<float> artificial_pressure_exponent;
            ::cuda::device_buffer<float> artificial_pressure_radius;
            ::cuda::device_buffer<float> xsph_viscosity;
            ::cuda::device_buffer<float> vorticity_confinement;
        };

        struct ParameterAdjoint final {
            ParticleParameterAdjoint particles;
            ::cuda::device_buffer<double> relaxation;
            ::cuda::device_buffer<double> artificial_pressure_strength;
            ::cuda::device_buffer<double> artificial_pressure_exponent;
            ::cuda::device_buffer<double> artificial_pressure_radius;
            ::cuda::device_buffer<double> xsph_viscosity;
            ::cuda::device_buffer<double> vorticity_confinement;
        };

        struct StepCache final {
            struct IterationCheckpoint final {
                std::uint32_t iteration;
                VectorField positions;
            };

            Neighborhood neighborhood;
            VectorField predicted_positions;
            VectorField corrected_positions;
            std::vector<IterationCheckpoint> checkpoints;
            VectorField reconstructed_velocities;
            VectorField vorticities;
            ScalarField vorticity_magnitudes;
            VectorField vorticity_normals;
            ScalarField vorticity_normalizers;
            VectorField confined_velocities;
        };

        PBF(DomainConfiguration domain_configuration, Configuration configuration, ExecutionMode mode, ::cuda::stream_ref stream);

        PBF(const PBF&) = delete;
        PBF& operator=(const PBF&) = delete;
        PBF(PBF&&) = delete;
        PBF& operator=(PBF&&) = delete;

        [[nodiscard]] State allocate_state() const;
        [[nodiscard]] Control allocate_control() const;
        [[nodiscard]] Parameters allocate_parameters() const;
        [[nodiscard]] StepCache allocate_step_cache() const;
        [[nodiscard]] StateTangent allocate_state_tangent() const;
        [[nodiscard]] ControlTangent allocate_control_tangent() const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent() const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint() const;
        [[nodiscard]] ControlAdjoint allocate_control_adjoint() const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint() const;
        void copy_state(const State& source, State& destination) const;
        void copy_state_tangent(const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const;
        void forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache);
        void jvp_step(const State& state, const Control& control, const Parameters& parameters, const State& next_state, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent);
        void vjp_step(const State& state, const Control& control, const Parameters& parameters, const State& next_state, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint);

    private:
        struct IterationPrimal final {
            ScalarField densities;
            VectorField gradient_sums;
            ScalarField denominators;
            ScalarField lambdas;
            VectorField corrections;
            ::cuda::device_buffer<std::uint32_t> collision_masks;
        };

        struct Differentiation final {
            std::vector<VectorField> segment_position_history;
            std::vector<IterationPrimal> segment_iteration_history;
            VectorField current_position_tangent;
            VectorField next_position_tangent;
            ScalarField density_tangent;
            VectorField gradient_sum_tangent;
            ScalarField denominator_tangent;
            ScalarField lambda_tangent;
            VectorField correction_tangent;
            VectorField reconstructed_velocity_tangent;
            VectorField vorticity_tangent;
            ScalarField vorticity_magnitude_tangent;
            VectorField vorticity_normal_tangent;
            VectorField confined_velocity_tangent;
            VectorAdjointField current_position_adjoint;
            VectorAdjointField next_position_adjoint;
            VectorAdjointField correction_adjoint;
            ScalarAdjointField lambda_adjoint;
            ScalarAdjointField density_adjoint;
            VectorAdjointField reconstructed_velocity_adjoint;
            VectorAdjointField vorticity_adjoint;
            ScalarAdjointField vorticity_magnitude_adjoint;
            VectorAdjointField vorticity_normal_adjoint;
            VectorAdjointField confined_velocity_adjoint;
        };

        const Configuration configuration;
        Domain domain;
        NeighborhoodSearch neighborhood_search;
        IterationPrimal forward_iteration;
        std::optional<Differentiation> differentiation;

        [[nodiscard]] IterationPrimal allocate_iteration() const;
    };
} // namespace physica::fluids::liquid::particle
