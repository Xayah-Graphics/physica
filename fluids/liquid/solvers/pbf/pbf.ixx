module;

#include <physica/cuda.h>

export module physica.fluids.liquid.pbf;

import std;
import physica.fluids.liquid.domain;
import physica.fluids.liquid.operators.density;
import physica.fluids.liquid.operators.neighborhood;

export namespace physica::fluids::liquid::pbf {
    struct Solver final {
        struct Configuration final {
            std::uint32_t pressure_iterations{5u};
            std::uint32_t checkpoint_interval{2u};
            Vector3 gravity{.x = 0.0F, .y = -9.81F, .z = 0.0F};
        };

        struct State final {
            VectorField<float> positions;
            VectorField<float> velocities;
            std::uint64_t step_index{};
        };

        struct StateTangent final {
            VectorField<float> positions;
            VectorField<float> velocities;
        };

        struct StateAdjoint final {
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

        struct Parameters final {
            ::cuda::device_buffer<float> masses;
            ::cuda::device_buffer<float> rest_densities;
            ::cuda::device_buffer<float> viscosities;
            ::cuda::device_buffer<float> surface_tensions;
            ::cuda::device_buffer<float> relaxation;
            ::cuda::device_buffer<float> artificial_pressure_strength;
            ::cuda::device_buffer<float> artificial_pressure_exponent;
            ::cuda::device_buffer<float> artificial_pressure_radius;
            ::cuda::device_buffer<float> xsph_viscosity;
            ::cuda::device_buffer<float> vorticity_confinement;
        };

        struct ParameterTangent final {
            ::cuda::device_buffer<float> masses;
            ::cuda::device_buffer<float> rest_densities;
            ::cuda::device_buffer<float> viscosities;
            ::cuda::device_buffer<float> surface_tensions;
            ::cuda::device_buffer<float> relaxation;
            ::cuda::device_buffer<float> artificial_pressure_strength;
            ::cuda::device_buffer<float> artificial_pressure_exponent;
            ::cuda::device_buffer<float> artificial_pressure_radius;
            ::cuda::device_buffer<float> xsph_viscosity;
            ::cuda::device_buffer<float> vorticity_confinement;
        };

        struct ParameterAdjoint final {
            ::cuda::device_buffer<double> masses;
            ::cuda::device_buffer<double> rest_densities;
            ::cuda::device_buffer<double> viscosities;
            ::cuda::device_buffer<double> surface_tensions;
            ::cuda::device_buffer<double> relaxation;
            ::cuda::device_buffer<double> artificial_pressure_strength;
            ::cuda::device_buffer<double> artificial_pressure_exponent;
            ::cuda::device_buffer<double> artificial_pressure_radius;
            ::cuda::device_buffer<double> xsph_viscosity;
            ::cuda::device_buffer<double> vorticity_confinement;
        };

        struct IterationWorkspace final {
            ScalarField<float> densities;
            VectorField<float> gradient_sums;
            ScalarField<float> denominators;
            ScalarField<float> lambdas;
            VectorField<float> corrections;
            ::cuda::device_buffer<std::uint32_t> collision_masks;
        };

        struct StepCache final {
            struct IterationCheckpoint final {
                std::uint32_t iteration;
                VectorField<float> positions;
            };

            operators::Neighborhood neighborhood;
            VectorField<float> predicted_positions;
            VectorField<float> corrected_positions;
            std::vector<IterationCheckpoint> checkpoints;
            VectorField<float> reconstructed_velocities;
            VectorField<float> vorticities;
            ScalarField<float> vorticity_magnitudes;
            VectorField<float> vorticity_normals;
            ScalarField<float> vorticity_normalizers;
            VectorField<float> confined_velocities;
        };

        struct Workspace final {
            operators::UniformGridNeighborhood::Workspace neighborhood;
            IterationWorkspace iteration;
        };

        struct TangentWorkspace final {
            VectorField<float> positions;
            IterationWorkspace iteration;
            VectorField<float> current_positions;
            VectorField<float> next_positions;
            ScalarField<float> densities;
            VectorField<float> gradient_sums;
            ScalarField<float> denominators;
            ScalarField<float> lambdas;
            VectorField<float> corrections;
            VectorField<float> reconstructed_velocities;
            VectorField<float> vorticities;
            ScalarField<float> vorticity_magnitudes;
            VectorField<float> vorticity_normals;
            VectorField<float> confined_velocities;
        };

        struct AdjointWorkspace final {
            std::vector<VectorField<float>> position_history;
            std::vector<IterationWorkspace> iteration_history;
            VectorField<double> current_positions;
            VectorField<double> next_positions;
            VectorField<double> corrections;
            ScalarField<double> lambdas;
            ScalarField<double> densities;
            VectorField<double> reconstructed_velocities;
            VectorField<double> vorticities;
            VectorField<double> vorticity_normals;
            VectorField<double> confined_velocities;
        };

        explicit Solver(Configuration configuration);

        [[nodiscard]] State allocate_state(const Domain& domain) const;
        [[nodiscard]] Control allocate_control(const Domain& domain) const;
        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const;
        [[nodiscard]] ControlTangent allocate_control_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const;
        [[nodiscard]] ControlAdjoint allocate_control_adjoint(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;
        void copy_state(const Domain& domain, const State& source, State& destination) const;
        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const Domain& domain, const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const;
        void jvp(const Domain& domain, const State& state, const Parameters& parameters, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const State& state, const Parameters& parameters, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
        [[no_unique_address]] operators::UniformGridNeighborhood neighborhood;
        [[no_unique_address]] operators::Poly6Density density;

        [[nodiscard]] IterationWorkspace allocate_iteration_workspace(const Domain& domain) const;
    };
} // namespace physica::fluids::liquid::pbf
