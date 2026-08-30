module;

#include <physica/cuda.h>

export module physica.fluids.liquid.solvers.pbf;

import std;
export import physica.fluids.liquid.meshfree;
import physica.fluids.liquid.operators.density;
import physica.fluids.liquid.operators.neighborhood;

export namespace physica::fluids::liquid::solvers::pbf {
    struct Solver final {
        struct Configuration final {
            std::uint32_t pressure_iterations{5u};
            std::uint32_t checkpoint_interval{2u};
            Vector3<float> gravity{.x = 0.0F, .y = -9.81F, .z = 0.0F};
        };

        struct State final {
            simulation::VectorField<float> positions;
            simulation::VectorField<float> velocities;
            std::uint64_t step_index{};
        };

        struct StateTangent final {
            simulation::VectorField<float> positions;
            simulation::VectorField<float> velocities;
        };

        struct StateAdjoint final {
            simulation::VectorField<double> positions;
            simulation::VectorField<double> velocities;
        };

        struct Control final {
            simulation::VectorField<float> external_accelerations;
        };

        struct ControlTangent final {
            simulation::VectorField<float> external_accelerations;
        };

        struct ControlAdjoint final {
            simulation::VectorField<double> external_accelerations;
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
            simulation::ScalarField<float> densities;
            simulation::VectorField<float> gradient_sums;
            simulation::ScalarField<float> denominators;
            simulation::ScalarField<float> lambdas;
            simulation::VectorField<float> corrections;
            ::cuda::device_buffer<std::uint32_t> collision_masks;
        };

        struct StepCache final {
            struct IterationCheckpoint final {
                std::uint32_t iteration;
                simulation::VectorField<float> positions;
            };

            operators::Neighborhood neighborhood;
            simulation::VectorField<float> predicted_positions;
            simulation::VectorField<float> corrected_positions;
            std::vector<IterationCheckpoint> checkpoints;
            simulation::VectorField<float> reconstructed_velocities;
            simulation::VectorField<float> vorticities;
            simulation::ScalarField<float> vorticity_magnitudes;
            simulation::VectorField<float> vorticity_normals;
            simulation::ScalarField<float> vorticity_normalizers;
            simulation::VectorField<float> confined_velocities;
        };

        struct Workspace final {
            operators::UniformGridNeighborhood::Workspace neighborhood;
            IterationWorkspace iteration;
        };

        struct TangentWorkspace final {
            simulation::VectorField<float> positions;
            IterationWorkspace iteration;
            simulation::VectorField<float> current_positions;
            simulation::VectorField<float> next_positions;
            simulation::ScalarField<float> densities;
            simulation::VectorField<float> gradient_sums;
            simulation::ScalarField<float> denominators;
            simulation::ScalarField<float> lambdas;
            simulation::VectorField<float> corrections;
            simulation::VectorField<float> reconstructed_velocities;
            simulation::VectorField<float> vorticities;
            simulation::ScalarField<float> vorticity_magnitudes;
            simulation::VectorField<float> vorticity_normals;
            simulation::VectorField<float> confined_velocities;
        };

        struct AdjointWorkspace final {
            std::vector<simulation::VectorField<float>> position_history;
            std::vector<IterationWorkspace> iteration_history;
            simulation::VectorField<double> current_positions;
            simulation::VectorField<double> next_positions;
            simulation::VectorField<double> corrections;
            simulation::ScalarField<double> lambdas;
            simulation::ScalarField<double> densities;
            simulation::VectorField<double> reconstructed_velocities;
            simulation::VectorField<double> vorticities;
            simulation::VectorField<double> vorticity_normals;
            simulation::VectorField<double> confined_velocities;
        };

        explicit Solver(Configuration configuration);

        [[nodiscard]] State allocate_state(const meshfree::Model& model) const;
        [[nodiscard]] Control allocate_control(const meshfree::Model& model) const;
        [[nodiscard]] Parameters allocate_parameters(const meshfree::Model& model) const;
        [[nodiscard]] StepCache allocate_step_cache(const meshfree::Model& model) const;
        [[nodiscard]] Workspace allocate_workspace(const meshfree::Model& model) const;
        [[nodiscard]] StateTangent allocate_state_tangent(const meshfree::Model& model) const;
        [[nodiscard]] ControlTangent allocate_control_tangent(const meshfree::Model& model) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const meshfree::Model& model) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const meshfree::Model& model) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(const meshfree::Model& model) const;
        [[nodiscard]] ControlAdjoint allocate_control_adjoint(const meshfree::Model& model) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const meshfree::Model& model) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const meshfree::Model& model) const;
        void copy_state(const meshfree::Model& model, const State& source, State& destination) const;
        void copy_state_tangent(const meshfree::Model& model, const StateTangent& source, StateTangent& destination) const;
        void copy_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const;
        void accumulate_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const;
        void forward(const meshfree::Model& model, const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const;
        void jvp(const meshfree::Model& model, const State& state, const Parameters& parameters, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const;
        void vjp(const meshfree::Model& model, const State& state, const Parameters& parameters, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
        [[no_unique_address]] operators::UniformGridNeighborhood neighborhood;
        [[no_unique_address]] operators::Poly6Density density;

        [[nodiscard]] IterationWorkspace allocate_iteration_workspace(const meshfree::Model& model) const;
    };
} // namespace physica::fluids::liquid::solvers::pbf
