module;

#include <physica/cuda.h>

export module physica.fluids.gas.adjoint_control.objective;

import std;
export import physica.fluids.gas.adjoint_control.solver;

export namespace physica::fluids::gas::adjoint_control {
    struct Keyframe final {
        std::uint32_t step{};
        State target;
        double density_weight{1.0};
        double velocity_weight{};
    };

    struct Problem final {
        std::uint32_t begin_step;
        std::uint32_t step_count;
        State initial_state;
        std::vector<Keyframe> keyframes;

        Problem(std::uint32_t step_count, State initial_state, std::vector<Keyframe> keyframes);
        Problem(std::uint32_t begin_step, std::uint32_t step_count, State initial_state, std::vector<Keyframe> keyframes);
        ~Problem();

        Problem(Problem&&) noexcept;
        Problem& operator=(Problem&&) noexcept;
        Problem(const Problem&) = delete;
        Problem& operator=(const Problem&) = delete;
    };

    struct KeyframeCache final {
        ScalarField density_residual;
        ScalarField blurred_density_residual;
        StaggeredVectorField velocity_residual;
        StaggeredVectorField blurred_velocity_residual;
        ::cuda::device_buffer<double> density_loss;
        ::cuda::device_buffer<double> velocity_loss;
        ::cuda::device_buffer<double> directional_derivative;
    };

    struct StepObjectiveCache final {
        ::cuda::device_buffer<double> control_effort;
        ::cuda::device_buffer<double> directional_derivative;
    };

    struct StepMetrics final {
        double input_mass{};
        double advected_mass{};
        double control_effort{};
        double directional_derivative{};
    };

    struct KeyframeMetrics final {
        double density_loss{};
        double velocity_loss{};
        double directional_derivative{};
    };

    struct StepRecord final {
        std::uint32_t step{};
        StepMetrics metrics;
    };

    struct KeyframeRecord final {
        std::size_t keyframe_index{};
        KeyframeMetrics metrics;
    };

    struct ObjectiveConfiguration final {
        double control_effort_weight{1.0e-5};
        float blur_sigma_cells{};
    };

    struct Objective final {
        ObjectiveConfiguration configuration;

        Objective(const Domain& domain, ObjectiveConfiguration configuration);

        [[nodiscard]] KeyframeCache allocate_keyframe_cache(const Domain& domain) const;
        [[nodiscard]] StepObjectiveCache allocate_step_objective_cache(const Domain& domain) const;
        void set_blur_sigma(const Domain& domain, float sigma_cells);
        void evaluate_keyframe(const Domain& domain, const State& state, const Keyframe& keyframe, KeyframeCache& cache);
        void keyframe_jvp(const Domain& domain, const StateTangent& state_tangent, const Keyframe& keyframe, KeyframeCache& cache);
        void keyframe_vjp(const Domain& domain, const Keyframe& keyframe, const KeyframeCache& cache, StateAdjoint& state_adjoint);
        void evaluate_control_effort(const Domain& domain, const DenseControl& control, StepObjectiveCache& cache) const;
        void control_effort_jvp(const Domain& domain, const DenseControl& control, const DenseControlTangent& control_tangent, StepObjectiveCache& cache) const;
        void control_effort_vjp(const Domain& domain, const DenseControl& control, DenseControlAdjoint& control_adjoint) const;

    private:
        std::uint32_t blur_radius;
        ::cuda::device_buffer<float> blur_weights;
        State first;
        State second;
        State third;
    };
} // namespace physica::fluids::gas::adjoint_control
