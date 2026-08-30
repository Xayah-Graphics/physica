module;

#include <physica/cuda.h>

export module physica.fluids.gas.operators.objective;

import std;
import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    struct KeyframeCache final {
        simulation::ScalarField<float> density_residual;
        simulation::ScalarField<float> blurred_density_residual;
        simulation::VectorField<float> velocity_residual;
        simulation::VectorField<float> blurred_velocity_residual;
        ::cuda::device_buffer<double> density_loss;
        ::cuda::device_buffer<double> velocity_loss;
        ::cuda::device_buffer<double> directional_derivative;
    };

    struct StepCache final {
        ::cuda::device_buffer<double> control_effort;
        ::cuda::device_buffer<double> directional_derivative;
    };

    struct Configuration final {
        double control_effort_weight{1.0e-5};
        float blur_sigma_cells{};
    };

    struct Workspace final {
        simulation::ScalarField<float> density_blur_first;
        simulation::ScalarField<float> density_blur_second;
        simulation::ScalarField<float> density_blur_output;
        simulation::VectorField<float> velocity_blur_first;
        simulation::VectorField<float> velocity_blur_second;
        simulation::VectorField<float> velocity_blur_output;
    };

    struct Quadratic final {
        Quadratic(const Domain& domain, Configuration configuration);

        [[nodiscard]] KeyframeCache allocate_keyframe_cache(const Domain& domain) const;
        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] Quadratic with_blur_sigma(const Domain& domain, float sigma_cells) const;
        void evaluate_keyframe(const Domain& domain, const simulation::ScalarField<float>& density, const simulation::VectorField<float>& velocity, const simulation::ScalarField<float>& target_density, const simulation::VectorField<float>& target_velocity, double density_weight, double velocity_weight, KeyframeCache& cache, Workspace& workspace) const;
        void keyframe_jvp(const Domain& domain, const simulation::ScalarField<float>& density_tangent, const simulation::VectorField<float>& velocity_tangent, double density_weight, double velocity_weight, KeyframeCache& cache, Workspace& workspace) const;
        void keyframe_vjp(const Domain& domain, double density_weight, double velocity_weight, const KeyframeCache& cache, simulation::ScalarField<double>& density_adjoint, simulation::VectorField<double>& velocity_adjoint, Workspace& workspace) const;
        void evaluate_control_effort(const Domain& domain, const simulation::VectorField<float>& control, StepCache& cache) const;
        void accumulate(const Domain& domain, const ::cuda::device_buffer<double>& contribution, ::cuda::device_buffer<double>& objective) const;
        void control_effort_jvp(const Domain& domain, const simulation::VectorField<float>& control, const simulation::VectorField<float>& control_tangent, StepCache& cache) const;
        void control_effort_vjp(const Domain& domain, const simulation::VectorField<float>& control, simulation::VectorField<double>& control_adjoint) const;

    private:
        const Configuration configuration;
        const std::uint32_t blur_radius;
        ::cuda::device_buffer<float> blur_weights;
    };
} // namespace physica::fluids::gas::operators
