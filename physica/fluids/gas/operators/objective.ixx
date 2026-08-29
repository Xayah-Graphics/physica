module;

#include <physica/cuda.h>

export module physica.fluids.gas.operators.objective;

import std;
import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    struct KeyframeCache final {
        ScalarField<float> density_residual;
        ScalarField<float> blurred_density_residual;
        VectorField<float> velocity_residual;
        VectorField<float> blurred_velocity_residual;
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
        ScalarField<float> density_blur_first;
        ScalarField<float> density_blur_second;
        ScalarField<float> density_blur_output;
        VectorField<float> velocity_blur_first;
        VectorField<float> velocity_blur_second;
        VectorField<float> velocity_blur_output;
    };

    struct Quadratic final {
        Quadratic(const Domain& domain, Configuration configuration);

        [[nodiscard]] KeyframeCache allocate_keyframe_cache(const Domain& domain) const;
        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] Quadratic with_blur_sigma(const Domain& domain, float sigma_cells) const;
        void evaluate_keyframe(const Domain& domain, const ScalarField<float>& density, const VectorField<float>& velocity, const ScalarField<float>& target_density, const VectorField<float>& target_velocity, double density_weight, double velocity_weight, KeyframeCache& cache, Workspace& workspace) const;
        void keyframe_jvp(const Domain& domain, const ScalarField<float>& density_tangent, const VectorField<float>& velocity_tangent, double density_weight, double velocity_weight, KeyframeCache& cache, Workspace& workspace) const;
        void keyframe_vjp(const Domain& domain, double density_weight, double velocity_weight, const KeyframeCache& cache, ScalarField<double>& density_adjoint, VectorField<double>& velocity_adjoint, Workspace& workspace) const;
        void evaluate_control_effort(const Domain& domain, const VectorField<float>& control, StepCache& cache) const;
        void accumulate(const Domain& domain, const ::cuda::device_buffer<double>& contribution, ::cuda::device_buffer<double>& objective) const;
        void control_effort_jvp(const Domain& domain, const VectorField<float>& control, const VectorField<float>& control_tangent, StepCache& cache) const;
        void control_effort_vjp(const Domain& domain, const VectorField<float>& control, VectorField<double>& control_adjoint) const;

    private:
        const Configuration configuration;
        const std::uint32_t blur_radius;
        ::cuda::device_buffer<float> blur_weights;
    };
} // namespace physica::fluids::gas::operators
