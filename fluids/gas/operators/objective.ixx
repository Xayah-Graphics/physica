module;

#include <physica/cuda.h>

export module physica.fluids.gas.operators.objective;

import std;
import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    struct KeyframeCache final {
        CellField<float> density_residual;
        CellField<float> blurred_density_residual;
        StaggeredVectorField<float> velocity_residual;
        StaggeredVectorField<float> blurred_velocity_residual;
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
        CellField<float> density_blur_first;
        CellField<float> density_blur_second;
        CellField<float> density_blur_output;
        StaggeredVectorField<float> velocity_blur_first;
        StaggeredVectorField<float> velocity_blur_second;
        StaggeredVectorField<float> velocity_blur_output;
    };

    struct Quadratic final {
        Quadratic(const Domain& domain, Configuration configuration);

        [[nodiscard]] KeyframeCache allocate_keyframe_cache(const Domain& domain) const;
        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] Quadratic with_blur_sigma(const Domain& domain, float sigma_cells) const;
        void evaluate_keyframe(const Domain& domain, const CellField<float>& density, const StaggeredVectorField<float>& velocity, const CellField<float>& target_density, const StaggeredVectorField<float>& target_velocity, double density_weight, double velocity_weight, KeyframeCache& cache, Workspace& workspace) const;
        void keyframe_jvp(const Domain& domain, const CellField<float>& density_tangent, const StaggeredVectorField<float>& velocity_tangent, double density_weight, double velocity_weight, KeyframeCache& cache, Workspace& workspace) const;
        void keyframe_vjp(const Domain& domain, double density_weight, double velocity_weight, const KeyframeCache& cache, CellField<double>& density_adjoint, StaggeredVectorField<double>& velocity_adjoint, Workspace& workspace) const;
        void evaluate_control_effort(const Domain& domain, const CenteredVectorField<float>& control, StepCache& cache) const;
        void control_effort_jvp(const Domain& domain, const CenteredVectorField<float>& control, const CenteredVectorField<float>& control_tangent, StepCache& cache) const;
        void control_effort_vjp(const Domain& domain, const CenteredVectorField<float>& control, CenteredVectorField<double>& control_adjoint) const;

    private:
        const Configuration configuration;
        const std::uint32_t blur_radius;
        ::cuda::device_buffer<float> blur_weights;
    };
} // namespace physica::fluids::gas::operators
