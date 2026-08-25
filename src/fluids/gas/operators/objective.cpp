module;

#include "../detail/cuda/interop.h"
#include "objective-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.objective;

import std;

namespace physica::fluids::gas::operators {
    namespace {
        ::cuda::device_buffer<double> allocate_scalar(const Domain& domain) {
            return ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init};
        }

        std::vector<float> gaussian_weights(const float sigma, const std::uint32_t radius) {
            std::vector<float> result(radius * 2u + 1u);
            if (radius == 0u) {
                result[0] = 1.0F;
                return result;
            }
            float sum{};
            for (int offset = -static_cast<int>(radius); offset <= static_cast<int>(radius); ++offset) {
                const float value       = std::exp(-0.5F * offset * offset / (sigma * sigma));
                result[offset + radius] = value;
                sum += value;
            }
            for (float& value : result) value /= sum;
            return result;
        }
    } // namespace

    Quadratic::Quadratic(const Domain& domain, Configuration next_configuration) : configuration(std::move(next_configuration)), blur_radius(static_cast<std::uint32_t>(std::ceil(3.0F * configuration.blur_sigma_cells))), blur_weights(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), blur_radius * 2u + 1u, ::cuda::no_init) {
        const std::vector<float> weights = gaussian_weights(configuration.blur_sigma_cells, blur_radius);
        ::cuda::copy_bytes(domain.stream, weights, blur_weights);
    }

    KeyframeCache Quadratic::allocate_keyframe_cache(const Domain& domain) const {
        return {
            .density_residual          = domain.allocate_cell_field<float>(),
            .blurred_density_residual  = domain.allocate_cell_field<float>(),
            .velocity_residual         = domain.allocate_staggered_vector_field<float>(),
            .blurred_velocity_residual = domain.allocate_staggered_vector_field<float>(),
            .density_loss              = allocate_scalar(domain),
            .velocity_loss             = allocate_scalar(domain),
            .directional_derivative    = allocate_scalar(domain),
        };
    }

    StepCache Quadratic::allocate_step_cache(const Domain& domain) const {
        return {.control_effort = allocate_scalar(domain), .directional_derivative = allocate_scalar(domain)};
    }

    Workspace Quadratic::allocate_workspace(const Domain& domain) const {
        return {
            .density_blur_first   = domain.allocate_cell_field<float>(),
            .density_blur_second  = domain.allocate_cell_field<float>(),
            .density_blur_output  = domain.allocate_cell_field<float>(),
            .velocity_blur_first  = domain.allocate_staggered_vector_field<float>(),
            .velocity_blur_second = domain.allocate_staggered_vector_field<float>(),
            .velocity_blur_output = domain.allocate_staggered_vector_field<float>(),
        };
    }

    Quadratic Quadratic::with_blur_sigma(const Domain& domain, const float sigma_cells) const {
        return Quadratic{domain, {.control_effort_weight = configuration.control_effort_weight, .blur_sigma_cells = sigma_cells}};
    }

    void Quadratic::evaluate_keyframe(const Domain& domain, const CellField<float>& density, const StaggeredVectorField<float>& velocity, const CellField<float>& target_density, const StaggeredVectorField<float>& target_velocity, const double density_weight, const double velocity_weight, KeyframeCache& cache, Workspace& workspace) const {
        const detail::cuda::Grid grid = detail::cuda::grid(domain.configuration);
        cuda_backend::residual_forward(domain.stream, grid, detail::cuda::scalar(density), detail::cuda::scalar(target_density), detail::cuda::scalar(cache.density_residual));
        cuda_backend::residual_forward(domain.stream, grid, detail::cuda::staggered(velocity), detail::cuda::staggered(target_velocity), detail::cuda::staggered(cache.velocity_residual));
        cuda_backend::blur_forward(domain.stream, grid, blur_radius, blur_weights.data(), detail::cuda::scalar(cache.density_residual), detail::cuda::scalar(workspace.density_blur_first), detail::cuda::scalar(workspace.density_blur_second), detail::cuda::scalar(cache.blurred_density_residual));
        cuda_backend::blur_forward(domain.stream, grid, blur_radius, blur_weights.data(), detail::cuda::staggered(cache.velocity_residual), detail::cuda::staggered(workspace.velocity_blur_first), detail::cuda::staggered(workspace.velocity_blur_second), detail::cuda::staggered(cache.blurred_velocity_residual));
        cuda_backend::squared_loss(domain.stream, grid, density_weight, detail::cuda::scalar(cache.blurred_density_residual), cache.density_loss.data());
        cuda_backend::squared_loss(domain.stream, grid, velocity_weight, detail::cuda::staggered(cache.blurred_velocity_residual), cache.velocity_loss.data());
    }

    void Quadratic::keyframe_jvp(const Domain& domain, const CellField<float>& density_tangent, const StaggeredVectorField<float>& velocity_tangent, const double density_weight, const double velocity_weight, KeyframeCache& cache, Workspace& workspace) const {
        cuda_backend::directional_loss(domain.stream, detail::cuda::grid(domain.configuration), blur_radius, blur_weights.data(), density_weight, velocity_weight, detail::cuda::scalar(cache.blurred_density_residual), detail::cuda::staggered(cache.blurred_velocity_residual), detail::cuda::scalar(density_tangent), detail::cuda::staggered(velocity_tangent), detail::cuda::scalar(workspace.density_blur_first), detail::cuda::scalar(workspace.density_blur_second), detail::cuda::scalar(workspace.density_blur_output), detail::cuda::staggered(workspace.velocity_blur_first), detail::cuda::staggered(workspace.velocity_blur_second), detail::cuda::staggered(workspace.velocity_blur_output), cache.directional_derivative.data());
    }

    void Quadratic::keyframe_vjp(const Domain& domain, const double density_weight, const double velocity_weight, const KeyframeCache& cache, CellField<double>& density_adjoint, StaggeredVectorField<double>& velocity_adjoint, Workspace& workspace) const {
        cuda_backend::inject_keyframe_adjoint(domain.stream, detail::cuda::grid(domain.configuration), blur_radius, blur_weights.data(), density_weight, velocity_weight, detail::cuda::scalar(cache.blurred_density_residual), detail::cuda::staggered(cache.blurred_velocity_residual), detail::cuda::scalar(workspace.density_blur_first), detail::cuda::scalar(workspace.density_blur_second), detail::cuda::staggered(workspace.velocity_blur_first), detail::cuda::staggered(workspace.velocity_blur_second), detail::cuda::scalar_adjoint(density_adjoint), detail::cuda::staggered_adjoint(velocity_adjoint));
    }

    void Quadratic::evaluate_control_effort(const Domain& domain, const CenteredVectorField<float>& control, StepCache& cache) const {
        cuda_backend::control_effort(domain.stream, detail::cuda::grid(domain.configuration), configuration.control_effort_weight, detail::cuda::centered(control), cache.control_effort.data());
    }

    void Quadratic::accumulate(const Domain& domain, const ::cuda::device_buffer<double>& contribution, ::cuda::device_buffer<double>& objective) const {
        cuda_backend::accumulate(domain.stream, contribution.data(), objective.data());
    }

    void Quadratic::control_effort_jvp(const Domain& domain, const CenteredVectorField<float>& control, const CenteredVectorField<float>& control_tangent, StepCache& cache) const {
        cuda_backend::control_effort_jvp(domain.stream, detail::cuda::grid(domain.configuration), configuration.control_effort_weight, detail::cuda::centered(control), detail::cuda::centered(control_tangent), cache.directional_derivative.data());
    }

    void Quadratic::control_effort_vjp(const Domain& domain, const CenteredVectorField<float>& control, CenteredVectorField<double>& control_adjoint) const {
        cuda_backend::control_effort_vjp(domain.stream, detail::cuda::grid(domain.configuration), configuration.control_effort_weight, detail::cuda::centered(control), detail::cuda::centered_adjoint(control_adjoint));
    }
} // namespace physica::fluids::gas::operators
