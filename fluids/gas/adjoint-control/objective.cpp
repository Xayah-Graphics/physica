module;

#include "interop.h"
#include "objective_kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.adjoint_control.objective;

import std;

namespace physica::fluids::gas::adjoint_control {
    namespace {
        ::cuda::device_buffer<double> allocate_scalar_value(const Domain& domain) {
            return ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init};
        }

        std::vector<float> gaussian_weights(const float sigma, const std::uint32_t radius) {
            std::vector<float> result(radius * 2u + 1u);
            if (radius == 0u) {
                result[0] = 1.0F;
                return result;
            }
            float sum = 0.0F;
            for (int offset = -static_cast<int>(radius); offset <= static_cast<int>(radius); ++offset) {
                const float value = std::exp(-0.5F * offset * offset / (sigma * sigma));
                result[offset + radius] = value;
                sum += value;
            }
            for (float& value : result) value /= sum;
            return result;
        }
    } // namespace

    Problem::Problem(const std::uint32_t next_step_count, State next_initial_state, std::vector<Keyframe> next_keyframes)
        : begin_step(0u), step_count(next_step_count), initial_state(std::move(next_initial_state)), keyframes(std::move(next_keyframes)) {}

    Problem::Problem(const std::uint32_t next_begin_step, const std::uint32_t next_step_count, State next_initial_state, std::vector<Keyframe> next_keyframes)
        : begin_step(next_begin_step), step_count(next_step_count), initial_state(std::move(next_initial_state)), keyframes(std::move(next_keyframes)) {}

    Problem::~Problem() = default;

    Problem::Problem(Problem&&) noexcept = default;

    Problem& Problem::operator=(Problem&&) noexcept = default;

    Objective::Objective(const Domain& domain, ObjectiveConfiguration next_configuration)
        : configuration(std::move(next_configuration)),
          blur_radius(static_cast<std::uint32_t>(std::ceil(3.0F * configuration.blur_sigma_cells))),
          blur_weights(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), blur_radius * 2u + 1u, ::cuda::no_init),
          first{.density = domain.allocate_scalar_field(), .velocity = domain.allocate_staggered_vector_field()},
          second{.density = domain.allocate_scalar_field(), .velocity = domain.allocate_staggered_vector_field()},
          third{.density = domain.allocate_scalar_field(), .velocity = domain.allocate_staggered_vector_field()} {
        const std::vector<float> weights = gaussian_weights(configuration.blur_sigma_cells, blur_radius);
        ::cuda::copy_bytes(domain.stream, weights, blur_weights);
    }

    KeyframeCache Objective::allocate_keyframe_cache(const Domain& domain) const {
        return {
            .density_residual = domain.allocate_scalar_field(),
            .blurred_density_residual = domain.allocate_scalar_field(),
            .velocity_residual = domain.allocate_staggered_vector_field(),
            .blurred_velocity_residual = domain.allocate_staggered_vector_field(),
            .density_loss = allocate_scalar_value(domain),
            .velocity_loss = allocate_scalar_value(domain),
            .directional_derivative = allocate_scalar_value(domain),
        };
    }

    StepObjectiveCache Objective::allocate_step_objective_cache(const Domain& domain) const {
        return {.control_effort = allocate_scalar_value(domain), .directional_derivative = allocate_scalar_value(domain)};
    }

    void Objective::set_blur_sigma(const Domain& domain, const float sigma_cells) {
        configuration.blur_sigma_cells = sigma_cells;
        blur_radius = static_cast<std::uint32_t>(std::ceil(3.0F * sigma_cells));
        blur_weights = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), blur_radius * 2u + 1u, ::cuda::no_init};
        const std::vector<float> weights = gaussian_weights(sigma_cells, blur_radius);
        ::cuda::copy_bytes(domain.stream, weights, blur_weights);
    }

    void Objective::evaluate_keyframe(const Domain& domain, const State& state, const Keyframe& keyframe, KeyframeCache& cache) {
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        cuda_detail::residual_forward(domain.stream, grid, cuda_detail::scalar(state.density), cuda_detail::scalar(keyframe.target.density), cuda_detail::scalar(cache.density_residual));
        cuda_detail::residual_forward(domain.stream, grid, cuda_detail::staggered(state.velocity), cuda_detail::staggered(keyframe.target.velocity), cuda_detail::staggered(cache.velocity_residual));
        cuda_detail::blur_forward(domain.stream, grid, blur_radius, blur_weights.data(), cuda_detail::scalar(cache.density_residual), cuda_detail::scalar(first.density), cuda_detail::scalar(second.density), cuda_detail::scalar(cache.blurred_density_residual));
        cuda_detail::blur_forward(domain.stream, grid, blur_radius, blur_weights.data(), cuda_detail::staggered(cache.velocity_residual), cuda_detail::staggered(first.velocity), cuda_detail::staggered(second.velocity), cuda_detail::staggered(cache.blurred_velocity_residual));
        cuda_detail::squared_loss(domain.stream, grid, keyframe.density_weight, cuda_detail::scalar(cache.blurred_density_residual), cache.density_loss.data());
        cuda_detail::squared_loss(domain.stream, grid, keyframe.velocity_weight, cuda_detail::staggered(cache.blurred_velocity_residual), cache.velocity_loss.data());
    }

    void Objective::keyframe_jvp(const Domain& domain, const StateTangent& state_tangent, const Keyframe& keyframe, KeyframeCache& cache) {
        cuda_detail::directional_loss(domain.stream, cuda_detail::grid(domain.configuration), blur_radius, blur_weights.data(), keyframe.density_weight, keyframe.velocity_weight, cuda_detail::scalar(cache.blurred_density_residual), cuda_detail::staggered(cache.blurred_velocity_residual), cuda_detail::scalar(state_tangent.density), cuda_detail::staggered(state_tangent.velocity), cuda_detail::scalar(first.density), cuda_detail::scalar(second.density), cuda_detail::scalar(third.density), cuda_detail::staggered(first.velocity), cuda_detail::staggered(second.velocity), cuda_detail::staggered(third.velocity), cache.directional_derivative.data());
    }

    void Objective::keyframe_vjp(const Domain& domain, const Keyframe& keyframe, const KeyframeCache& cache, StateAdjoint& state_adjoint) {
        cuda_detail::inject_keyframe_adjoint(domain.stream, cuda_detail::grid(domain.configuration), blur_radius, blur_weights.data(), keyframe.density_weight, keyframe.velocity_weight, cuda_detail::scalar(cache.blurred_density_residual), cuda_detail::staggered(cache.blurred_velocity_residual), cuda_detail::scalar(first.density), cuda_detail::scalar(second.density), cuda_detail::staggered(first.velocity), cuda_detail::staggered(second.velocity), cuda_detail::scalar_adjoint(state_adjoint.density), cuda_detail::staggered_adjoint(state_adjoint.velocity));
    }

    void Objective::evaluate_control_effort(const Domain& domain, const DenseControl& control, StepObjectiveCache& cache) const {
        cuda_detail::control_effort(domain.stream, cuda_detail::grid(domain.configuration), configuration.control_effort_weight, cuda_detail::centered(control.force), cache.control_effort.data());
    }

    void Objective::control_effort_jvp(const Domain& domain, const DenseControl& control, const DenseControlTangent& control_tangent, StepObjectiveCache& cache) const {
        cuda_detail::control_effort_jvp(domain.stream, cuda_detail::grid(domain.configuration), configuration.control_effort_weight, cuda_detail::centered(control.force), cuda_detail::centered(control_tangent.force), cache.directional_derivative.data());
    }

    void Objective::control_effort_vjp(const Domain& domain, const DenseControl& control, DenseControlAdjoint& control_adjoint) const {
        cuda_detail::control_effort_vjp(domain.stream, cuda_detail::grid(domain.configuration), configuration.control_effort_weight, cuda_detail::centered(control.force), cuda_detail::centered_adjoint(control_adjoint.force));
    }
} // namespace physica::fluids::gas::adjoint_control
