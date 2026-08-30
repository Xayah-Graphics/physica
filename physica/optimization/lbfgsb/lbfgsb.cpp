module;

#include "lbfgsb-kernels.h"
#include <physica/cuda.h>
module physica.optimization.lbfgsb;

import std;

namespace physica::optimization {
    struct LbfgsbBackendAccess final {
        [[nodiscard]] static kernels::Storage storage(Lbfgsb& optimizer) {
            return {
                .parameters                      = optimizer.parameters.data(),
                .lower_bounds                    = optimizer.lower_bounds.data(),
                .upper_bounds                    = optimizer.upper_bounds.data(),
                .gradient                        = optimizer.gradient.data(),
                .trial_parameters                = optimizer.trial_parameters.data(),
                .correction_steps                = optimizer.correction_steps.data(),
                .correction_gradient_differences = optimizer.correction_gradient_differences.data(),
                .inverse_curvatures              = optimizer.inverse_curvatures.data(),
                .hessian_steps                   = optimizer.hessian_steps.data(),
                .hessian_step_curvatures         = optimizer.hessian_step_curvatures.data(),
                .gradient_curvatures             = optimizer.gradient_curvatures.data(),
                .direction                       = optimizer.direction.data(),
                .path                            = optimizer.path.data(),
                .free_direction                  = optimizer.free_direction.data(),
                .cauchy                          = optimizer.cauchy.data(),
                .displacement                    = optimizer.displacement.data(),
                .hessian_displacement            = optimizer.hessian_displacement.data(),
                .model_gradient                  = optimizer.model_gradient.data(),
                .subspace_direction              = optimizer.subspace_direction.data(),
                .residual                        = optimizer.residual.data(),
                .conjugate                       = optimizer.conjugate.data(),
                .hessian_conjugate               = optimizer.hessian_conjugate.data(),
                .breakpoints                     = optimizer.breakpoints.data(),
                .sorted_breakpoints              = optimizer.sorted_breakpoints.data(),
                .breakpoint_indices              = optimizer.breakpoint_indices.data(),
                .sorted_breakpoint_indices       = optimizer.sorted_breakpoint_indices.data(),
                .free_mask                       = optimizer.free_mask.data(),
                .sort_scratch                    = optimizer.sort_scratch.data(),
                .sort_scratch_bytes              = optimizer.sort_scratch.size(),
                .status                          = optimizer.status.data(),
            };
        }
    };

    Lbfgsb::Lbfgsb(const ::cuda::stream_ref next_stream, LbfgsbConfiguration next_configuration, const std::span<const double> initial_parameters, const std::span<const double> next_lower_bounds, const std::span<const double> next_upper_bounds)
        : configuration(std::move(next_configuration)), parameter_count(initial_parameters.size()), parameters(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), stream(next_stream), lower_bounds(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), upper_bounds(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), gradient(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), trial_parameters(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), correction_steps(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), static_cast<std::size_t>(configuration.memory) * parameter_count, ::cuda::no_init),
          correction_gradient_differences(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), static_cast<std::size_t>(configuration.memory) * parameter_count, ::cuda::no_init), inverse_curvatures(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), configuration.memory, ::cuda::no_init), hessian_steps(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), static_cast<std::size_t>(configuration.memory) * parameter_count, ::cuda::no_init), hessian_step_curvatures(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), configuration.memory, ::cuda::no_init), gradient_curvatures(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), configuration.memory, ::cuda::no_init), direction(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), path(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init),
          free_direction(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), cauchy(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), displacement(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), hessian_displacement(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), model_gradient(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), subspace_direction(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), residual(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), conjugate(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init),
          hessian_conjugate(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), breakpoints(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), sorted_breakpoints(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), breakpoint_indices(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), sorted_breakpoint_indices(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), free_mask(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), parameter_count, ::cuda::no_init), sort_scratch(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), kernels::sort_scratch_size(static_cast<std::uint32_t>(parameter_count)), ::cuda::no_init),
          status(next_stream, ::cuda::device_default_memory_pool(next_stream.device()), std::max({sizeof(kernels::SearchResult), sizeof(kernels::GradientMetrics), sizeof(kernels::AcceptanceResult)}), ::cuda::no_init) {
        ::cuda::copy_bytes(stream, initial_parameters, parameters);
        ::cuda::copy_bytes(stream, next_lower_bounds, lower_bounds);
        ::cuda::copy_bytes(stream, next_upper_bounds, upper_bounds);
        kernels::initialize(stream, static_cast<std::uint32_t>(parameter_count), LbfgsbBackendAccess::storage(*this));
        stream.sync();
    }

    LbfgsbRequest Lbfgsb::request() const {
        if (phase == Phase::complete) return {.kind = LbfgsbRequestKind::complete, .iteration = iteration, .evaluation = evaluation, .line_search_evaluation = line_search_evaluation, .step_length = step_length, .parameters = {parameters.data(), parameters.size()}};
        return {
            .kind                   = LbfgsbRequestKind::objective_gradient,
            .iteration              = iteration,
            .evaluation             = evaluation,
            .line_search_evaluation = line_search_evaluation,
            .step_length            = phase == Phase::initial ? 0.0 : step_length,
            .parameters             = phase == Phase::initial ? ::cuda::std::span<const double>{parameters.data(), parameters.size()} : ::cuda::std::span<const double>{trial_parameters.data(), trial_parameters.size()},
        };
    }

    LbfgsbGradientMetrics Lbfgsb::submit(const double* next_objective_device, const ::cuda::std::span<const double> next_gradient) {
        double next_objective;
        ::cuda::copy_bytes(stream, ::cuda::std::span{next_objective_device, 1u}, ::cuda::std::span{&next_objective, 1u});
        const double* request_parameters             = phase == Phase::initial ? parameters.data() : trial_parameters.data();
        const kernels::GradientMetrics device_result = kernels::gradient_metrics(stream, static_cast<std::uint32_t>(parameter_count), request_parameters, next_gradient.data(), LbfgsbBackendAccess::storage(*this));
        const LbfgsbGradientMetrics metrics{.gradient_norm = device_result.gradient_norm, .projected_gradient_norm = device_result.projected_gradient_norm};
        ++evaluation;

        if (phase == Phase::initial) {
            objective               = next_objective;
            gradient_norm           = metrics.gradient_norm;
            projected_gradient_norm = metrics.projected_gradient_norm;
            ::cuda::copy_bytes(stream, next_gradient, gradient);
            if (projected_gradient_norm <= configuration.projected_gradient_tolerance) {
                finish(LbfgsbStopReason::projected_gradient);
                return metrics;
            }
            begin_line_search(projected_gradient_norm);
            return metrics;
        }

        const double sufficient_decrease = base_objective + configuration.armijo * step_length * base_directional_derivative;
        if (next_objective <= sufficient_decrease) {
            accept_trial(next_objective, next_gradient, metrics);
            return metrics;
        }

        ++line_search_evaluation;
        step_length *= 0.5;
        if (line_search_evaluation >= configuration.maximum_line_search_evaluations || step_length < configuration.minimum_step) {
            finish(LbfgsbStopReason::line_search_failed);
            return metrics;
        }
        set_trial();
        return metrics;
    }

    void Lbfgsb::begin_line_search(const double projected_gradient_norm) {
        if (iteration >= configuration.maximum_iterations) {
            finish(LbfgsbStopReason::maximum_iterations);
            return;
        }
        if (evaluation >= configuration.maximum_evaluations) {
            finish(LbfgsbStopReason::maximum_evaluations);
            return;
        }

        const kernels::SearchResult search = kernels::prepare_direction(stream, static_cast<std::uint32_t>(parameter_count), configuration.memory, correction_count, correction_head, LbfgsbBackendAccess::storage(*this));
        if (search.error != 0u) throw std::runtime_error(search.error == 1u ? "Limited-memory Hessian lost positive curvature" : "Generalized Cauchy point is not a descent direction");

        base_directional_derivative = search.base_directional_derivative;
        base_objective              = objective;
        line_search_evaluation      = 0u;
        step_length                 = iteration == 0u ? std::min(1.0, 1.0 / std::max(1.0, projected_gradient_norm)) : 1.0;
        phase                       = Phase::line_search;
        set_trial();
    }

    void Lbfgsb::set_trial() {
        kernels::set_trial(stream, static_cast<std::uint32_t>(parameter_count), step_length, LbfgsbBackendAccess::storage(*this));
    }

    void Lbfgsb::accept_trial(const double next_objective, const ::cuda::std::span<const double> next_gradient, const LbfgsbGradientMetrics metrics) {
        const std::uint32_t correction_slot        = correction_count < configuration.memory ? correction_count : correction_head;
        const kernels::AcceptanceResult acceptance = kernels::accept_trial(stream, static_cast<std::uint32_t>(parameter_count), correction_slot, next_gradient.data(), LbfgsbBackendAccess::storage(*this));
        const bool accepted                        = acceptance.correction_accepted != 0u;
        if (accepted) {
            if (correction_count < configuration.memory) ++correction_count;
            else correction_head = (correction_head + 1u) % configuration.memory;
        }

        const double previous_objective = objective;
        objective                       = next_objective;
        gradient_norm                   = metrics.gradient_norm;
        projected_gradient_norm         = metrics.projected_gradient_norm;
        ++iteration;
        iterations.push_back({.iteration = iteration, .evaluation = evaluation, .objective = objective, .projected_gradient_norm = projected_gradient_norm, .step_length = step_length, .line_search_evaluations = line_search_evaluation + 1u, .correction_count = correction_count, .correction_accepted = accepted});
        phase = Phase::initial;

        if (projected_gradient_norm <= configuration.projected_gradient_tolerance) {
            finish(LbfgsbStopReason::projected_gradient);
            return;
        }
        if (std::abs(previous_objective - objective) <= configuration.relative_objective_tolerance * std::max({1.0, std::abs(previous_objective), std::abs(objective)})) {
            finish(LbfgsbStopReason::relative_objective);
            return;
        }
        begin_line_search(projected_gradient_norm);
    }

    void Lbfgsb::finish(const LbfgsbStopReason reason) {
        stop_reason = reason;
        phase       = Phase::complete;
    }
} // namespace physica::optimization
