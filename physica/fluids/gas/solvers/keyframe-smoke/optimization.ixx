module;

#include <physica/cuda.h>

export module physica.fluids.gas.solvers.keyframe_smoke.optimization;

import std;
import physica.fluids.gas.solvers.keyframe_smoke.control;
import physica.fluids.gas.domain;
import physica.fluids.gas.solvers.keyframe_smoke.evaluation;
import physica.optimization.lbfgsb;
import physica.fluids.gas.operators.objective;

export namespace physica::fluids::gas::solvers::keyframe_smoke {
    struct ContinuationLevel final {
        float blur_sigma_cells{};
        optimization::LbfgsbConfiguration optimizer;
    };

    struct OptimizationCoordinates final {
        std::uint32_t shooting_pass{};
        std::uint32_t schedule{};
        std::uint32_t segment{};
        std::uint32_t continuation_level{};
        std::uint32_t optimizer_iteration{};
        std::uint32_t objective_evaluation{};
        std::uint32_t line_search_evaluation{};
        double line_search_step{};
    };

    struct OptimizationEvaluation final {
        OptimizationCoordinates coordinates;
        std::vector<double> parameters;
        EvaluationSummary summary;
        double gradient_norm{};
        double projected_gradient_norm{};
    };

    struct OptimizationResult final {
        std::vector<double> parameters;
        std::vector<OptimizationEvaluation> evaluations;
        std::vector<optimization::LbfgsbStopReason> level_stop_reasons;
        std::optional<EvaluationTrace> final_trace;
        double gradient_norm{};
        double projected_gradient_norm{};
    };

    template <typename SolverType>
    struct OptimizationRunner final {
        const Domain& domain;
        Evaluator<SolverType>& evaluator;
        operators::Quadratic& objective_function;
        const ControlSystem& control;
        std::vector<ContinuationLevel> continuation;

        [[nodiscard]] OptimizationResult run(std::span<const double> initial_parameters, std::span<const std::uint8_t> active_parameters, OptimizationCoordinates coordinates = {});
    };

    template <typename SolverType>
    OptimizationResult OptimizationRunner<SolverType>::run(const std::span<const double> initial_parameters, const std::span<const std::uint8_t> active_parameters, OptimizationCoordinates coordinates) {
        OptimizationResult result{};
        result.parameters.assign(initial_parameters.begin(), initial_parameters.end());
        ::cuda::device_buffer<double> final_parameters(domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), initial_parameters.size(), ::cuda::no_init);
        const std::uint32_t first_continuation_level = coordinates.continuation_level;
        for (std::uint32_t level = 0u; level < continuation.size(); ++level) {
            const ContinuationLevel& level_configuration = continuation[level];
            operators::Quadratic level_objective         = objective_function.with_blur_sigma(domain, level_configuration.blur_sigma_cells);
            std::vector<double> lower_bounds(control.lower_bounds);
            std::vector<double> upper_bounds(control.upper_bounds);
            for (std::size_t parameter = 0u; parameter < active_parameters.size(); ++parameter) {
                if (active_parameters[parameter] != 0u) continue;
                lower_bounds[parameter] = result.parameters[parameter];
                upper_bounds[parameter] = result.parameters[parameter];
            }
            optimization::Lbfgsb optimizer(domain.grid.stream, level_configuration.optimizer, result.parameters, lower_bounds, upper_bounds);
            while (optimizer.request().kind == optimization::LbfgsbRequestKind::objective_gradient) {
                const optimization::LbfgsbRequest request = optimizer.request();
                std::vector<double> evaluation_parameters(request.parameters.size());
                ::cuda::copy_bytes(domain.grid.stream, request.parameters, ::cuda::std::span{evaluation_parameters.data(), evaluation_parameters.size()});
                EvaluationTrace trace                             = evaluator.evaluate(level_objective, request.parameters, EvaluationMode::objective_gradient);
                coordinates.continuation_level                    = first_continuation_level + level;
                coordinates.optimizer_iteration                   = request.iteration;
                coordinates.objective_evaluation                  = request.evaluation;
                coordinates.line_search_evaluation                = request.line_search_evaluation;
                coordinates.line_search_step                      = request.step_length;
                const optimization::LbfgsbGradientMetrics metrics = optimizer.submit(trace.objective.data(), {trace.reverse->parameter_gradient.data(), trace.reverse->parameter_gradient.size()});
                result.evaluations.push_back({.coordinates = coordinates, .parameters = std::move(evaluation_parameters), .summary = trace.summary, .gradient_norm = metrics.gradient_norm, .projected_gradient_norm = metrics.projected_gradient_norm});
            }
            ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span<const double>{optimizer.parameters.data(), optimizer.parameters.size()}, ::cuda::std::span{final_parameters.data(), final_parameters.size()});
            ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span<const double>{optimizer.parameters.data(), optimizer.parameters.size()}, ::cuda::std::span{result.parameters.data(), result.parameters.size()});
            domain.grid.stream.sync();
            result.level_stop_reasons.push_back(optimizer.stop_reason);
            result.gradient_norm           = optimizer.gradient_norm;
            result.projected_gradient_norm = optimizer.projected_gradient_norm;
        }
        operators::Quadratic final_objective = objective_function.with_blur_sigma(domain, continuation.back().blur_sigma_cells);
        result.final_trace.emplace(evaluator.evaluate(final_objective, {final_parameters.data(), final_parameters.size()}, EvaluationMode::objective_gradient));
        return result;
    }
} // namespace physica::fluids::gas::solvers::keyframe_smoke
