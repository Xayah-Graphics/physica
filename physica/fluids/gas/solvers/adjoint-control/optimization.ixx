module;

#include <physica/cuda.h>

export module physica.fluids.gas.solvers.adjoint_control.optimization;

import std;
import physica.fluids.gas.solvers.adjoint_control.control;
import physica.fluids.gas.solvers.adjoint_control.evaluation;
import physica.fluids.gas.domain;
import physica.optimization.lbfgsb;

export namespace physica::fluids::gas::solvers::adjoint_control {
    struct OptimizationCoordinates final {
        std::uint32_t optimizer_iteration{};
        std::uint32_t objective_evaluation{};
        std::uint32_t line_search_evaluation{};
        double line_search_step{};
    };

    struct OptimizationEvaluation final {
        OptimizationCoordinates coordinates;
        EvaluationSummary summary;
        double gradient_norm{};
        double projected_gradient_norm{};
    };

    struct OptimizationResult final {
        std::vector<double> parameters;
        std::vector<OptimizationEvaluation> evaluations;
        optimization::LbfgsbStopReason stop_reason{optimization::LbfgsbStopReason::running};
        std::optional<EvaluationTrace> final_trace;
        double gradient_norm{};
        double projected_gradient_norm{};
    };

    template <typename SolverType>
    struct OptimizationRunner final {
        const Domain& domain;
        Evaluator<SolverType>& evaluator;
        const ControlSystem& control;
        const optimization::LbfgsbConfiguration configuration;

        [[nodiscard]] OptimizationResult run(std::span<const double> initial_parameters, std::span<const std::uint8_t> active_parameters);
    };

    template <typename SolverType>
    OptimizationResult OptimizationRunner<SolverType>::run(const std::span<const double> initial_parameters, const std::span<const std::uint8_t> active_parameters) {
        std::vector<double> lower_bounds(control.lower_bounds);
        std::vector<double> upper_bounds(control.upper_bounds);
        for (std::size_t parameter = 0u; parameter < active_parameters.size(); ++parameter) {
            if (active_parameters[parameter] != 0u) continue;
            lower_bounds[parameter] = initial_parameters[parameter];
            upper_bounds[parameter] = initial_parameters[parameter];
        }
        optimization::Lbfgsb optimizer(domain.grid.stream, configuration, initial_parameters, lower_bounds, upper_bounds);
        OptimizationResult result{};
        while (optimizer.request().kind == optimization::LbfgsbRequestKind::objective_gradient) {
            const optimization::LbfgsbRequest request         = optimizer.request();
            EvaluationTrace trace                             = evaluator.evaluate(request.parameters, EvaluationMode::objective_gradient);
            const optimization::LbfgsbGradientMetrics metrics = optimizer.submit(trace.objective.data(), {trace.reverse->parameter_gradient.data(), trace.reverse->parameter_gradient.size()});
            result.evaluations.push_back({
                .coordinates =
                    {
                        .optimizer_iteration    = request.iteration,
                        .objective_evaluation   = request.evaluation,
                        .line_search_evaluation = request.line_search_evaluation,
                        .line_search_step       = request.step_length,
                    },
                .summary                 = trace.summary,
                .gradient_norm           = metrics.gradient_norm,
                .projected_gradient_norm = metrics.projected_gradient_norm,
            });
        }
        result.parameters.resize(optimizer.parameters.size());
        ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span<const double>{optimizer.parameters.data(), optimizer.parameters.size()}, ::cuda::std::span{result.parameters.data(), result.parameters.size()});
        domain.grid.stream.sync();
        result.stop_reason             = optimizer.stop_reason;
        result.gradient_norm           = optimizer.gradient_norm;
        result.projected_gradient_norm = optimizer.projected_gradient_norm;
        result.final_trace.emplace(evaluator.evaluate({optimizer.parameters.data(), optimizer.parameters.size()}, EvaluationMode::objective_gradient));
        return result;
    }
} // namespace physica::fluids::gas::solvers::adjoint_control
