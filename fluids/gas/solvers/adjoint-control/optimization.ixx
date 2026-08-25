export module physica.fluids.gas.adjoint_control.optimization;

import std;
import physica.fluids.gas.adjoint_control.control;
import physica.fluids.gas.adjoint_control.evaluation;
import physica.optimization.lbfgsb;

export namespace physica::fluids::gas::adjoint_control {
    struct OptimizationCoordinates final {
        std::uint32_t optimizer_iteration{};
        std::uint32_t objective_evaluation{};
        std::uint32_t line_search_evaluation{};
        double line_search_step{};
    };

    struct OptimizationEvaluation final {
        OptimizationCoordinates coordinates;
        EvaluationSummary summary;
    };

    struct OptimizationResult final {
        std::vector<double> parameters;
        std::vector<OptimizationEvaluation> evaluations;
        optimization::LbfgsbStopReason stop_reason{optimization::LbfgsbStopReason::running};
        std::optional<EvaluationTrace> final_trace;
    };

    template <typename SolverType>
    struct OptimizationRunner final {
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
        optimization::Lbfgsb optimizer(configuration, initial_parameters, lower_bounds, upper_bounds);
        OptimizationResult result{};
        while (optimizer.request().kind == optimization::LbfgsbRequestKind::objective_gradient) {
            const optimization::LbfgsbRequest request = optimizer.request();
            EvaluationTrace trace                     = evaluator.evaluate(request.parameters, EvaluationMode::objective_gradient);
            result.evaluations.push_back({
                .coordinates =
                    {
                        .optimizer_iteration    = request.iteration,
                        .objective_evaluation   = request.evaluation,
                        .line_search_evaluation = request.line_search_evaluation,
                        .line_search_step       = request.step_length,
                    },
                .summary = trace.summary,
            });
            optimizer.submit(trace.summary.objective, trace.reverse->parameter_gradient);
        }
        result.parameters  = optimizer.parameters;
        result.stop_reason = optimizer.stop_reason;
        result.final_trace.emplace(evaluator.evaluate(result.parameters, EvaluationMode::objective_gradient));
        return result;
    }
} // namespace physica::fluids::gas::adjoint_control
