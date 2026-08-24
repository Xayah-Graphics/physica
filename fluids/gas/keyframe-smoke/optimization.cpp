module;

#include <physica/cuda.h>

module physica.fluids.gas.keyframe_smoke.optimization;

import std;

namespace physica::fluids::gas::keyframe_smoke {
    OptimizationResult OptimizationRunner::run(const std::span<const double> initial_parameters, const std::span<const std::uint8_t> active_parameters, OptimizationCoordinates coordinates) {
        OptimizationResult result{};
        result.parameters.assign(initial_parameters.begin(), initial_parameters.end());
        const std::uint32_t first_continuation_level = coordinates.continuation_level;
        for (std::uint32_t level = 0u; level < continuation.size(); ++level) {
            const ContinuationLevel& level_configuration = continuation[level];
            objective.set_blur_sigma(domain, level_configuration.blur_sigma_cells);
            std::vector<double> lower_bounds(control.parameters.lower_bounds);
            std::vector<double> upper_bounds(control.parameters.upper_bounds);
            for (std::size_t parameter = 0u; parameter < active_parameters.size(); ++parameter) {
                if (active_parameters[parameter] != 0u) continue;
                lower_bounds[parameter] = result.parameters[parameter];
                upper_bounds[parameter] = result.parameters[parameter];
            }
            Lbfgsb optimizer(level_configuration.optimizer, result.parameters, lower_bounds, upper_bounds);
            while (optimizer.request().kind == LbfgsbRequestKind::objective_gradient) {
                const LbfgsbRequest request = optimizer.request();
                EvaluationTrace trace = evaluator.evaluate(request.parameters, EvaluationMode::objective_gradient);
                coordinates.continuation_level = first_continuation_level + level;
                coordinates.optimizer_iteration = request.iteration;
                coordinates.objective_evaluation = request.evaluation;
                coordinates.line_search_evaluation = request.line_search_evaluation;
                coordinates.line_search_step = request.step_length;
                const double value = trace.summary.objective;
                const std::vector<double> gradient = trace.reverse->parameter_gradient;
                result.evaluations.push_back({.coordinates = coordinates, .parameters = {request.parameters.begin(), request.parameters.end()}, .summary = trace.summary});
                optimizer.submit(value, gradient);
            }
            result.parameters = optimizer.parameters;
            result.level_stop_reasons.push_back(optimizer.stop_reason);
        }
        result.final_trace.emplace(evaluator.evaluate(result.parameters, EvaluationMode::objective_gradient));
        return result;
    }
} // namespace physica::fluids::gas::keyframe_smoke
