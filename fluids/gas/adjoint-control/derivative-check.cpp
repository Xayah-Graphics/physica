module;

module physica.fluids.gas.adjoint_control.derivative_check;

import std;

namespace physica::fluids::gas::adjoint_control {
    DerivativeChecker::DerivativeChecker(Evaluator& next_evaluator)
        : evaluator(next_evaluator) {}

    DirectionalDerivativeCheck DerivativeChecker::directional(const std::span<const double> parameters, const std::span<const double> direction, const double epsilon) {
        EvaluationTrace analytic = evaluator.evaluate(parameters, EvaluationMode::objective_gradient_jvp, direction);
        std::vector<double> positive(parameters.begin(), parameters.end());
        std::vector<double> negative(parameters.begin(), parameters.end());
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            positive[parameter] += epsilon * direction[parameter];
            negative[parameter] -= epsilon * direction[parameter];
        }
        const double positive_objective = evaluator.evaluate(positive, EvaluationMode::objective).summary.objective;
        const double negative_objective = evaluator.evaluate(negative, EvaluationMode::objective).summary.objective;
        const double finite_difference = (positive_objective - negative_objective) / (2.0 * epsilon);
        const double finite_difference_jvp_scale = std::max({1.0e-12, std::abs(finite_difference), std::abs(analytic.summary.directional_derivative)});
        const double jvp_vjp_scale = std::max({1.0e-12, std::abs(analytic.summary.directional_derivative), std::abs(analytic.summary.gradient_dot_direction)});
        return {
            .objective = analytic.summary.objective,
            .finite_difference = finite_difference,
            .jvp = analytic.summary.directional_derivative,
            .vjp_dot_direction = analytic.summary.gradient_dot_direction,
            .finite_difference_jvp_relative_error = std::abs(finite_difference - analytic.summary.directional_derivative) / finite_difference_jvp_scale,
            .jvp_vjp_relative_error = std::abs(analytic.summary.directional_derivative - analytic.summary.gradient_dot_direction) / jvp_vjp_scale,
        };
    }

    std::vector<ComponentDerivativeCheck> DerivativeChecker::components(const std::span<const double> parameters, const std::span<const std::size_t> parameter_indices, const double epsilon) {
        EvaluationTrace analytic = evaluator.evaluate(parameters, EvaluationMode::objective_gradient);
        std::vector<ComponentDerivativeCheck> result;
        result.reserve(parameter_indices.size());
        for (const std::size_t parameter : parameter_indices) {
            std::vector<double> positive(parameters.begin(), parameters.end());
            std::vector<double> negative(parameters.begin(), parameters.end());
            positive[parameter] += epsilon;
            negative[parameter] -= epsilon;
            const double finite_difference = (evaluator.evaluate(positive, EvaluationMode::objective).summary.objective - evaluator.evaluate(negative, EvaluationMode::objective).summary.objective) / (2.0 * epsilon);
            const double gradient = analytic.reverse->parameter_gradient[parameter];
            result.push_back({
                .parameter = parameter,
                .analytic = gradient,
                .finite_difference = finite_difference,
                .relative_error = std::abs(gradient - finite_difference) / std::max({1.0e-12, std::abs(gradient), std::abs(finite_difference)}),
            });
        }
        return result;
    }
} // namespace physica::fluids::gas::adjoint_control
