module physica.fluids.gas.adjoint_control.lbfgsb;

import std;

namespace physica::fluids::gas::adjoint_control {
    namespace {
        double dot(const std::span<const double> left, const std::span<const double> right) {
            double result = 0.0;
            for (std::size_t index = 0u; index < left.size(); ++index) result += left[index] * right[index];
            return result;
        }
    } // namespace

    Lbfgsb::Lbfgsb(LbfgsbConfiguration next_configuration, const std::span<const double> initial_parameters, const std::span<const double> next_lower_bounds, const std::span<const double> next_upper_bounds)
        : configuration(std::move(next_configuration)),
          lower_bounds(next_lower_bounds.begin(), next_lower_bounds.end()),
          upper_bounds(next_upper_bounds.begin(), next_upper_bounds.end()),
          parameters(initial_parameters.begin(), initial_parameters.end()),
          gradient(parameters.size()),
          projected_gradient(parameters.size()),
          direction(parameters.size()),
          base_parameters(parameters.size()),
          base_gradient(parameters.size()),
          trial_parameters(parameters.size()) {
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) parameters[parameter] = std::clamp(parameters[parameter], lower_bounds[parameter], upper_bounds[parameter]);
    }

    LbfgsbRequest Lbfgsb::request() const {
        if (phase == Phase::complete) return {.kind = LbfgsbRequestKind::complete, .iteration = iteration, .evaluation = evaluation, .line_search_evaluation = line_search_evaluation, .step_length = step_length, .parameters = parameters};
        return {
            .kind = LbfgsbRequestKind::objective_gradient,
            .iteration = iteration,
            .evaluation = evaluation,
            .line_search_evaluation = line_search_evaluation,
            .step_length = phase == Phase::initial ? 0.0 : step_length,
            .parameters = phase == Phase::initial ? std::span<const double>{parameters} : std::span<const double>{trial_parameters},
        };
    }

    void Lbfgsb::submit(const double next_objective, const std::span<const double> next_gradient) {
        ++evaluation;
        if (phase == Phase::initial) {
            objective = next_objective;
            std::ranges::copy(next_gradient, gradient.begin());
            if (projected_gradient_norm() <= configuration.projected_gradient_tolerance) {
                finish(LbfgsbStopReason::projected_gradient);
                return;
            }
            begin_line_search();
            return;
        }

        const double sufficient_decrease = base_objective + configuration.armijo * step_length * base_directional_derivative;
        if (next_objective <= sufficient_decrease) {
            accept_trial(next_objective, next_gradient);
            return;
        }

        ++line_search_evaluation;
        step_length *= 0.5;
        if (line_search_evaluation >= configuration.maximum_line_search_evaluations || step_length < configuration.minimum_step) {
            parameters = base_parameters;
            gradient = base_gradient;
            objective = base_objective;
            finish(LbfgsbStopReason::line_search_failed);
            return;
        }
        set_trial();
    }

    double Lbfgsb::projected_gradient_norm() {
        double squared = 0.0;
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            const double value = parameters[parameter] - std::clamp(parameters[parameter] - gradient[parameter], lower_bounds[parameter], upper_bounds[parameter]);
            projected_gradient[parameter] = value;
            squared += value * value;
        }
        return std::sqrt(squared);
    }

    Lbfgsb::HessianRepresentation Lbfgsb::hessian_representation() const {
        HessianRepresentation result;
        if (!corrections.empty()) {
            const Correction& latest = corrections.back();
            result.scale = dot(latest.gradient_difference, latest.gradient_difference) * latest.inverse_curvature;
        }
        result.hessian_steps.reserve(corrections.size());
        result.hessian_step_curvatures.reserve(corrections.size());
        result.gradient_curvatures.reserve(corrections.size());
        for (std::size_t index = 0u; index < corrections.size(); ++index) {
            const Correction& correction = corrections[index];
            std::vector<double> hessian_step(correction.step.size());
            for (std::size_t parameter = 0u; parameter < correction.step.size(); ++parameter) hessian_step[parameter] = result.scale * correction.step[parameter];
            for (std::size_t previous = 0u; previous < index; ++previous) {
                const double hessian_coefficient = dot(result.hessian_steps[previous], correction.step) / result.hessian_step_curvatures[previous];
                const double gradient_coefficient = dot(corrections[previous].gradient_difference, correction.step) / result.gradient_curvatures[previous];
                for (std::size_t parameter = 0u; parameter < correction.step.size(); ++parameter) hessian_step[parameter] += -hessian_coefficient * result.hessian_steps[previous][parameter] + gradient_coefficient * corrections[previous].gradient_difference[parameter];
            }
            result.hessian_step_curvatures.push_back(dot(correction.step, hessian_step));
            result.gradient_curvatures.push_back(1.0 / correction.inverse_curvature);
            result.hessian_steps.push_back(std::move(hessian_step));
        }
        return result;
    }

    void Lbfgsb::hessian_product(const HessianRepresentation& hessian, const std::span<const double> input, const std::span<double> output) const {
        for (std::size_t parameter = 0u; parameter < input.size(); ++parameter) output[parameter] = hessian.scale * input[parameter];
        for (std::size_t index = 0u; index < corrections.size(); ++index) {
            const double hessian_coefficient = dot(hessian.hessian_steps[index], input) / hessian.hessian_step_curvatures[index];
            const double gradient_coefficient = dot(corrections[index].gradient_difference, input) / hessian.gradient_curvatures[index];
            for (std::size_t parameter = 0u; parameter < input.size(); ++parameter) output[parameter] += -hessian_coefficient * hessian.hessian_steps[index][parameter] + gradient_coefficient * corrections[index].gradient_difference[parameter];
        }
    }

    void Lbfgsb::inverse_hessian_product(const std::span<const double> input, const std::span<double> output) const {
        std::ranges::copy(input, output.begin());
        std::vector<double> coefficients(corrections.size());
        for (std::size_t reverse = 0u; reverse < corrections.size(); ++reverse) {
            const std::size_t index = corrections.size() - 1u - reverse;
            const Correction& correction = corrections[index];
            coefficients[index] = correction.inverse_curvature * dot(correction.step, output);
            for (std::size_t parameter = 0u; parameter < output.size(); ++parameter) output[parameter] -= coefficients[index] * correction.gradient_difference[parameter];
        }
        double scale = 1.0;
        if (!corrections.empty()) {
            const Correction& latest = corrections.back();
            scale = 1.0 / (dot(latest.gradient_difference, latest.gradient_difference) * latest.inverse_curvature);
        }
        for (double& value : output) value *= scale;
        for (std::size_t index = 0u; index < corrections.size(); ++index) {
            const Correction& correction = corrections[index];
            const double coefficient = coefficients[index] - correction.inverse_curvature * dot(correction.gradient_difference, output);
            for (std::size_t parameter = 0u; parameter < output.size(); ++parameter) output[parameter] += coefficient * correction.step[parameter];
        }
    }

    void Lbfgsb::generalized_cauchy_point(const HessianRepresentation& hessian, const std::span<double> cauchy) const {
        struct Breakpoint final {
            double time;
            std::size_t parameter;
        };

        std::vector<double> path(parameters.size());
        std::vector<double> free_direction(parameters.size());
        std::vector<Breakpoint> breakpoints;
        breakpoints.reserve(parameters.size());
        double gradient_direction = 0.0;
        double direction_squared = 0.0;
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            double value = -gradient[parameter];
            if (parameters[parameter] <= lower_bounds[parameter] && value < 0.0) value = 0.0;
            if (parameters[parameter] >= upper_bounds[parameter] && value > 0.0) value = 0.0;
            path[parameter] = value;
            free_direction[parameter] = value;
            gradient_direction += gradient[parameter] * value;
            direction_squared += value * value;
            const double breakpoint = value > 0.0 ? (upper_bounds[parameter] - parameters[parameter]) / value : value < 0.0 ? (lower_bounds[parameter] - parameters[parameter]) / value : std::numeric_limits<double>::infinity();
            if (std::isfinite(breakpoint)) breakpoints.push_back({.time = breakpoint, .parameter = parameter});
        }
        std::ranges::sort(breakpoints, {}, &Breakpoint::time);

        std::vector<double> hessian_path(corrections.size());
        std::vector<double> gradient_path(corrections.size());
        std::vector<double> hessian_displacement(corrections.size());
        std::vector<double> gradient_displacement(corrections.size());
        for (std::size_t index = 0u; index < corrections.size(); ++index) {
            hessian_path[index] = dot(hessian.hessian_steps[index], free_direction);
            gradient_path[index] = dot(corrections[index].gradient_difference, free_direction);
        }

        double direction_displacement = 0.0;
        double current_time = 0.0;
        std::size_t next_breakpoint = 0u;
        for (;;) {
            double slope = gradient_direction + hessian.scale * direction_displacement;
            double curvature = hessian.scale * direction_squared;
            for (std::size_t index = 0u; index < corrections.size(); ++index) {
                slope += -hessian_path[index] * hessian_displacement[index] / hessian.hessian_step_curvatures[index] + gradient_path[index] * gradient_displacement[index] / hessian.gradient_curvatures[index];
                curvature += -hessian_path[index] * hessian_path[index] / hessian.hessian_step_curvatures[index] + gradient_path[index] * gradient_path[index] / hessian.gradient_curvatures[index];
            }
            const double interval = next_breakpoint == breakpoints.size() ? std::numeric_limits<double>::infinity() : breakpoints[next_breakpoint].time - current_time;
            const double stationary_time = -slope / curvature;
            if (stationary_time >= 0.0 && stationary_time <= interval) {
                current_time += stationary_time;
                break;
            }
            if (!std::isfinite(interval)) throw std::runtime_error("Limited-memory Hessian lost positive curvature");
            direction_displacement += interval * direction_squared;
            for (std::size_t index = 0u; index < corrections.size(); ++index) {
                hessian_displacement[index] += interval * hessian_path[index];
                gradient_displacement[index] += interval * gradient_path[index];
            }
            current_time = breakpoints[next_breakpoint].time;
            while (next_breakpoint < breakpoints.size() && breakpoints[next_breakpoint].time == current_time) {
                const std::size_t parameter = breakpoints[next_breakpoint].parameter;
                const double value = free_direction[parameter];
                const double displacement = value > 0.0 ? upper_bounds[parameter] - parameters[parameter] : lower_bounds[parameter] - parameters[parameter];
                gradient_direction -= gradient[parameter] * value;
                direction_displacement -= value * displacement;
                direction_squared -= value * value;
                for (std::size_t index = 0u; index < corrections.size(); ++index) {
                    hessian_path[index] -= hessian.hessian_steps[index][parameter] * value;
                    gradient_path[index] -= corrections[index].gradient_difference[parameter] * value;
                }
                free_direction[parameter] = 0.0;
                ++next_breakpoint;
            }
            if (direction_squared == 0.0) break;
        }
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) cauchy[parameter] = std::clamp(parameters[parameter] + current_time * path[parameter], lower_bounds[parameter], upper_bounds[parameter]);
    }

    void Lbfgsb::subspace_minimization(const HessianRepresentation& hessian, const std::span<const double> cauchy, const std::span<double> candidate) const {
        std::vector<double> displacement(parameters.size());
        std::vector<double> hessian_displacement(parameters.size());
        std::vector<double> model_gradient(parameters.size());
        std::vector<std::uint8_t> free(parameters.size());
        bool all_free = true;
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            displacement[parameter] = cauchy[parameter] - parameters[parameter];
            free[parameter] = cauchy[parameter] > lower_bounds[parameter] && cauchy[parameter] < upper_bounds[parameter];
            all_free = all_free && free[parameter] != 0u;
        }
        hessian_product(hessian, displacement, hessian_displacement);
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) model_gradient[parameter] = gradient[parameter] + hessian_displacement[parameter];

        std::vector<double> subspace_direction(parameters.size());
        if (all_free) {
            inverse_hessian_product(model_gradient, subspace_direction);
            for (double& value : subspace_direction) value = -value;
        } else {
            std::vector<double> residual(parameters.size());
            std::vector<double> conjugate(parameters.size());
            std::vector<double> hessian_conjugate(parameters.size());
            double residual_squared = 0.0;
            for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
                residual[parameter] = free[parameter] == 0u ? 0.0 : -model_gradient[parameter];
                conjugate[parameter] = residual[parameter];
                residual_squared += residual[parameter] * residual[parameter];
            }
            const double initial_residual_squared = residual_squared;
            for (std::size_t iteration = 0u; iteration <= 2u * corrections.size() && residual_squared > 1.0e-20 * initial_residual_squared; ++iteration) {
                hessian_product(hessian, conjugate, hessian_conjugate);
                for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) if (free[parameter] == 0u) hessian_conjugate[parameter] = 0.0;
                const double step = residual_squared / dot(conjugate, hessian_conjugate);
                for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
                    subspace_direction[parameter] += step * conjugate[parameter];
                    residual[parameter] -= step * hessian_conjugate[parameter];
                }
                const double next_residual_squared = dot(residual, residual);
                const double coefficient = next_residual_squared / residual_squared;
                for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) conjugate[parameter] = residual[parameter] + coefficient * conjugate[parameter];
                residual_squared = next_residual_squared;
            }
        }

        double step = 1.0;
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            if (subspace_direction[parameter] > 0.0) step = std::min(step, (upper_bounds[parameter] - cauchy[parameter]) / subspace_direction[parameter]);
            if (subspace_direction[parameter] < 0.0) step = std::min(step, (lower_bounds[parameter] - cauchy[parameter]) / subspace_direction[parameter]);
        }
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) candidate[parameter] = std::clamp(cauchy[parameter] + step * subspace_direction[parameter], lower_bounds[parameter], upper_bounds[parameter]);
    }

    void Lbfgsb::begin_line_search() {
        if (iteration >= configuration.maximum_iterations) {
            finish(LbfgsbStopReason::maximum_iterations);
            return;
        }
        if (evaluation >= configuration.maximum_evaluations) {
            finish(LbfgsbStopReason::maximum_evaluations);
            return;
        }

        const double projected_norm = projected_gradient_norm();
        const HessianRepresentation hessian = hessian_representation();
        std::vector<double> cauchy(parameters.size());
        generalized_cauchy_point(hessian, cauchy);
        subspace_minimization(hessian, cauchy, trial_parameters);
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) direction[parameter] = trial_parameters[parameter] - parameters[parameter];
        base_directional_derivative = dot(gradient, direction);
        if (!(base_directional_derivative < 0.0)) {
            for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) direction[parameter] = cauchy[parameter] - parameters[parameter];
            base_directional_derivative = dot(gradient, direction);
        }
        if (!(base_directional_derivative < 0.0)) throw std::runtime_error("Generalized Cauchy point is not a descent direction");

        maximum_step = 1.0;
        base_parameters = parameters;
        base_gradient = gradient;
        base_objective = objective;
        line_search_evaluation = 0u;
        step_length = std::min(1.0, maximum_step);
        if (iteration == 0u) step_length = std::min(step_length, 1.0 / std::max(1.0, projected_norm));
        phase = Phase::line_search;
        set_trial();
    }

    void Lbfgsb::set_trial() {
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) trial_parameters[parameter] = std::clamp(base_parameters[parameter] + step_length * direction[parameter], lower_bounds[parameter], upper_bounds[parameter]);
    }

    void Lbfgsb::accept_trial(const double next_objective, const std::span<const double> next_gradient) {
        Correction correction{.step = std::vector<double>(parameters.size()), .gradient_difference = std::vector<double>(parameters.size())};
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            correction.step[parameter] = trial_parameters[parameter] - base_parameters[parameter];
            correction.gradient_difference[parameter] = next_gradient[parameter] - base_gradient[parameter];
        }
        const double curvature = dot(correction.step, correction.gradient_difference);
        const double step_squared = dot(correction.step, correction.step);
        const bool accepted = curvature > 1.0e-12 * step_squared;
        if (accepted) {
            correction.inverse_curvature = 1.0 / curvature;
            if (corrections.size() == configuration.memory) corrections.erase(corrections.begin());
            corrections.push_back(std::move(correction));
        }

        const double previous_objective = objective;
        parameters = trial_parameters;
        objective = next_objective;
        std::ranges::copy(next_gradient, gradient.begin());
        ++iteration;
        const double projected_norm = projected_gradient_norm();
        iterations.push_back({.iteration = iteration, .evaluation = evaluation, .objective = objective, .projected_gradient_norm = projected_norm, .step_length = step_length, .line_search_evaluations = line_search_evaluation + 1u, .correction_count = static_cast<std::uint32_t>(corrections.size()), .correction_accepted = accepted});
        phase = Phase::initial;

        if (projected_norm <= configuration.projected_gradient_tolerance) {
            finish(LbfgsbStopReason::projected_gradient);
            return;
        }
        if (std::abs(previous_objective - objective) <= configuration.relative_objective_tolerance * std::max({1.0, std::abs(previous_objective), std::abs(objective)})) {
            finish(LbfgsbStopReason::relative_objective);
            return;
        }
        begin_line_search();
    }

    void Lbfgsb::finish(const LbfgsbStopReason reason) {
        stop_reason = reason;
        phase = Phase::complete;
    }
} // namespace physica::fluids::gas::adjoint_control
