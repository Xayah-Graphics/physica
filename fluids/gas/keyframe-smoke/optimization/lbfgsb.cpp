module;

module physica.fluids.gas.keyframe_smoke.lbfgsb;

import std;

namespace physica::fluids::gas::keyframe_smoke {
    namespace {
        double dot(const std::span<const double> first, const std::span<const double> second) {
            return std::transform_reduce(first.begin(), first.end(), second.begin(), 0.0);
        }

        std::vector<double> multiply(const std::span<const double> matrix, const std::span<const double> vector) {
            const std::size_t size = vector.size();
            std::vector<double> result(size, 0.0);
            for (std::size_t row = 0u; row < size; ++row) for (std::size_t column = 0u; column < size; ++column) result[row] += matrix[row * size + column] * vector[column];
            return result;
        }

        std::vector<double> solve_positive_definite(std::vector<double> matrix, std::vector<double> right_hand_side) {
            const std::size_t size = right_hand_side.size();
            for (std::size_t row = 0u; row < size; ++row) {
                for (std::size_t column = 0u; column <= row; ++column) {
                    double value = matrix[row * size + column];
                    for (std::size_t inner = 0u; inner < column; ++inner) value -= matrix[row * size + inner] * matrix[column * size + inner];
                    matrix[row * size + column] = row == column ? std::sqrt(value) : value / matrix[column * size + column];
                }
            }
            for (std::size_t row = 0u; row < size; ++row) {
                for (std::size_t column = 0u; column < row; ++column) right_hand_side[row] -= matrix[row * size + column] * right_hand_side[column];
                right_hand_side[row] /= matrix[row * size + row];
            }
            for (std::size_t reverse = 0u; reverse < size; ++reverse) {
                const std::size_t row = size - 1u - reverse;
                for (std::size_t column = row + 1u; column < size; ++column) right_hand_side[row] -= matrix[column * size + row] * right_hand_side[column];
                right_hand_side[row] /= matrix[row * size + row];
            }
            return right_hand_side;
        }
    } // namespace

    Lbfgsb::Lbfgsb(LbfgsbConfiguration next_configuration, const std::span<const double> initial_parameters, const std::span<const double> next_lower_bounds, const std::span<const double> next_upper_bounds)
        : configuration(std::move(next_configuration)),
          lower_bounds(next_lower_bounds.begin(), next_lower_bounds.end()),
          upper_bounds(next_upper_bounds.begin(), next_upper_bounds.end()),
          parameters(initial_parameters.begin(), initial_parameters.end()),
          trial_parameters(parameters),
          direction(parameters.size()),
          gradient(parameters.size()) {
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) parameters[parameter] = std::clamp(parameters[parameter], lower_bounds[parameter], upper_bounds[parameter]);
        trial_parameters = parameters;
    }

    LbfgsbRequest Lbfgsb::request() const {
        return {
            .kind = phase == Phase::complete ? LbfgsbRequestKind::complete : LbfgsbRequestKind::objective_gradient,
            .iteration = iteration,
            .evaluation = evaluation,
            .line_search_evaluation = line_search_evaluation,
            .step_length = phase == Phase::line_search ? step_length : 0.0,
            .parameters = phase == Phase::line_search ? std::span<const double>{trial_parameters} : std::span<const double>{parameters},
        };
    }

    void Lbfgsb::submit(const double next_objective, const std::span<const double> next_gradient) {
        ++evaluation;
        if (phase == Phase::initial) {
            objective = next_objective;
            std::ranges::copy(next_gradient, gradient.begin());
            iterations.push_back({
                .iteration = 0u,
                .evaluation = evaluation,
                .objective = objective,
                .projected_gradient_norm = projected_gradient_norm(),
            });
            if (projected_gradient_norm() <= configuration.projected_gradient_tolerance) {
                finish(LbfgsbStopReason::projected_gradient);
                return;
            }
            if (evaluation >= configuration.maximum_evaluations) {
                finish(LbfgsbStopReason::maximum_evaluations);
                return;
            }
            begin_line_search();
            return;
        }

        ++line_search_evaluation;
        const double directional_derivative = dot(next_gradient, direction);
        const bool armijo = next_objective <= objective + configuration.armijo * step_length * initial_directional_derivative;
        const bool strong_wolfe = std::abs(directional_derivative) <= configuration.curvature * std::abs(initial_directional_derivative);
        if (armijo && (strong_wolfe || step_length == maximum_step || line_search_evaluation >= 4u)) {
            accept_trial(next_objective, next_gradient);
            return;
        }
        if (!armijo || (high_step_exists && next_objective >= low_objective)) {
            high_step = step_length;
            high_step_exists = true;
        } else if (directional_derivative >= 0.0) {
            high_step = step_length;
            high_step_exists = true;
        } else {
            low_step = step_length;
            low_objective = next_objective;
        }
        if (line_search_evaluation >= configuration.maximum_line_search_evaluations) {
            finish(LbfgsbStopReason::line_search_failed);
            return;
        }
        if (evaluation >= configuration.maximum_evaluations) {
            finish(LbfgsbStopReason::maximum_evaluations);
            return;
        }
        step_length = high_step_exists ? 0.5 * (low_step + high_step) : std::min(2.0 * step_length, maximum_step);
        set_trial();
    }

    double Lbfgsb::projected_gradient_norm() const {
        double result = 0.0;
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            const double projected = parameters[parameter] - std::clamp(parameters[parameter] - gradient[parameter], lower_bounds[parameter], upper_bounds[parameter]);
            result = std::max(result, std::abs(projected));
        }
        return result;
    }

    std::vector<double> Lbfgsb::limited_memory_hessian() const {
        const std::size_t size = parameters.size();
        double scale = 1.0;
        if (!corrections.empty()) {
            const Correction& correction = corrections.back();
            scale = dot(correction.gradient_difference, correction.gradient_difference) / dot(correction.step, correction.gradient_difference);
        }
        std::vector<double> hessian(size * size, 0.0);
        for (std::size_t parameter = 0u; parameter < size; ++parameter) hessian[parameter * size + parameter] = scale;
        for (const Correction& correction : corrections) {
            const std::vector<double> hessian_step = multiply(hessian, correction.step);
            const double step_hessian_step = dot(correction.step, hessian_step);
            const double step_gradient = dot(correction.step, correction.gradient_difference);
            for (std::size_t row = 0u; row < size; ++row) for (std::size_t column = 0u; column < size; ++column) hessian[row * size + column] += correction.gradient_difference[row] * correction.gradient_difference[column] / step_gradient - hessian_step[row] * hessian_step[column] / step_hessian_step;
        }
        return hessian;
    }

    std::vector<double> Lbfgsb::generalized_cauchy_point(const std::vector<double>& hessian) const {
        struct Breakpoint final {
            double time;
            std::size_t parameter;
        };
        std::vector<Breakpoint> breakpoints;
        std::vector<double> displacement(parameters.size(), 0.0);
        std::vector<double> path_direction(parameters.size());
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            path_direction[parameter] = -gradient[parameter];
            if (gradient[parameter] > 0.0) breakpoints.push_back({.time = (parameters[parameter] - lower_bounds[parameter]) / gradient[parameter], .parameter = parameter});
            if (gradient[parameter] < 0.0) breakpoints.push_back({.time = (upper_bounds[parameter] - parameters[parameter]) / -gradient[parameter], .parameter = parameter});
        }
        std::ranges::sort(breakpoints, {}, &Breakpoint::time);
        double time = 0.0;
        std::size_t breakpoint = 0u;
        while (breakpoint < breakpoints.size()) {
            const double next_time = breakpoints[breakpoint].time;
            const std::vector<double> hessian_direction = multiply(hessian, path_direction);
            const std::vector<double> hessian_displacement = multiply(hessian, displacement);
            const double derivative = dot(gradient, path_direction) + dot(hessian_displacement, path_direction);
            const double curvature = dot(path_direction, hessian_direction);
            const double minimizer = -derivative / curvature;
            if (minimizer <= 0.0) {
                std::vector<double> result(parameters);
                for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) result[parameter] += displacement[parameter];
                return result;
            }
            if (time + minimizer <= next_time) {
                for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) displacement[parameter] += minimizer * path_direction[parameter];
                std::vector<double> result(parameters);
                for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) result[parameter] += displacement[parameter];
                return result;
            }
            const double interval = next_time - time;
            for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) displacement[parameter] += interval * path_direction[parameter];
            time = next_time;
            while (breakpoint < breakpoints.size() && breakpoints[breakpoint].time == next_time) {
                const std::size_t parameter = breakpoints[breakpoint].parameter;
                displacement[parameter] = gradient[parameter] > 0.0 ? lower_bounds[parameter] - parameters[parameter] : upper_bounds[parameter] - parameters[parameter];
                path_direction[parameter] = 0.0;
                ++breakpoint;
            }
        }
        if (dot(path_direction, path_direction) == 0.0) {
            std::vector<double> result(parameters);
            for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) result[parameter] += displacement[parameter];
            return result;
        }
        const std::vector<double> hessian_direction = multiply(hessian, path_direction);
        const std::vector<double> hessian_displacement = multiply(hessian, displacement);
        const double minimizer = -(dot(gradient, path_direction) + dot(hessian_displacement, path_direction)) / dot(path_direction, hessian_direction);
        if (minimizer > 0.0) for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) displacement[parameter] += minimizer * path_direction[parameter];
        std::vector<double> result(parameters);
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) result[parameter] += displacement[parameter];
        return result;
    }

    std::vector<double> Lbfgsb::subspace_minimum(const std::vector<double>& hessian, const std::vector<double>& cauchy) const {
        std::vector<std::size_t> free_parameters;
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) if (cauchy[parameter] > lower_bounds[parameter] && cauchy[parameter] < upper_bounds[parameter]) free_parameters.push_back(parameter);
        if (free_parameters.empty()) return cauchy;
        std::vector<double> displacement(cauchy.size());
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) displacement[parameter] = cauchy[parameter] - parameters[parameter];
        const std::vector<double> hessian_displacement = multiply(hessian, displacement);
        const std::size_t free_count = free_parameters.size();
        std::vector<double> reduced_hessian(free_count * free_count);
        std::vector<double> right_hand_side(free_count);
        for (std::size_t row = 0u; row < free_count; ++row) {
            right_hand_side[row] = -gradient[free_parameters[row]] - hessian_displacement[free_parameters[row]];
            for (std::size_t column = 0u; column < free_count; ++column) reduced_hessian[row * free_count + column] = hessian[free_parameters[row] * parameters.size() + free_parameters[column]];
        }
        const std::vector<double> reduced_step = solve_positive_definite(std::move(reduced_hessian), std::move(right_hand_side));
        double step = 1.0;
        for (std::size_t index = 0u; index < free_count; ++index) {
            const std::size_t parameter = free_parameters[index];
            if (reduced_step[index] > 0.0) step = std::min(step, (upper_bounds[parameter] - cauchy[parameter]) / reduced_step[index]);
            if (reduced_step[index] < 0.0) step = std::min(step, (lower_bounds[parameter] - cauchy[parameter]) / reduced_step[index]);
        }
        std::vector<double> result(cauchy);
        for (std::size_t index = 0u; index < free_count; ++index) result[free_parameters[index]] += step * reduced_step[index];
        return result;
    }

    void Lbfgsb::begin_line_search() {
        const std::vector<double> hessian = limited_memory_hessian();
        const std::vector<double> cauchy = generalized_cauchy_point(hessian);
        const std::vector<double> subspace = subspace_minimum(hessian, cauchy);
        maximum_step = std::numeric_limits<double>::infinity();
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            direction[parameter] = subspace[parameter] - parameters[parameter];
            if (direction[parameter] > 0.0) maximum_step = std::min(maximum_step, (upper_bounds[parameter] - parameters[parameter]) / direction[parameter]);
            if (direction[parameter] < 0.0) maximum_step = std::min(maximum_step, (lower_bounds[parameter] - parameters[parameter]) / direction[parameter]);
        }
        initial_directional_derivative = dot(gradient, direction);
        if (initial_directional_derivative >= 0.0) {
            finish(LbfgsbStopReason::non_descent_direction);
            return;
        }
        phase = Phase::line_search;
        line_search_evaluation = 0u;
        step_length = std::min(1.0 / std::sqrt(dot(direction, direction)), maximum_step);
        low_step = 0.0;
        low_objective = objective;
        high_step = 0.0;
        high_step_exists = false;
        set_trial();
    }

    void Lbfgsb::set_trial() {
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) trial_parameters[parameter] = std::clamp(parameters[parameter] + step_length * direction[parameter], lower_bounds[parameter], upper_bounds[parameter]);
    }

    void Lbfgsb::accept_trial(const double next_objective, const std::span<const double> next_gradient) {
        const double previous_objective = objective;
        Correction correction{.step = std::vector<double>(parameters.size()), .gradient_difference = std::vector<double>(parameters.size())};
        for (std::size_t parameter = 0u; parameter < parameters.size(); ++parameter) {
            correction.step[parameter] = trial_parameters[parameter] - parameters[parameter];
            correction.gradient_difference[parameter] = next_gradient[parameter] - gradient[parameter];
        }
        const double curvature = dot(correction.step, correction.gradient_difference);
        const double step_squared = dot(correction.step, correction.step);
        const bool correction_accepted = curvature > 1.0e-10 * step_squared;
        if (correction_accepted) {
            corrections.push_back(std::move(correction));
            if (corrections.size() > configuration.memory) corrections.erase(corrections.begin());
        }
        parameters = trial_parameters;
        objective = next_objective;
        std::ranges::copy(next_gradient, gradient.begin());
        ++iteration;
        iterations.push_back({
            .iteration = iteration,
            .evaluation = evaluation,
            .objective = objective,
            .projected_gradient_norm = projected_gradient_norm(),
            .step_length = step_length,
            .line_search_evaluations = line_search_evaluation,
            .correction_count = static_cast<std::uint32_t>(corrections.size()),
            .correction_accepted = correction_accepted,
        });
        if (projected_gradient_norm() <= configuration.projected_gradient_tolerance) {
            finish(LbfgsbStopReason::projected_gradient);
            return;
        }
        if (std::abs(previous_objective - objective) <= configuration.relative_objective_tolerance * std::max({1.0, std::abs(previous_objective), std::abs(objective)})) {
            finish(LbfgsbStopReason::relative_objective);
            return;
        }
        if (iteration >= configuration.maximum_iterations) {
            finish(LbfgsbStopReason::maximum_iterations);
            return;
        }
        if (evaluation >= configuration.maximum_evaluations) {
            finish(LbfgsbStopReason::maximum_evaluations);
            return;
        }
        begin_line_search();
    }

    void Lbfgsb::finish(const LbfgsbStopReason reason) {
        phase = Phase::complete;
        stop_reason = reason;
    }
} // namespace physica::fluids::gas::keyframe_smoke
