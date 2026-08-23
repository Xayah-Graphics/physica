module;

#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/std/span>

module physica.fluids.gas.keyframe_smoke.evaluation;

import std;

namespace physica::fluids::gas::keyframe_smoke {
    EvaluationTrace::EvaluationTrace() = default;

    EvaluationTrace::~EvaluationTrace() = default;

    EvaluationTrace::EvaluationTrace(EvaluationTrace&&) noexcept = default;

    EvaluationTrace& EvaluationTrace::operator=(EvaluationTrace&&) noexcept = default;

    OptimizationResult::OptimizationResult() = default;

    OptimizationResult::~OptimizationResult() = default;

    OptimizationResult::OptimizationResult(OptimizationResult&&) noexcept = default;

    OptimizationResult& OptimizationResult::operator=(OptimizationResult&&) noexcept = default;

    Evaluator::Evaluator(const Domain& next_domain, Solver& next_solver, ControlSystem& next_control, Objective& next_objective, const Problem& next_problem)
        : domain(next_domain), solver(next_solver), control(next_control), objective(next_objective), problem(next_problem) {}

    EvaluationTrace Evaluator::evaluate(const std::span<const double> parameter_values, const EvaluationMode mode, const std::span<const double> direction) {
        EvaluationTrace trace{};
        trace.evaluation = evaluation_count++;
        trace.parameters.assign(parameter_values.begin(), parameter_values.end());
        trace.state.reserve(problem.step_count + 1u);
        trace.steps.reserve(problem.step_count);
        trace.keyframes.reserve(problem.keyframes.size());
        trace.state.push_back(solver.allocate_state(domain));
        solver.copy(domain, problem.initial_state, trace.state.front());
        control.upload_parameters(domain, parameter_values);

        if (mode == EvaluationMode::objective_gradient_jvp) {
            trace.tangent.emplace(TangentTrace{.direction = {direction.begin(), direction.end()}});
            trace.tangent->state.reserve(problem.step_count + 1u);
            trace.tangent->control.reserve(problem.step_count);
            trace.tangent->state.push_back(solver.allocate_state_tangent(domain));
            solver.clear(domain, trace.tangent->state.front());
            control.upload_direction(domain, direction);
        }

        for (std::uint32_t state_index = 0u; state_index <= problem.step_count; ++state_index) {
            if (state_index != 0u) {
                trace.state.push_back(solver.allocate_state(domain));
                StepRecord record{
                    .step = problem.begin_step + state_index - 1u,
                    .control = solver.allocate_control(domain),
                    .cache = solver.allocate_step_cache(domain),
                    .objective = objective.allocate_step_objective_cache(domain),
                };
                control.forward(domain, record.step, record.control);
                objective.evaluate_control_effort(domain, record.control, record.objective);
                solver.forward(domain, trace.state[state_index - 1u], record.control, trace.state[state_index], record.cache);
                if (trace.tangent) {
                    trace.tangent->control.push_back(solver.allocate_control_tangent(domain));
                    trace.tangent->state.push_back(solver.allocate_state_tangent(domain));
                    control.jvp(domain, record.step, trace.tangent->control.back());
                    objective.control_effort_jvp(domain, record.control, trace.tangent->control.back(), record.objective);
                    solver.jvp(domain, trace.state[state_index - 1u], record.control, record.cache, trace.tangent->state[state_index - 1u], trace.tangent->control.back(), trace.tangent->state[state_index]);
                }
                trace.steps.push_back(std::move(record));
            }

            for (std::size_t keyframe_index = 0u; keyframe_index < problem.keyframes.size(); ++keyframe_index) {
                const Keyframe& keyframe = problem.keyframes[keyframe_index];
                if (keyframe.step != problem.begin_step + state_index) continue;
                KeyframeRecord record{.keyframe_index = keyframe_index, .cache = objective.allocate_keyframe_cache(domain)};
                objective.evaluate_keyframe(domain, trace.state[state_index], keyframe, record.cache);
                if (trace.tangent) objective.keyframe_jvp(domain, trace.tangent->state[state_index], keyframe, record.cache);
                trace.keyframes.push_back(std::move(record));
            }
        }

        if (mode != EvaluationMode::objective) {
            trace.reverse.emplace();
            trace.reverse->state.reserve(problem.step_count + 1u);
            trace.reverse->control.reserve(problem.step_count);
            for (std::uint32_t state_index = 0u; state_index <= problem.step_count; ++state_index) {
                trace.reverse->state.push_back(solver.allocate_state_adjoint(domain));
                solver.clear(domain, trace.reverse->state.back());
            }
            for (std::uint32_t step = 0u; step < problem.step_count; ++step) {
                trace.reverse->control.push_back(solver.allocate_control_adjoint(domain));
                solver.clear(domain, trace.reverse->control.back());
            }
            control.clear_gradient(domain);
            for (std::uint32_t reverse_index = 0u; reverse_index <= problem.step_count; ++reverse_index) {
                const std::uint32_t state_index = problem.step_count - reverse_index;
                for (KeyframeRecord& record : trace.keyframes) {
                    const Keyframe& keyframe = problem.keyframes[record.keyframe_index];
                    if (keyframe.step == problem.begin_step + state_index) objective.keyframe_vjp(domain, keyframe, record.cache, trace.reverse->state[state_index]);
                }
                if (state_index == 0u) continue;
                const std::uint32_t step = state_index - 1u;
                solver.vjp(domain, trace.state[step], trace.steps[step].control, trace.steps[step].cache, trace.reverse->state[state_index], trace.reverse->state[step], trace.reverse->control[step]);
                objective.control_effort_vjp(domain, trace.steps[step].control, trace.reverse->control[step]);
                control.vjp(domain, problem.begin_step + step, trace.reverse->control[step]);
            }
            trace.reverse->parameter_gradient.resize(parameter_values.size());
            control.download_gradient(domain, trace.reverse->parameter_gradient);
        }

        for (StepRecord& record : trace.steps) {
            ::cuda::copy_bytes(domain.stream, record.cache.input_mass, ::cuda::std::span{&record.metrics.input_mass, 1u});
            ::cuda::copy_bytes(domain.stream, record.cache.advected_mass, ::cuda::std::span{&record.metrics.advected_mass, 1u});
            ::cuda::copy_bytes(domain.stream, record.objective.control_effort, ::cuda::std::span{&record.metrics.control_effort, 1u});
            if (trace.tangent) ::cuda::copy_bytes(domain.stream, record.objective.directional_derivative, ::cuda::std::span{&record.metrics.directional_derivative, 1u});
        }
        for (KeyframeRecord& record : trace.keyframes) {
            ::cuda::copy_bytes(domain.stream, record.cache.density_loss, ::cuda::std::span{&record.metrics.density_loss, 1u});
            ::cuda::copy_bytes(domain.stream, record.cache.velocity_loss, ::cuda::std::span{&record.metrics.velocity_loss, 1u});
            if (trace.tangent) ::cuda::copy_bytes(domain.stream, record.cache.directional_derivative, ::cuda::std::span{&record.metrics.directional_derivative, 1u});
        }
        domain.stream.sync();

        for (const StepRecord& record : trace.steps) {
            trace.summary.control_effort += record.metrics.control_effort;
            trace.summary.directional_derivative += record.metrics.directional_derivative;
        }
        for (const KeyframeRecord& record : trace.keyframes) {
            trace.summary.density_loss += record.metrics.density_loss;
            trace.summary.velocity_loss += record.metrics.velocity_loss;
            trace.summary.directional_derivative += record.metrics.directional_derivative;
        }
        trace.summary.objective = trace.summary.density_loss + trace.summary.velocity_loss + trace.summary.control_effort;

        if (trace.reverse) {
            double gradient_squared = 0.0;
            double projected_gradient_squared = 0.0;
            for (std::size_t parameter = 0u; parameter < parameter_values.size(); ++parameter) {
                const double gradient = trace.reverse->parameter_gradient[parameter];
                gradient_squared += gradient * gradient;
                const double projected = parameter_values[parameter] - std::clamp(parameter_values[parameter] - gradient, control.parameters.lower_bounds[parameter], control.parameters.upper_bounds[parameter]);
                projected_gradient_squared += projected * projected;
                if (trace.tangent) trace.summary.gradient_dot_direction += gradient * direction[parameter];
            }
            trace.summary.gradient_norm = std::sqrt(gradient_squared);
            trace.summary.projected_gradient_norm = std::sqrt(projected_gradient_squared);
        }
        return trace;
    }
} // namespace physica::fluids::gas::keyframe_smoke
