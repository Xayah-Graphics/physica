module;

#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>

module physica.fluids.gas.adjoint_control.evaluation;

import std;

namespace physica::fluids::gas::adjoint_control {
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
        trace.state.reserve(problem.step_count + 1u);
        trace.steps.reserve(problem.step_count);
        trace.keyframes.reserve(problem.keyframes.size());
        trace.state.push_back(solver.allocate_state(domain));
        solver.copy(domain, problem.initial_state, trace.state.front());
        control.upload_parameters(domain, parameter_values);

        DenseControl dense_control = solver.allocate_control(domain);
        StepCache step_cache = solver.allocate_step_cache(domain);
        StepObjectiveCache step_objective = objective.allocate_step_objective_cache(domain);
        KeyframeCache keyframe_cache = objective.allocate_keyframe_cache(domain);

        DenseControlTangent dense_control_tangent = solver.allocate_control_tangent(domain);
        if (mode == EvaluationMode::objective_gradient_jvp) {
            trace.tangent.emplace();
            trace.tangent->state.reserve(problem.step_count + 1u);
            trace.tangent->state.push_back(solver.allocate_state_tangent(domain));
            solver.clear(domain, trace.tangent->state.front());
            control.upload_direction(domain, direction);
        }

        for (std::uint32_t state_index = 0u; state_index <= problem.step_count; ++state_index) {
            if (state_index != 0u) {
                const std::uint32_t step = problem.begin_step + state_index - 1u;
                trace.state.push_back(solver.allocate_state(domain));
                trace.steps.push_back({.step = step});
                StepRecord& record = trace.steps.back();
                control.forward(domain, step, dense_control);
                objective.evaluate_control_effort(domain, dense_control, step_objective);
                solver.forward(domain, trace.state[state_index - 1u], dense_control, trace.state[state_index], step_cache);
                ::cuda::copy_bytes(domain.stream, step_cache.input_mass, ::cuda::std::span{&record.metrics.input_mass, 1u});
                ::cuda::copy_bytes(domain.stream, step_cache.advected_mass, ::cuda::std::span{&record.metrics.advected_mass, 1u});
                ::cuda::copy_bytes(domain.stream, step_objective.control_effort, ::cuda::std::span{&record.metrics.control_effort, 1u});
                if (trace.tangent) {
                    trace.tangent->state.push_back(solver.allocate_state_tangent(domain));
                    control.jvp(domain, step, dense_control_tangent);
                    objective.control_effort_jvp(domain, dense_control, dense_control_tangent, step_objective);
                    solver.jvp(domain, trace.state[state_index - 1u], dense_control, step_cache, trace.tangent->state[state_index - 1u], dense_control_tangent, trace.tangent->state[state_index]);
                    ::cuda::copy_bytes(domain.stream, step_objective.directional_derivative, ::cuda::std::span{&record.metrics.directional_derivative, 1u});
                }
            }

            for (std::size_t keyframe_index = 0u; keyframe_index < problem.keyframes.size(); ++keyframe_index) {
                const Keyframe& keyframe = problem.keyframes[keyframe_index];
                if (keyframe.step != problem.begin_step + state_index) continue;
                trace.keyframes.push_back({.keyframe_index = keyframe_index});
                KeyframeRecord& record = trace.keyframes.back();
                objective.evaluate_keyframe(domain, trace.state[state_index], keyframe, keyframe_cache);
                ::cuda::copy_bytes(domain.stream, keyframe_cache.density_loss, ::cuda::std::span{&record.metrics.density_loss, 1u});
                ::cuda::copy_bytes(domain.stream, keyframe_cache.velocity_loss, ::cuda::std::span{&record.metrics.velocity_loss, 1u});
                if (trace.tangent) {
                    objective.keyframe_jvp(domain, trace.tangent->state[state_index], keyframe, keyframe_cache);
                    ::cuda::copy_bytes(domain.stream, keyframe_cache.directional_derivative, ::cuda::std::span{&record.metrics.directional_derivative, 1u});
                }
            }
        }
        domain.stream.sync();

        if (mode != EvaluationMode::objective) {
            trace.reverse.emplace();
            State replay = solver.allocate_state(domain);
            StateAdjoint current = solver.allocate_state_adjoint(domain);
            StateAdjoint previous = solver.allocate_state_adjoint(domain);
            DenseControlAdjoint control_adjoint = solver.allocate_control_adjoint(domain);
            solver.clear(domain, current);
            control.clear_gradient(domain);
            for (std::uint32_t reverse_index = 0u; reverse_index <= problem.step_count; ++reverse_index) {
                const std::uint32_t state_index = problem.step_count - reverse_index;
                for (std::size_t keyframe_index = 0u; keyframe_index < problem.keyframes.size(); ++keyframe_index) {
                    const Keyframe& keyframe = problem.keyframes[keyframe_index];
                    if (keyframe.step != problem.begin_step + state_index) continue;
                    objective.evaluate_keyframe(domain, trace.state[state_index], keyframe, keyframe_cache);
                    objective.keyframe_vjp(domain, keyframe, keyframe_cache, current);
                }
                if (state_index == 0u) continue;
                const std::uint32_t step = problem.begin_step + state_index - 1u;
                control.forward(domain, step, dense_control);
                solver.forward(domain, trace.state[state_index - 1u], dense_control, replay, step_cache);
                solver.vjp(domain, trace.state[state_index - 1u], dense_control, step_cache, current, previous, control_adjoint);
                objective.control_effort_vjp(domain, dense_control, control_adjoint);
                control.vjp(domain, step, control_adjoint);
                std::swap(current, previous);
            }
            trace.reverse->parameter_gradient.resize(parameter_values.size());
            control.download_gradient(domain, trace.reverse->parameter_gradient);
        }

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
                const double gradient_value = trace.reverse->parameter_gradient[parameter];
                gradient_squared += gradient_value * gradient_value;
                const double projected = parameter_values[parameter] - std::clamp(parameter_values[parameter] - gradient_value, control.parameters.lower_bounds[parameter], control.parameters.upper_bounds[parameter]);
                projected_gradient_squared += projected * projected;
                if (trace.tangent) trace.summary.gradient_dot_direction += gradient_value * direction[parameter];
            }
            trace.summary.gradient_norm = std::sqrt(gradient_squared);
            trace.summary.projected_gradient_norm = std::sqrt(projected_gradient_squared);
        }
        return trace;
    }
} // namespace physica::fluids::gas::adjoint_control
