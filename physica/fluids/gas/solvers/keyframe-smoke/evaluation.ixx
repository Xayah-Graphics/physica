module;

#include <physica/cuda.h>

export module physica.fluids.gas.solvers.keyframe_smoke.evaluation;

import std;
import physica.fluids.gas.domain;
import physica.fluids.gas.solvers.keyframe_smoke;
import physica.fluids.gas.solvers.keyframe_smoke.control;
import physica.fluids.gas.operators.objective;

export namespace physica::fluids::gas::solvers::keyframe_smoke {
    enum class EvaluationMode : std::uint32_t {
        objective,
        objective_gradient,
        objective_gradient_jvp,
    };

    struct EvaluationSummary final {
        double objective{};
        double density_loss{};
        double velocity_loss{};
        double control_effort{};
        double directional_derivative{};
    };

    struct ReverseTrace final {
        ::cuda::device_buffer<double> parameter_gradient;
    };

    struct TangentTrace final {
        std::vector<StateTangent> state;
    };

    struct StepMetrics final {
        double input_mass{};
        double advected_mass{};
        double control_effort{};
        double directional_derivative{};
    };

    struct KeyframeMetrics final {
        double density_loss{};
        double velocity_loss{};
        double directional_derivative{};
    };

    struct StepRecord final {
        std::uint32_t step{};
        StepMetrics metrics;
    };

    struct KeyframeRecord final {
        std::size_t keyframe_index{};
        KeyframeMetrics metrics;
    };

    struct EvaluationTrace final {
        explicit EvaluationTrace(const Domain& domain) : objective(domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 1u, ::cuda::no_init) {
            ::cuda::fill_bytes(domain.grid.stream, ::cuda::std::span{objective.data(), objective.size()}, 0u);
        }

        std::uint64_t evaluation{};
        ::cuda::device_buffer<double> objective;
        std::vector<State> state;
        std::vector<StepRecord> steps;
        std::vector<KeyframeRecord> keyframes;
        std::optional<ReverseTrace> reverse;
        std::optional<TangentTrace> tangent;
        EvaluationSummary summary;
    };

    template <typename SolverType>
    struct Evaluator final {
        Evaluator(const Domain& next_domain, SolverType& next_solver, ControlSystem& next_control, operators::Quadratic& next_objective, const Problem& next_problem) : domain(next_domain), solver(next_solver), control(next_control), objective_function(next_objective), problem(next_problem) {}

        [[nodiscard]] EvaluationTrace evaluate(::cuda::std::span<const double> parameters, EvaluationMode mode = EvaluationMode::objective_gradient, ::cuda::std::span<const double> direction = {});
        [[nodiscard]] EvaluationTrace evaluate(operators::Quadratic& objective_function, ::cuda::std::span<const double> parameters, EvaluationMode mode, ::cuda::std::span<const double> direction = {});

    private:
        const Domain& domain;
        SolverType& solver;
        ControlSystem& control;
        operators::Quadratic& objective_function;
        const Problem& problem;
        std::uint64_t evaluation_count{};
    };

    template <typename SolverType>
    EvaluationTrace Evaluator<SolverType>::evaluate(const ::cuda::std::span<const double> parameter_values, const EvaluationMode mode, const ::cuda::std::span<const double> direction) {
        return evaluate(objective_function, parameter_values, mode, direction);
    }

    template <typename SolverType>
    EvaluationTrace Evaluator<SolverType>::evaluate(operators::Quadratic& selected_objective, const ::cuda::std::span<const double> parameter_values, const EvaluationMode mode, const ::cuda::std::span<const double> direction) {
        EvaluationTrace trace{domain};
        trace.evaluation = evaluation_count++;
        trace.state.reserve(problem.step_count + 1u);
        trace.steps.reserve(problem.step_count);
        trace.keyframes.reserve(problem.keyframes.size());
        trace.state.push_back(solver.allocate_state(domain));
        domain.grid.copy(problem.initial_state.density, trace.state.front().density);
        domain.grid.copy(problem.initial_state.velocity, trace.state.front().velocity);

        DenseControl dense_control                       = solver.allocate_control(domain);
        typename SolverType::StepCache step_cache        = solver.allocate_step_cache(domain);
        typename SolverType::Workspace forward_workspace = solver.allocate_workspace(domain);
        operators::StepCache step_objective              = selected_objective.allocate_step_cache(domain);
        operators::KeyframeCache keyframe_cache          = selected_objective.allocate_keyframe_cache(domain);
        operators::Workspace objective_workspace         = selected_objective.allocate_workspace(domain);
        std::optional<typename SolverType::TangentWorkspace> tangent_workspace;
        std::optional<DenseControlTangent> dense_control_tangent;

        if (mode == EvaluationMode::objective_gradient_jvp) {
            tangent_workspace.emplace(solver.allocate_tangent_workspace(domain));
            dense_control_tangent.emplace(solver.allocate_control_tangent(domain));
            trace.tangent.emplace();
            trace.tangent->state.reserve(problem.step_count + 1u);
            trace.tangent->state.push_back(solver.allocate_state_tangent(domain));
        }

        for (std::uint32_t state_index = 0u; state_index <= problem.step_count; ++state_index) {
            if (state_index != 0u) {
                const std::uint32_t step = problem.begin_step + state_index - 1u;
                trace.state.push_back(solver.allocate_state(domain));
                trace.steps.push_back({.step = step});
                StepRecord& record = trace.steps.back();
                control.forward(domain, step, parameter_values, dense_control);
                selected_objective.evaluate_control_effort(domain, dense_control.force, step_objective);
                selected_objective.accumulate(domain, step_objective.control_effort, trace.objective);
                solver.forward(domain, trace.state[state_index - 1u], dense_control, trace.state[state_index], step_cache, forward_workspace);
                ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span{step_cache.conservation.input_mass.data(), step_cache.conservation.input_mass.size()}, ::cuda::std::span{&record.metrics.input_mass, 1u});
                ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span{step_cache.conservation.advected_mass.data(), step_cache.conservation.advected_mass.size()}, ::cuda::std::span{&record.metrics.advected_mass, 1u});
                ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span{step_objective.control_effort.data(), step_objective.control_effort.size()}, ::cuda::std::span{&record.metrics.control_effort, 1u});
                if (trace.tangent) {
                    trace.tangent->state.push_back(solver.allocate_state_tangent(domain));
                    control.jvp(domain, step, parameter_values, direction, *dense_control_tangent);
                    selected_objective.control_effort_jvp(domain, dense_control.force, dense_control_tangent->force, step_objective);
                    solver.jvp(domain, trace.state[state_index - 1u], trace.state[state_index], step_cache, trace.tangent->state[state_index - 1u], *dense_control_tangent, trace.tangent->state[state_index], *tangent_workspace);
                    ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span{step_objective.directional_derivative.data(), step_objective.directional_derivative.size()}, ::cuda::std::span{&record.metrics.directional_derivative, 1u});
                }
            }

            for (std::size_t keyframe_index = 0u; keyframe_index < problem.keyframes.size(); ++keyframe_index) {
                const Keyframe& keyframe = problem.keyframes[keyframe_index];
                if (keyframe.step != problem.begin_step + state_index) continue;
                trace.keyframes.push_back({.keyframe_index = keyframe_index});
                KeyframeRecord& record = trace.keyframes.back();
                selected_objective.evaluate_keyframe(domain, trace.state[state_index].density, trace.state[state_index].velocity, keyframe.target.density, keyframe.target.velocity, keyframe.density_weight, keyframe.velocity_weight, keyframe_cache, objective_workspace);
                selected_objective.accumulate(domain, keyframe_cache.density_loss, trace.objective);
                selected_objective.accumulate(domain, keyframe_cache.velocity_loss, trace.objective);
                ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span{keyframe_cache.density_loss.data(), keyframe_cache.density_loss.size()}, ::cuda::std::span{&record.metrics.density_loss, 1u});
                ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span{keyframe_cache.velocity_loss.data(), keyframe_cache.velocity_loss.size()}, ::cuda::std::span{&record.metrics.velocity_loss, 1u});
                if (trace.tangent) {
                    selected_objective.keyframe_jvp(domain, trace.tangent->state[state_index].density, trace.tangent->state[state_index].velocity, keyframe.density_weight, keyframe.velocity_weight, keyframe_cache, objective_workspace);
                    ::cuda::copy_bytes(domain.grid.stream, ::cuda::std::span{keyframe_cache.directional_derivative.data(), keyframe_cache.directional_derivative.size()}, ::cuda::std::span{&record.metrics.directional_derivative, 1u});
                }
            }
        }
        domain.grid.stream.sync();

        if (mode != EvaluationMode::objective) {
            trace.reverse.emplace(ReverseTrace{.parameter_gradient = ::cuda::device_buffer<double>{domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), parameter_values.size(), ::cuda::no_init}});
            ::cuda::fill_bytes(domain.grid.stream, ::cuda::std::span{trace.reverse->parameter_gradient.data(), trace.reverse->parameter_gradient.size()}, 0u);
            typename SolverType::AdjointWorkspace adjoint_workspace = solver.allocate_adjoint_workspace(domain);
            State replay                                            = solver.allocate_state(domain);
            StateAdjoint current                                    = solver.allocate_state_adjoint(domain);
            StateAdjoint previous                                   = solver.allocate_state_adjoint(domain);
            DenseControlAdjoint control_adjoint                     = solver.allocate_control_adjoint(domain);
            for (std::uint32_t reverse_index = 0u; reverse_index <= problem.step_count; ++reverse_index) {
                const std::uint32_t state_index = problem.step_count - reverse_index;
                for (const Keyframe& keyframe : problem.keyframes) {
                    if (keyframe.step != problem.begin_step + state_index) continue;
                    selected_objective.evaluate_keyframe(domain, trace.state[state_index].density, trace.state[state_index].velocity, keyframe.target.density, keyframe.target.velocity, keyframe.density_weight, keyframe.velocity_weight, keyframe_cache, objective_workspace);
                    selected_objective.keyframe_vjp(domain, keyframe.density_weight, keyframe.velocity_weight, keyframe_cache, current.density, current.velocity, objective_workspace);
                }
                if (state_index == 0u) continue;
                const std::uint32_t step = problem.begin_step + state_index - 1u;
                control.forward(domain, step, parameter_values, dense_control);
                solver.forward(domain, trace.state[state_index - 1u], dense_control, replay, step_cache, forward_workspace);
                solver.vjp(domain, trace.state[state_index - 1u], replay, step_cache, current, previous, control_adjoint, adjoint_workspace);
                selected_objective.control_effort_vjp(domain, dense_control.force, control_adjoint.force);
                control.vjp(domain, step, parameter_values, control_adjoint, {trace.reverse->parameter_gradient.data(), trace.reverse->parameter_gradient.size()});
                std::swap(current, previous);
            }
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

        return trace;
    }
} // namespace physica::fluids::gas::solvers::keyframe_smoke
