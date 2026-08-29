module;

#include <physica/cuda.h>

export module physica.fluids.gas.keyframe_smoke.multiple_shooting;

import std;
import physica.fluids.gas.keyframe_smoke;
import physica.fluids.gas.keyframe_smoke.control;
import physica.fluids.gas.domain;
import physica.fluids.gas.keyframe_smoke.evaluation;
import physica.fluids.gas.operators.objective;
import physica.fluids.gas.keyframe_smoke.optimization;
import physica.optimization.lbfgsb;

export namespace physica::fluids::gas::keyframe_smoke {
    struct ShootingSegment final {
        Problem problem;
        std::vector<std::uint8_t> active_parameters;
    };

    enum class ShootingScheduleKind : std::uint32_t {
        initial,
        alternate,
    };

    enum class ShootingProcessing : std::uint32_t {
        independent,
        sequential,
    };

    struct LayeredMultipleShootingConfiguration final {
        std::uint32_t independent_schedule_sets{2u};
        std::uint32_t sequential_schedule_sets{2u};
        double pseudo_density_weight{1.0};
        double pseudo_velocity_weight{1.0};
        double propagated_velocity_weight{1.0};
        std::vector<ContinuationLevel> continuation;
    };

    struct ShootingSchedule final {
        ShootingScheduleKind kind{};
        std::vector<ShootingSegment> segments;
    };

    struct SegmentOptimization final {
        std::uint32_t begin_step{};
        std::uint32_t end_step{};
        OptimizationResult optimization;
    };

    struct ShootingPass final {
        std::uint32_t pass{};
        ShootingScheduleKind schedule{};
        ShootingProcessing processing{};
        std::vector<SegmentOptimization> segments;
        std::vector<double> parameters;
    };

    struct SegmentOptimizationSummary final {
        std::uint32_t begin_step{};
        std::uint32_t end_step{};
        std::vector<double> parameters;
        std::vector<OptimizationEvaluation> evaluations;
        std::vector<optimization::LbfgsbStopReason> level_stop_reasons;
        EvaluationSummary final_summary;
    };

    struct ShootingPassSummary final {
        std::uint32_t pass{};
        ShootingScheduleKind schedule{};
        ShootingProcessing processing{};
        std::vector<SegmentOptimizationSummary> segments;
        std::vector<double> parameters;
    };

    struct LayeredMultipleShootingResult final {
        std::vector<double> parameters;
        std::vector<ShootingPassSummary> passes;
        std::optional<EvaluationTrace> final_trace;
    };

    template <typename SolverType>
    struct LayeredMultipleShooting final {
        const Domain& domain;
        SolverType& solver;
        ControlSystem& control;
        operators::Quadratic& objective_function;
        const LayeredMultipleShootingConfiguration configuration;

        [[nodiscard]] LayeredMultipleShootingResult run(const Problem& problem, std::span<const double> initial_parameters);

    private:
        [[nodiscard]] State copy_state(const State& source) const;
        [[nodiscard]] std::vector<std::uint32_t> boundary_steps(const Problem& problem) const;
        [[nodiscard]] std::vector<State> initial_boundary_states(const Problem& problem, std::span<const std::uint32_t> boundaries) const;
        [[nodiscard]] ShootingSchedule initial_schedule(const Problem& problem, std::span<const std::uint32_t> boundaries, const std::vector<State>& states, bool propagated) const;
        [[nodiscard]] ShootingSchedule alternate_schedule(const Problem& problem, const ShootingSchedule& initial, const ShootingPass& pass) const;
        [[nodiscard]] ShootingPass process_schedule(ShootingSchedule& schedule, ShootingProcessing processing, std::uint32_t pass, std::span<const double> parameters);
        [[nodiscard]] std::vector<State> propagate_boundary_states(const Problem& problem, std::span<const std::uint32_t> boundaries, const std::vector<State>& current, const ShootingSchedule& alternate, const ShootingPass& pass) const;
        [[nodiscard]] static ShootingPassSummary summarize(const ShootingPass& pass);
    };

    template <typename SolverType>
    LayeredMultipleShootingResult LayeredMultipleShooting<SolverType>::run(const Problem& problem, const std::span<const double> initial_parameters) {
        LayeredMultipleShootingResult result{.parameters = {initial_parameters.begin(), initial_parameters.end()}};
        const std::vector<std::uint32_t> boundaries = boundary_steps(problem);
        std::vector<State> states                   = initial_boundary_states(problem, boundaries);
        bool propagated                             = false;
        const std::uint32_t schedule_sets           = configuration.independent_schedule_sets + configuration.sequential_schedule_sets;
        result.passes.reserve(schedule_sets * 2u);
        for (std::uint32_t pass = 0u; pass < schedule_sets; ++pass) {
            const ShootingProcessing processing = pass < configuration.independent_schedule_sets ? ShootingProcessing::independent : ShootingProcessing::sequential;
            ShootingSchedule initial            = initial_schedule(problem, boundaries, states, propagated);
            ShootingPass initial_pass           = process_schedule(initial, processing, pass, result.parameters);
            ShootingSchedule alternate          = alternate_schedule(problem, initial, initial_pass);
            ShootingPass alternate_pass         = process_schedule(alternate, processing, pass, initial_pass.parameters);
            states                              = propagate_boundary_states(problem, boundaries, states, alternate, alternate_pass);
            result.parameters                   = alternate_pass.parameters;
            result.passes.push_back(summarize(initial_pass));
            result.passes.push_back(summarize(alternate_pass));
            propagated = true;
        }
        Evaluator evaluator(domain, solver, control, objective_function, problem);
        ::cuda::device_buffer<double> final_parameters(domain.grid.fields.stream, ::cuda::device_default_memory_pool(domain.grid.fields.stream.device()), result.parameters.size(), ::cuda::no_init);
        ::cuda::copy_bytes(domain.grid.fields.stream, ::cuda::std::span<const double>{result.parameters.data(), result.parameters.size()}, ::cuda::std::span{final_parameters.data(), final_parameters.size()});
        result.final_trace.emplace(evaluator.evaluate({final_parameters.data(), final_parameters.size()}, EvaluationMode::objective_gradient));
        return result;
    }

    template <typename SolverType>
    State LayeredMultipleShooting<SolverType>::copy_state(const State& source) const {
        State result = solver.allocate_state(domain);
        domain.grid.copy(source.density, result.density);
        domain.grid.copy(source.velocity, result.velocity);
        return result;
    }

    template <typename SolverType>
    std::vector<std::uint32_t> LayeredMultipleShooting<SolverType>::boundary_steps(const Problem& problem) const {
        std::vector<std::uint32_t> result{problem.begin_step};
        for (const Keyframe& keyframe : problem.keyframes) result.push_back(keyframe.step);
        std::ranges::sort(result);
        const auto unique_end = std::ranges::unique(result).begin();
        result.erase(unique_end, result.end());
        return result;
    }

    template <typename SolverType>
    std::vector<State> LayeredMultipleShooting<SolverType>::initial_boundary_states(const Problem& problem, const std::span<const std::uint32_t> boundaries) const {
        std::vector<State> result;
        result.reserve(boundaries.size());
        result.push_back(copy_state(problem.initial_state));
        for (std::size_t boundary = 1u; boundary < boundaries.size(); ++boundary) {
            const auto keyframe = std::ranges::find_if(problem.keyframes, [&](const Keyframe& value) { return value.step == boundaries[boundary]; });
            result.push_back(copy_state(keyframe->target));
        }
        return result;
    }

    template <typename SolverType>
    ShootingSchedule LayeredMultipleShooting<SolverType>::initial_schedule(const Problem& problem, const std::span<const std::uint32_t> boundaries, const std::vector<State>& states, const bool propagated) const {
        ShootingSchedule result{.kind = ShootingScheduleKind::initial};
        result.segments.reserve(boundaries.size() - 1u);
        for (std::size_t segment = 0u; segment + 1u < boundaries.size(); ++segment) {
            const std::uint32_t begin_step = boundaries[segment];
            const std::uint32_t end_step   = boundaries[segment + 1u];
            const auto original            = std::ranges::find_if(problem.keyframes, [&](const Keyframe& value) { return value.step == end_step; });
            const double velocity_weight   = original->velocity_weight != 0.0 ? original->velocity_weight : propagated && segment + 1u < boundaries.size() - 1u ? configuration.propagated_velocity_weight : 0.0;
            std::vector<Keyframe> keyframes;
            keyframes.push_back({
                .step            = end_step,
                .target          = copy_state(states[segment + 1u]),
                .density_weight  = original->density_weight,
                .velocity_weight = velocity_weight,
            });
            result.segments.push_back({
                .problem           = Problem{.begin_step = begin_step, .step_count = end_step - begin_step, .initial_state = copy_state(states[segment]), .keyframes = std::move(keyframes)},
                .active_parameters = control.active_parameters(begin_step, end_step),
            });
        }
        return result;
    }

    template <typename SolverType>
    ShootingSchedule LayeredMultipleShooting<SolverType>::alternate_schedule(const Problem& problem, const ShootingSchedule& initial, const ShootingPass& pass) const {
        std::vector<std::uint32_t> midpoints;
        std::vector<State> midpoint_states;
        midpoints.reserve(initial.segments.size());
        midpoint_states.reserve(initial.segments.size());
        for (std::size_t segment = 0u; segment < initial.segments.size(); ++segment) {
            const Problem& segment_problem = initial.segments[segment].problem;
            const std::uint32_t midpoint   = segment_problem.begin_step + segment_problem.step_count / 2u;
            midpoints.push_back(midpoint);
            const EvaluationTrace& trace = *pass.segments[segment].optimization.final_trace;
            midpoint_states.push_back(copy_state(trace.state[midpoint - segment_problem.begin_step]));
        }
        ShootingSchedule result{.kind = ShootingScheduleKind::alternate};
        result.segments.reserve(midpoints.size() - 1u);
        for (std::size_t segment = 0u; segment + 1u < midpoints.size(); ++segment) {
            const std::uint32_t begin_step = midpoints[segment];
            const std::uint32_t end_step   = midpoints[segment + 1u];
            std::vector<Keyframe> keyframes;
            for (const Keyframe& keyframe : problem.keyframes)
                if (keyframe.step > begin_step && keyframe.step < end_step)
                    keyframes.push_back({
                        .step            = keyframe.step,
                        .target          = copy_state(keyframe.target),
                        .density_weight  = keyframe.density_weight,
                        .velocity_weight = keyframe.velocity_weight,
                    });
            keyframes.push_back({
                .step            = end_step,
                .target          = copy_state(midpoint_states[segment + 1u]),
                .density_weight  = configuration.pseudo_density_weight,
                .velocity_weight = configuration.pseudo_velocity_weight,
            });
            result.segments.push_back({
                .problem           = Problem{.begin_step = begin_step, .step_count = end_step - begin_step, .initial_state = copy_state(midpoint_states[segment]), .keyframes = std::move(keyframes)},
                .active_parameters = control.active_parameters(begin_step, end_step),
            });
        }
        return result;
    }

    template <typename SolverType>
    ShootingPass LayeredMultipleShooting<SolverType>::process_schedule(ShootingSchedule& schedule, const ShootingProcessing processing, const std::uint32_t pass, const std::span<const double> parameter_values) {
        ShootingPass result{
            .pass       = pass,
            .schedule   = schedule.kind,
            .processing = processing,
            .parameters = {parameter_values.begin(), parameter_values.end()},
        };
        result.segments.reserve(schedule.segments.size());
        const std::vector<double> independent_start(result.parameters);
        std::vector<double> independent_delta(result.parameters.size());
        std::vector<std::uint32_t> independent_contributions(result.parameters.size());
        for (std::uint32_t segment_index = 0u; segment_index < schedule.segments.size(); ++segment_index) {
            ShootingSegment& segment = schedule.segments[segment_index];
            if (processing == ShootingProcessing::sequential && segment_index != 0u) {
                domain.grid.copy(result.segments.back().optimization.final_trace->state.back().density, segment.problem.initial_state.density);
                domain.grid.copy(result.segments.back().optimization.final_trace->state.back().velocity, segment.problem.initial_state.velocity);
            }
            Evaluator evaluator(domain, solver, control, objective_function, segment.problem);
            OptimizationRunner runner{.domain = domain, .evaluator = evaluator, .objective_function = objective_function, .control = control, .continuation = configuration.continuation};
            OptimizationCoordinates coordinates{
                .shooting_pass = pass,
                .schedule      = std::to_underlying(schedule.kind),
                .segment       = segment_index,
            };
            const std::span<const double> start = processing == ShootingProcessing::independent ? std::span<const double>{independent_start} : std::span<const double>{result.parameters};
            OptimizationResult optimization     = runner.run(start, segment.active_parameters, coordinates);
            if (processing == ShootingProcessing::independent)
                for (std::size_t parameter = 0u; parameter < independent_delta.size(); ++parameter)
                    if (segment.active_parameters[parameter] != 0u) {
                        independent_delta[parameter] += optimization.parameters[parameter] - independent_start[parameter];
                        ++independent_contributions[parameter];
                    }
            if (processing == ShootingProcessing::sequential) result.parameters = optimization.parameters;
            result.segments.push_back({
                .begin_step   = segment.problem.begin_step,
                .end_step     = segment.problem.begin_step + segment.problem.step_count,
                .optimization = std::move(optimization),
            });
        }
        if (processing == ShootingProcessing::independent)
            for (std::size_t parameter = 0u; parameter < result.parameters.size(); ++parameter)
                if (independent_contributions[parameter] != 0u) result.parameters[parameter] = independent_start[parameter] + independent_delta[parameter] / independent_contributions[parameter];
        return result;
    }

    template <typename SolverType>
    std::vector<State> LayeredMultipleShooting<SolverType>::propagate_boundary_states(const Problem& problem, const std::span<const std::uint32_t> boundaries, const std::vector<State>& current, const ShootingSchedule& alternate, const ShootingPass& pass) const {
        std::vector<State> result;
        result.reserve(current.size());
        for (const State& state : current) result.push_back(copy_state(state));
        for (std::size_t boundary = 1u; boundary + 1u < boundaries.size(); ++boundary) {
            const std::size_t segment      = boundary - 1u;
            const Problem& segment_problem = alternate.segments[segment].problem;
            const EvaluationTrace& trace   = *pass.segments[segment].optimization.final_trace;
            const auto original            = std::ranges::find_if(problem.keyframes, [&](const Keyframe& value) { return value.step == boundaries[boundary]; });
            if (original->velocity_weight == 0.0) domain.grid.copy(trace.state[boundaries[boundary] - segment_problem.begin_step].velocity, result[boundary].velocity);
        }
        return result;
    }

    template <typename SolverType>
    ShootingPassSummary LayeredMultipleShooting<SolverType>::summarize(const ShootingPass& pass) {
        ShootingPassSummary result{
            .pass       = pass.pass,
            .schedule   = pass.schedule,
            .processing = pass.processing,
            .parameters = pass.parameters,
        };
        result.segments.reserve(pass.segments.size());
        for (const SegmentOptimization& segment : pass.segments)
            result.segments.push_back({
                .begin_step         = segment.begin_step,
                .end_step           = segment.end_step,
                .parameters         = segment.optimization.parameters,
                .evaluations        = segment.optimization.evaluations,
                .level_stop_reasons = segment.optimization.level_stop_reasons,
                .final_summary      = segment.optimization.final_trace->summary,
            });
        return result;
    }
} // namespace physica::fluids::gas::keyframe_smoke
