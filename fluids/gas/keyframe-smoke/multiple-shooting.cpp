module;

#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>

module physica.fluids.gas.keyframe_smoke.multiple_shooting;

import std;

namespace physica::fluids::gas::keyframe_smoke {
    LayeredMultipleShootingResult LayeredMultipleShooting::run(const Problem& problem, const std::span<const double> initial_parameters) {
        LayeredMultipleShootingResult result{.parameters = {initial_parameters.begin(), initial_parameters.end()}};
        const std::vector<std::uint32_t> boundaries = boundary_steps(problem);
        std::vector<State> states = initial_boundary_states(problem, boundaries);
        bool propagated = false;
        const std::uint32_t schedule_sets = configuration.parallel_schedule_sets + configuration.sequential_schedule_sets;
        result.passes.reserve(schedule_sets * 2u);
        for (std::uint32_t pass = 0u; pass < schedule_sets; ++pass) {
            const ShootingProcessing processing = pass < configuration.parallel_schedule_sets ? ShootingProcessing::parallel : ShootingProcessing::sequential;
            ShootingSchedule initial = initial_schedule(problem, boundaries, states, propagated);
            ShootingPass initial_pass = process_schedule(initial, processing, pass, result.parameters);
            ShootingSchedule alternate = alternate_schedule(problem, initial, initial_pass);
            ShootingPass alternate_pass = process_schedule(alternate, processing, pass, initial_pass.parameters);
            states = propagate_boundary_states(problem, boundaries, states, alternate, alternate_pass);
            result.parameters = alternate_pass.parameters;
            result.passes.push_back(std::move(initial_pass));
            result.passes.push_back(std::move(alternate_pass));
            propagated = true;
        }
        Evaluator evaluator(domain, solver, control, objective, problem);
        result.final_trace.emplace(evaluator.evaluate(result.parameters, EvaluationMode::objective_gradient));
        return result;
    }

    State LayeredMultipleShooting::copy_state(const State& source) const {
        State result = solver.allocate_state(domain);
        solver.copy(domain, source, result);
        return result;
    }

    Keyframe LayeredMultipleShooting::copy_keyframe(const Keyframe& source) const {
        return {
            .step = source.step,
            .target = copy_state(source.target),
            .density_weight = source.density_weight,
            .velocity_weight = source.velocity_weight,
            .pseudo = source.pseudo,
        };
    }

    std::vector<std::uint32_t> LayeredMultipleShooting::boundary_steps(const Problem& problem) const {
        std::vector<std::uint32_t> result{problem.begin_step};
        for (const Keyframe& keyframe : problem.keyframes) result.push_back(keyframe.step);
        result.push_back(problem.begin_step + problem.step_count);
        std::ranges::sort(result);
        const auto unique_end = std::ranges::unique(result).begin();
        result.erase(unique_end, result.end());
        return result;
    }

    std::vector<State> LayeredMultipleShooting::initial_boundary_states(const Problem& problem, const std::span<const std::uint32_t> boundaries) const {
        std::vector<State> result;
        result.reserve(boundaries.size());
        result.push_back(copy_state(problem.initial_state));
        for (std::size_t boundary = 1u; boundary < boundaries.size(); ++boundary) {
            const auto keyframe = std::ranges::find_if(problem.keyframes, [&](const Keyframe& value) { return value.step == boundaries[boundary]; });
            result.push_back(copy_state(keyframe->target));
        }
        return result;
    }

    ShootingSchedule LayeredMultipleShooting::initial_schedule(const Problem& problem, const std::span<const std::uint32_t> boundaries, const std::vector<State>& states, const bool propagated) const {
        ShootingSchedule result{.kind = ShootingScheduleKind::initial};
        result.segments.reserve(boundaries.size() - 1u);
        for (std::size_t segment = 0u; segment + 1u < boundaries.size(); ++segment) {
            const std::uint32_t begin_step = boundaries[segment];
            const std::uint32_t end_step = boundaries[segment + 1u];
            const auto original = std::ranges::find_if(problem.keyframes, [&](const Keyframe& value) { return value.step == end_step; });
            const double velocity_weight = original->velocity_weight != 0.0 ? original->velocity_weight : propagated && segment + 1u < boundaries.size() - 1u ? configuration.propagated_velocity_weight : 0.0;
            std::vector<Keyframe> keyframes;
            keyframes.push_back({
                .step = end_step,
                .target = copy_state(states[segment + 1u]),
                .density_weight = original->density_weight,
                .velocity_weight = velocity_weight,
                .pseudo = false,
            });
            result.segments.push_back({
                .problem = Problem{begin_step, end_step - begin_step, copy_state(states[segment]), std::move(keyframes)},
                .active_parameters = control.active_parameters(begin_step, end_step),
            });
        }
        return result;
    }

    ShootingSchedule LayeredMultipleShooting::alternate_schedule(const Problem& problem, const ShootingSchedule& initial, const ShootingPass& pass) const {
        std::vector<std::uint32_t> midpoints;
        std::vector<State> midpoint_states;
        midpoints.reserve(initial.segments.size());
        midpoint_states.reserve(initial.segments.size());
        for (std::size_t segment = 0u; segment < initial.segments.size(); ++segment) {
            const Problem& segment_problem = initial.segments[segment].problem;
            const std::uint32_t midpoint = segment_problem.begin_step + segment_problem.step_count / 2u;
            midpoints.push_back(midpoint);
            const EvaluationTrace& trace = *pass.segments[segment].optimization.final_trace;
            midpoint_states.push_back(copy_state(trace.state[midpoint - segment_problem.begin_step]));
        }
        ShootingSchedule result{.kind = ShootingScheduleKind::alternate};
        result.segments.reserve(midpoints.size() - 1u);
        for (std::size_t segment = 0u; segment + 1u < midpoints.size(); ++segment) {
            const std::uint32_t begin_step = midpoints[segment];
            const std::uint32_t end_step = midpoints[segment + 1u];
            std::vector<Keyframe> keyframes;
            for (const Keyframe& keyframe : problem.keyframes) if (keyframe.step > begin_step && keyframe.step < end_step) keyframes.push_back(copy_keyframe(keyframe));
            keyframes.push_back({
                .step = end_step,
                .target = copy_state(midpoint_states[segment + 1u]),
                .density_weight = configuration.pseudo_density_weight,
                .velocity_weight = configuration.pseudo_velocity_weight,
                .pseudo = true,
            });
            result.segments.push_back({
                .problem = Problem{begin_step, end_step - begin_step, copy_state(midpoint_states[segment]), std::move(keyframes)},
                .active_parameters = control.active_parameters(begin_step, end_step),
            });
        }
        return result;
    }

    ShootingPass LayeredMultipleShooting::process_schedule(ShootingSchedule& schedule, const ShootingProcessing processing, const std::uint32_t pass, const std::span<const double> parameter_values) {
        ShootingPass result{
            .pass = pass,
            .schedule = schedule.kind,
            .processing = processing,
            .parameters = {parameter_values.begin(), parameter_values.end()},
        };
        result.segments.reserve(schedule.segments.size());
        const std::vector<double> parallel_start(result.parameters);
        std::vector<double> parallel_merged(result.parameters);
        for (std::uint32_t segment_index = 0u; segment_index < schedule.segments.size(); ++segment_index) {
            ShootingSegment& segment = schedule.segments[segment_index];
            if (processing == ShootingProcessing::sequential && segment_index != 0u) solver.copy(domain, result.segments.back().optimization.final_trace->state.back(), segment.problem.initial_state);
            Evaluator evaluator(domain, solver, control, objective, segment.problem);
            OptimizationRunner runner{.domain = domain, .evaluator = evaluator, .objective = objective, .control = control, .continuation = configuration.continuation};
            OptimizationCoordinates coordinates{
                .shooting_pass = pass,
                .schedule = static_cast<std::uint32_t>(schedule.kind),
                .segment = segment_index,
            };
            const std::span<const double> start = processing == ShootingProcessing::parallel ? std::span<const double>{parallel_start} : std::span<const double>{result.parameters};
            OptimizationResult optimization = runner.run(start, segment.active_parameters, coordinates);
            if (processing == ShootingProcessing::parallel) for (std::size_t parameter = 0u; parameter < parallel_merged.size(); ++parameter) if (segment.active_parameters[parameter] != 0u) parallel_merged[parameter] = optimization.parameters[parameter];
            if (processing == ShootingProcessing::sequential) result.parameters = optimization.parameters;
            result.segments.push_back({
                .begin_step = segment.problem.begin_step,
                .end_step = segment.problem.begin_step + segment.problem.step_count,
                .optimization = std::move(optimization),
            });
        }
        if (processing == ShootingProcessing::parallel) result.parameters = std::move(parallel_merged);
        return result;
    }

    std::vector<State> LayeredMultipleShooting::propagate_boundary_states(const Problem& problem, const std::span<const std::uint32_t> boundaries, const std::vector<State>& current, const ShootingSchedule& alternate, const ShootingPass& pass) const {
        std::vector<State> result;
        result.reserve(current.size());
        for (const State& state : current) result.push_back(copy_state(state));
        for (std::size_t boundary = 1u; boundary + 1u < boundaries.size(); ++boundary) {
            const std::size_t segment = boundary - 1u;
            const Problem& segment_problem = alternate.segments[segment].problem;
            const EvaluationTrace& trace = *pass.segments[segment].optimization.final_trace;
            const auto original = std::ranges::find_if(problem.keyframes, [&](const Keyframe& value) { return value.step == boundaries[boundary]; });
            if (original->velocity_weight == 0.0) domain.copy(trace.state[boundaries[boundary] - segment_problem.begin_step].velocity, result[boundary].velocity);
        }
        return result;
    }

} // namespace physica::fluids::gas::keyframe_smoke
