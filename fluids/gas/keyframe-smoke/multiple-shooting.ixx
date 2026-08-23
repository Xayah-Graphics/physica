export module physica.fluids.gas.keyframe_smoke.multiple_shooting;

import std;
import physica.fluids.gas.keyframe_smoke.control;
import physica.fluids.gas.keyframe_smoke.domain;
import physica.fluids.gas.keyframe_smoke.evaluation;
import physica.fluids.gas.keyframe_smoke.objective;
import physica.fluids.gas.keyframe_smoke.optimization;
import physica.fluids.gas.keyframe_smoke.solver;

export namespace physica::fluids::gas::keyframe_smoke {
    enum class ShootingScheduleKind : std::uint32_t {
        initial,
        alternate,
    };

    enum class ShootingProcessing : std::uint32_t {
        parallel,
        sequential,
    };

    struct LayeredMultipleShootingConfiguration final {
        std::uint32_t parallel_schedule_sets{2u};
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

    struct LayeredMultipleShootingResult final {
        std::vector<double> parameters;
        std::vector<ShootingPass> passes;
        std::optional<EvaluationTrace> final_trace;
    };

    struct LayeredMultipleShooting final {
        const Domain& domain;
        Solver& solver;
        ControlSystem& control;
        Objective& objective;
        const LayeredMultipleShootingConfiguration configuration;

        [[nodiscard]] LayeredMultipleShootingResult run(const Problem& problem, std::span<const double> initial_parameters);

    private:
        [[nodiscard]] State copy_state(const State& source) const;
        [[nodiscard]] Keyframe copy_keyframe(const Keyframe& source) const;
        [[nodiscard]] std::vector<std::uint32_t> boundary_steps(const Problem& problem) const;
        [[nodiscard]] std::vector<State> initial_boundary_states(const Problem& problem, std::span<const std::uint32_t> boundaries) const;
        [[nodiscard]] ShootingSchedule initial_schedule(const Problem& problem, std::span<const std::uint32_t> boundaries, const std::vector<State>& states, bool propagated) const;
        [[nodiscard]] ShootingSchedule alternate_schedule(const Problem& problem, const ShootingSchedule& initial, const ShootingPass& pass) const;
        [[nodiscard]] ShootingPass process_schedule(ShootingSchedule& schedule, ShootingProcessing processing, std::uint32_t pass, std::span<const double> parameters);
        [[nodiscard]] std::vector<State> propagate_boundary_states(const Problem& problem, std::span<const std::uint32_t> boundaries, const std::vector<State>& current, const ShootingSchedule& alternate, const ShootingPass& pass) const;
    };
} // namespace physica::fluids::gas::keyframe_smoke
