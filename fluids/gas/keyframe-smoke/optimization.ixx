export module physica.fluids.gas.keyframe_smoke.optimization;

import std;
import physica.fluids.gas.keyframe_smoke.control;
import physica.fluids.gas.keyframe_smoke.domain;
import physica.fluids.gas.keyframe_smoke.evaluation;
import physica.fluids.gas.keyframe_smoke.lbfgsb;
import physica.fluids.gas.keyframe_smoke.objective;

export namespace physica::fluids::gas::keyframe_smoke {
    struct ContinuationLevel final {
        float blur_sigma_cells{};
        LbfgsbConfiguration optimizer;
    };

    struct OptimizationRunner final {
        const Domain& domain;
        Evaluator& evaluator;
        Objective& objective;
        const ControlSystem& control;
        std::vector<ContinuationLevel> continuation;

        [[nodiscard]] OptimizationResult run(std::span<const double> initial_parameters, std::span<const std::uint8_t> active_parameters, OptimizationCoordinates coordinates = {});
    };
} // namespace physica::fluids::gas::keyframe_smoke
