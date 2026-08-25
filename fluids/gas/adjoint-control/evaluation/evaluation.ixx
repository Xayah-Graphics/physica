module;

export module physica.fluids.gas.adjoint_control.evaluation;

import std;
export import physica.fluids.gas.adjoint_control.control;
export import physica.fluids.gas.adjoint_control.objective;
export import physica.fluids.gas.adjoint_control.lbfgsb;

export namespace physica::fluids::gas::adjoint_control {
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
        double gradient_dot_direction{};
        double gradient_norm{};
        double projected_gradient_norm{};
    };

    struct ReverseTrace final {
        std::vector<double> parameter_gradient;
    };

    struct TangentTrace final {
        std::vector<StateTangent> state;
    };

    struct EvaluationTrace final {
        EvaluationTrace();
        ~EvaluationTrace();

        EvaluationTrace(EvaluationTrace&&) noexcept;
        EvaluationTrace& operator=(EvaluationTrace&&) noexcept;
        EvaluationTrace(const EvaluationTrace&) = delete;
        EvaluationTrace& operator=(const EvaluationTrace&) = delete;

        std::uint64_t evaluation{};
        std::vector<State> state;
        std::vector<StepRecord> steps;
        std::vector<KeyframeRecord> keyframes;
        std::optional<ReverseTrace> reverse;
        std::optional<TangentTrace> tangent;
        EvaluationSummary summary;
    };

    struct OptimizationCoordinates final {
        std::uint32_t optimizer_iteration{};
        std::uint32_t objective_evaluation{};
        std::uint32_t line_search_evaluation{};
        double line_search_step{};
    };

    struct OptimizationEvaluation final {
        OptimizationCoordinates coordinates;
        EvaluationSummary summary;
    };

    struct OptimizationResult final {
        OptimizationResult();
        ~OptimizationResult();

        OptimizationResult(OptimizationResult&&) noexcept;
        OptimizationResult& operator=(OptimizationResult&&) noexcept;
        OptimizationResult(const OptimizationResult&) = delete;
        OptimizationResult& operator=(const OptimizationResult&) = delete;

        std::vector<double> parameters;
        std::vector<OptimizationEvaluation> evaluations;
        LbfgsbStopReason stop_reason{LbfgsbStopReason::running};
        std::optional<EvaluationTrace> final_trace;
    };

    struct Evaluator final {
        Evaluator(const Domain& domain, Solver& solver, ControlSystem& control, Objective& objective, const Problem& problem);

        [[nodiscard]] EvaluationTrace evaluate(std::span<const double> parameters, EvaluationMode mode = EvaluationMode::objective_gradient, std::span<const double> direction = {});

    private:
        const Domain& domain;
        Solver& solver;
        ControlSystem& control;
        Objective& objective;
        const Problem& problem;
        std::uint64_t evaluation_count{};
    };
} // namespace physica::fluids::gas::adjoint_control
