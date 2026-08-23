export module physica.fluids.gas.keyframe_smoke.derivative_check;

import std;
import physica.fluids.gas.keyframe_smoke.evaluation;

export namespace physica::fluids::gas::keyframe_smoke {
    struct DirectionalDerivativeCheck final {
        double objective{};
        double finite_difference{};
        double jvp{};
        double vjp_dot_direction{};
        double finite_difference_jvp_relative_error{};
        double jvp_vjp_relative_error{};
    };

    struct ComponentDerivativeCheck final {
        std::size_t parameter{};
        double analytic{};
        double finite_difference{};
        double relative_error{};
    };

    struct DerivativeChecker final {
        explicit DerivativeChecker(Evaluator& evaluator);

        [[nodiscard]] DirectionalDerivativeCheck directional(std::span<const double> parameters, std::span<const double> direction, double epsilon);
        [[nodiscard]] std::vector<ComponentDerivativeCheck> components(std::span<const double> parameters, std::span<const std::size_t> parameter_indices, double epsilon);

    private:
        Evaluator& evaluator;
    };
} // namespace physica::fluids::gas::keyframe_smoke
