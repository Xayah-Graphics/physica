export module physica.optimization.lbfgsb;

import std;

export namespace physica::optimization {
    struct LbfgsbConfiguration final {
        std::uint32_t memory{12u};
        std::uint32_t maximum_iterations{200u};
        std::uint32_t maximum_evaluations{1000u};
        std::uint32_t maximum_line_search_evaluations{20u};
        double projected_gradient_tolerance{1.0e-6};
        double relative_objective_tolerance{1.0e-12};
        double armijo{1.0e-4};
        double minimum_step{1.0e-12};
    };

    enum class LbfgsbRequestKind : std::uint32_t {
        objective_gradient,
        complete,
    };

    enum class LbfgsbStopReason : std::uint32_t {
        running,
        projected_gradient,
        relative_objective,
        maximum_iterations,
        maximum_evaluations,
        line_search_failed,
        non_descent_direction,
    };

    struct LbfgsbRequest final {
        LbfgsbRequestKind kind{LbfgsbRequestKind::objective_gradient};
        std::uint32_t iteration{};
        std::uint32_t evaluation{};
        std::uint32_t line_search_evaluation{};
        double step_length{};
        std::span<const double> parameters;
    };

    struct LbfgsbIteration final {
        std::uint32_t iteration{};
        std::uint32_t evaluation{};
        double objective{};
        double projected_gradient_norm{};
        double step_length{};
        std::uint32_t line_search_evaluations{};
        std::uint32_t correction_count{};
        bool correction_accepted{};
    };

    struct Lbfgsb final {
        const LbfgsbConfiguration configuration;
        const std::vector<double> lower_bounds;
        const std::vector<double> upper_bounds;
        std::vector<double> parameters;
        std::vector<LbfgsbIteration> iterations;
        LbfgsbStopReason stop_reason{LbfgsbStopReason::running};

        Lbfgsb(LbfgsbConfiguration configuration, std::span<const double> initial_parameters, std::span<const double> lower_bounds, std::span<const double> upper_bounds);

        [[nodiscard]] LbfgsbRequest request() const;
        void submit(double objective, std::span<const double> gradient);

    private:
        enum class Phase : std::uint32_t {
            initial,
            line_search,
            complete,
        };

        struct Correction final {
            std::vector<double> step;
            std::vector<double> gradient_difference;
            double inverse_curvature{};
        };

        struct HessianRepresentation final {
            double scale{1.0};
            std::vector<std::vector<double>> hessian_steps;
            std::vector<double> hessian_step_curvatures;
            std::vector<double> gradient_curvatures;
        };

        Phase phase{Phase::initial};
        std::uint32_t iteration{};
        std::uint32_t evaluation{};
        std::uint32_t line_search_evaluation{};
        double objective{};
        double base_objective{};
        double step_length{};
        double maximum_step{};
        double base_directional_derivative{};
        std::vector<double> gradient;
        std::vector<double> projected_gradient;
        std::vector<double> direction;
        std::vector<double> base_parameters;
        std::vector<double> base_gradient;
        std::vector<double> trial_parameters;
        std::vector<Correction> corrections;

        [[nodiscard]] double projected_gradient_norm();
        [[nodiscard]] HessianRepresentation hessian_representation() const;
        void hessian_product(const HessianRepresentation& hessian, std::span<const double> input, std::span<double> output) const;
        void inverse_hessian_product(std::span<const double> input, std::span<double> output) const;
        void generalized_cauchy_point(const HessianRepresentation& hessian, std::span<double> cauchy) const;
        void subspace_minimization(const HessianRepresentation& hessian, std::span<const double> cauchy, std::span<double> candidate) const;
        void begin_line_search();
        void set_trial();
        void accept_trial(double next_objective, std::span<const double> next_gradient);
        void finish(LbfgsbStopReason reason);
    };
} // namespace physica::optimization
