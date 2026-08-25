module;

#include <physica/cuda.h>

export module physica.optimization.lbfgsb;

import std;

namespace physica::optimization {
    struct LbfgsbBackendAccess;
}

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
    };

    struct LbfgsbRequest final {
        LbfgsbRequestKind kind{LbfgsbRequestKind::objective_gradient};
        std::uint32_t iteration{};
        std::uint32_t evaluation{};
        std::uint32_t line_search_evaluation{};
        double step_length{};
        ::cuda::std::span<const double> parameters;
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

    struct LbfgsbGradientMetrics final {
        double gradient_norm{};
        double projected_gradient_norm{};
    };

    struct Lbfgsb final {
        friend struct LbfgsbBackendAccess;

        const LbfgsbConfiguration configuration;
        const std::size_t parameter_count;
        ::cuda::device_buffer<double> parameters;
        std::vector<LbfgsbIteration> iterations;
        LbfgsbStopReason stop_reason{LbfgsbStopReason::running};
        double gradient_norm{};
        double projected_gradient_norm{};

        Lbfgsb(::cuda::stream_ref stream, LbfgsbConfiguration configuration, std::span<const double> initial_parameters, std::span<const double> lower_bounds, std::span<const double> upper_bounds);

        [[nodiscard]] LbfgsbRequest request() const;
        [[nodiscard]] LbfgsbGradientMetrics submit(const double* objective, ::cuda::std::span<const double> gradient);

    private:
        enum class Phase : std::uint32_t {
            initial,
            line_search,
            complete,
        };

        const ::cuda::stream_ref stream;
        Phase phase{Phase::initial};
        std::uint32_t iteration{};
        std::uint32_t evaluation{};
        std::uint32_t line_search_evaluation{};
        std::uint32_t correction_count{};
        std::uint32_t correction_head{};
        double objective{};
        double base_objective{};
        double step_length{};
        double base_directional_derivative{};
        ::cuda::device_buffer<double> lower_bounds;
        ::cuda::device_buffer<double> upper_bounds;
        ::cuda::device_buffer<double> gradient;
        ::cuda::device_buffer<double> trial_parameters;
        ::cuda::device_buffer<double> correction_steps;
        ::cuda::device_buffer<double> correction_gradient_differences;
        ::cuda::device_buffer<double> inverse_curvatures;
        ::cuda::device_buffer<double> hessian_steps;
        ::cuda::device_buffer<double> hessian_step_curvatures;
        ::cuda::device_buffer<double> gradient_curvatures;
        ::cuda::device_buffer<double> direction;
        ::cuda::device_buffer<double> path;
        ::cuda::device_buffer<double> free_direction;
        ::cuda::device_buffer<double> cauchy;
        ::cuda::device_buffer<double> displacement;
        ::cuda::device_buffer<double> hessian_displacement;
        ::cuda::device_buffer<double> model_gradient;
        ::cuda::device_buffer<double> subspace_direction;
        ::cuda::device_buffer<double> residual;
        ::cuda::device_buffer<double> conjugate;
        ::cuda::device_buffer<double> hessian_conjugate;
        ::cuda::device_buffer<double> breakpoints;
        ::cuda::device_buffer<double> sorted_breakpoints;
        ::cuda::device_buffer<std::uint32_t> breakpoint_indices;
        ::cuda::device_buffer<std::uint32_t> sorted_breakpoint_indices;
        ::cuda::device_buffer<std::uint8_t> free_mask;
        ::cuda::device_buffer<std::byte> sort_scratch;
        ::cuda::device_buffer<std::byte> status;
        void begin_line_search(double projected_gradient_norm);
        void set_trial();
        void accept_trial(double next_objective, ::cuda::std::span<const double> next_gradient, LbfgsbGradientMetrics metrics);
        void finish(LbfgsbStopReason reason);
    };
} // namespace physica::optimization
