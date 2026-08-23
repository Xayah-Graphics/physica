module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>
#include <cuda/stream>

export module physica.example.fluids.gas.adjoint_control;

import std;
import physica.fluids.gas.adjoint_control;

export namespace physica::examples::adjoint_control {
    struct ExperimentConfiguration final {
        std::uint32_t resolution{50u};
        std::uint32_t step_count{20u};
        std::uint32_t post_step_count{20u};
        std::array<std::uint32_t, 3> control_lattice{12u, 12u, 12u};
        std::uint32_t optimizer_iterations{200u};
        float time_step{1.0F / 25.0F};
    };

    struct ShapeMetrics final {
        double relative_l2{};
        double normalized_cross_correlation{};
        double soft_dice{};
        double mass_relative_error{};
        double maximum_divergence{};
        double rms_divergence{};
        double maximum_speed{};
        double dimensionless_maximum_divergence{};
    };

    struct Experiment final {
        const ExperimentConfiguration configuration;
        ::cuda::stream stream;
        fluids::gas::adjoint_control::Domain domain;
        fluids::gas::adjoint_control::Solver solver;
        fluids::gas::adjoint_control::ControlSystem control;
        fluids::gas::adjoint_control::Objective objective;
        const std::filesystem::path bunny_path;
        fluids::gas::adjoint_control::Problem problem;
        fluids::gas::adjoint_control::Evaluator evaluator;

        Experiment(ExperimentConfiguration configuration, std::filesystem::path bunny_path);
        ~Experiment();

        Experiment(const Experiment&) = delete;
        Experiment& operator=(const Experiment&) = delete;
        Experiment(Experiment&&) = delete;
        Experiment& operator=(Experiment&&) = delete;

        void verify(const std::filesystem::path& output_directory);
        void optimize(const std::filesystem::path& output_directory);
        [[nodiscard]] std::pair<std::size_t, double> measure_gradient();

    private:
        [[nodiscard]] fluids::gas::adjoint_control::DomainConfiguration create_domain_configuration() const;
        [[nodiscard]] fluids::gas::adjoint_control::ControlConfiguration create_control_configuration() const;
        [[nodiscard]] std::vector<fluids::gas::adjoint_control::Keyframe> create_keyframes();
        [[nodiscard]] fluids::gas::adjoint_control::State create_state(std::span<const float> density);
        [[nodiscard]] std::vector<float> create_initial_density() const;
        [[nodiscard]] std::vector<float> create_target_density() const;
        [[nodiscard]] std::vector<float> download_density(const fluids::gas::adjoint_control::State& state) const;
        [[nodiscard]] std::array<std::vector<float>, 3> download_velocity(const fluids::gas::adjoint_control::State& state) const;
        [[nodiscard]] ShapeMetrics shape_metrics(std::span<const float> density, std::span<const float> target, const fluids::gas::adjoint_control::State& state) const;
        void write_density(const std::filesystem::path& path, std::span<const float> density) const;
        void write_sequence(const std::filesystem::path& path, const std::vector<std::vector<float>>& frames) const;
        void write_parameters(const std::filesystem::path& path, std::span<const double> parameters) const;
    };

    void run_verification(std::string_view bunny_path, std::string_view output_directory);
    void run_bunny(std::string_view bunny_path, std::string_view output_directory);
    void benchmark(std::string_view bunny_path, std::string_view output_directory);
} // namespace physica::examples::adjoint_control
