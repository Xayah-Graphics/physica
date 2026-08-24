module;

#include <physica/cuda.h>

export module physica.example.fluids.gas.keyframe_smoke;

import std;
import physica.fluids.gas.keyframe_smoke;

export namespace physica::examples::keyframe_smoke {
    void compose(const std::filesystem::path& results_directory);

    struct ShapeMetrics final {
        double relative_l2{};
        double normalized_cross_correlation{};
        double soft_dice{};
        double mass_relative_error{};
    };

    struct Experiment final {
        inline static constexpr std::uint32_t resolution = 50u;
        inline static constexpr std::uint32_t step_count = 48u;
        inline static constexpr std::uint32_t control_window_steps = 6u;
        inline static constexpr float cell_size = 1.0F / resolution;
        inline static constexpr float time_step = 1.0F / 30.0F;

        ::cuda::stream stream;
        fluids::gas::keyframe_smoke::Domain domain;
        fluids::gas::keyframe_smoke::Solver solver;
        fluids::gas::keyframe_smoke::ControlSystem control;
        fluids::gas::keyframe_smoke::Objective objective;
        char letter;
        fluids::gas::keyframe_smoke::Problem problem;
        fluids::gas::keyframe_smoke::Evaluator evaluator;

        explicit Experiment(char letter);
        ~Experiment();

        Experiment(const Experiment&) = delete;
        Experiment& operator=(const Experiment&) = delete;
        Experiment(Experiment&&) = delete;
        Experiment& operator=(Experiment&&) = delete;

        void verify(const std::filesystem::path& output_directory);
        void optimize(const std::filesystem::path& output_directory);

    private:
        [[nodiscard]] static fluids::gas::keyframe_smoke::DomainConfiguration create_domain_configuration();
        [[nodiscard]] static fluids::gas::keyframe_smoke::ControlConfiguration create_control_configuration(char letter);
        [[nodiscard]] std::vector<fluids::gas::keyframe_smoke::Keyframe> create_keyframes();
        [[nodiscard]] fluids::gas::keyframe_smoke::State create_state(std::span<const float> density);
        [[nodiscard]] std::vector<float> create_initial_density() const;
        [[nodiscard]] std::vector<float> create_target_density() const;
        [[nodiscard]] std::vector<float> create_transport_density(float time) const;
        [[nodiscard]] std::vector<float> download_density(const fluids::gas::keyframe_smoke::State& state) const;
        [[nodiscard]] ShapeMetrics shape_metrics(std::span<const float> density, std::span<const float> target) const;
        void write_density(const std::filesystem::path& path, std::span<const float> density) const;
        void write_parameters(const std::filesystem::path& path, std::span<const double> parameters) const;
    };
} // namespace physica::examples::keyframe_smoke
