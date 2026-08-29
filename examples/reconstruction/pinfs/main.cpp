#include <physica/cuda.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

import std;
import physica.reconstruction.dataset.pinf;
import physica.reconstruction.pinfs;

int main(const int argc, char** argv) {
    constexpr std::uint32_t report_interval  = 1'000u;
    constexpr std::uint32_t evaluation_stride = 20u;
    constexpr std::uint32_t seed             = 42u;

    struct Experiment final {
        std::string_view name;
        std::filesystem::path dataset_path;
        std::filesystem::path output_path;
        physica::reconstruction::dataset::pinf::Resolution resolution;
        physica::reconstruction::pinfs::Configuration configuration;
        std::uint32_t training_steps;
        bool evaluate;
    };

    const std::array experiments{
        Experiment{"sphere-neus", "data/pinf/Sphere", "output/pinfs/sphere-neus", physica::reconstruction::dataset::pinf::Resolution::full, physica::reconstruction::pinfs::sphere_neus_configuration, 200'000u, true},
        Experiment{"game-neus", "data/pinf/Game", "output/pinfs/game-neus", physica::reconstruction::dataset::pinf::Resolution::half, physica::reconstruction::pinfs::game_neus_configuration, 200'000u, true},
        Experiment{"scalar-real", "data/pinf/ScalarReal", "output/pinfs/scalar-real", physica::reconstruction::dataset::pinf::Resolution::half, physica::reconstruction::pinfs::scalar_real_configuration, 600'000u, false},
    };
    const std::string_view experiment_name = argc > 1 ? argv[1] : "sphere-neus";
    const Experiment& experiment = *std::ranges::find_if(experiments, [&](const Experiment& candidate) { return candidate.name == experiment_name; });
    const std::uint32_t training_steps = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : experiment.training_steps;
    const std::filesystem::path output_path = argc > 4 ? argv[4] : experiment.output_path;
    std::filesystem::create_directories(output_path);

    const std::vector<std::string> frame_sets = experiment.evaluate ? std::vector<std::string>{"train", "test"} : std::vector<std::string>{"train"};
    const physica::reconstruction::dataset::pinf::Dataset dataset = physica::reconstruction::dataset::pinf::load(experiment.dataset_path, {.frame_sets = frame_sets, .resolution = experiment.resolution});
    physica::reconstruction::pinfs::PINFS pinfs{dataset, experiment.configuration, "data/pinf/vgg19-features.bin", 0u, seed};
    if (argc > 3) pinfs.load(argv[3]);

    while (pinfs.state.step < training_steps) {
        const std::uint32_t iterations = std::min(report_interval, training_steps - pinfs.state.step);
        const physica::reconstruction::pinfs::OptimizationStats training = pinfs.optimize(iterations);
        std::println("train step={:>6} loss={:.7f} image={:.7f} coarse={:.7f} vgg={:.7f} ghost={:.7f} overlay={:.7f} eikonal={:.7f} physics={:.7f} neumann={:.7f} psnr={:.3f}dB time={:.2f}ms", training.end_step, training.loss, training.image_loss, training.coarse_image_loss, training.perceptual_loss, training.ghost_loss, training.overlay_loss, training.eikonal_loss, training.physics_loss, training.neumann_loss, training.psnr, training.elapsed_ms);
        std::cout.flush();
    }

    pinfs.save(output_path / "final.safetensors");
    if (!experiment.evaluate) return 0;

    const physica::reconstruction::pinfs::EvaluationStats evaluation = pinfs.evaluate("test", 2u, evaluation_stride);
    std::println("complete step={} frames={} mse={:.8f} psnr={:.3f}dB time={:.2f}ms", evaluation.step, evaluation.frame_count, evaluation.mse, evaluation.psnr, evaluation.elapsed_ms);
    const physica::reconstruction::dataset::multiview::FrameSet& test = *std::ranges::find_if(dataset.multiview.frame_sets, [](const physica::reconstruction::dataset::multiview::FrameSet& frame_set) { return frame_set.name == "test"; });
    for (const std::uint32_t frame_index : std::views::iota(0u, evaluation.frame_count)) {
        const physica::reconstruction::dataset::multiview::Frame& frame = test.frames[frame_index * evaluation_stride];
        const physica::reconstruction::pinfs::RenderedFrame rendered    = pinfs.render(frame);
        std::vector<std::uint8_t> rgb(rendered.rgb.size() * 3uz);
        for (const std::size_t pixel : std::views::iota(0uz, rendered.rgb.size()))
            for (const std::size_t component : std::views::iota(0uz, 3uz)) rgb[pixel * 3uz + component] = static_cast<std::uint8_t>(255.0F * std::clamp(rendered.rgb[pixel][component], 0.0F, 1.0F));
        const std::filesystem::path frame_path = output_path / std::format("test-{:03}.png", frame_index * evaluation_stride);
        stbi_write_png(frame_path.string().c_str(), static_cast<int>(rendered.width), static_cast<int>(rendered.height), 3, rgb.data(), static_cast<int>(rendered.width * 3u));
    }
    return 0;
}
