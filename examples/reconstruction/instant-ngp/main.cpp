#include <physica/cuda.h>

import std;
import physica.reconstruction.dataset.nerf_synthetic;
import physica.reconstruction.instant_ngp;

int main() {
    constexpr std::uint32_t training_steps      = 30'000u;
    constexpr std::uint32_t report_interval     = 100u;
    constexpr std::uint32_t evaluation_interval = 1'000u;
    constexpr std::uint32_t seed                = 1337u;

    const std::filesystem::path dataset_path = "data/nerf-synthetic/lego";
    const std::filesystem::path output_path  = "output/instant-ngp/lego";
    std::filesystem::create_directories(output_path);

    const physica::reconstruction::dataset::multiview::Dataset dataset = physica::reconstruction::dataset::nerf_synthetic::load(dataset_path, {.frame_sets = {"train", "val"}});
    physica::reconstruction::instant_ngp::InstantNGP<
        physica::reconstruction::instant_ngp::nerf_synthetic_network_shape,
        physica::reconstruction::instant_ngp::nerf_synthetic_sampling_shape,
        physica::reconstruction::instant_ngp::nerf_synthetic_rendering_shape
    > instant_ngp{dataset, 0u, 0u, 0.33F, seed};

    float best_psnr = std::numeric_limits<float>::lowest();
    while (instant_ngp.state.step < training_steps) {
        const std::uint32_t iterations = (std::min)(report_interval, training_steps - instant_ngp.state.step);
        const physica::reconstruction::instant_ngp::OptimizationStats training = instant_ngp.optimize(iterations);
        std::println("train step={:>6} loss={:.7f} rays={} samples={}/{} efficiency={:.2f}% occupancy={:.2f}% time={:.2f}ms", training.end_step, training.loss, training.rays, training.compacted_samples, training.generated_samples, training.sample_efficiency * 100.0F, training.occupancy_ratio * 100.0F, training.elapsed_ms);

        if (training.end_step % evaluation_interval != 0u || training.end_step == training_steps) continue;
        const physica::reconstruction::instant_ngp::EvaluationStats validation = instant_ngp.evaluate(1u);
        std::println("validation step={:>6} frames={} mse={:.8f} psnr={:.3f}dB time={:.2f}ms", validation.step, validation.frame_count, validation.mse, validation.psnr, validation.elapsed_ms);
        if (validation.psnr > best_psnr) {
            best_psnr = validation.psnr;
            instant_ngp.save(output_path / "best.safetensors");
        }
    }

    const physica::reconstruction::instant_ngp::EvaluationStats validation = instant_ngp.evaluate(1u);
    if (validation.psnr > best_psnr) instant_ngp.save(output_path / "best.safetensors");
    instant_ngp.save(output_path / "final.safetensors");
    std::println("complete step={} mse={:.8f} psnr={:.3f}dB", validation.step, validation.mse, validation.psnr);
    return 0;
}
