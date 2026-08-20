module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>
#include <cuda/stream>

export module physica.reconstruction.instant_ngp.rendering;

import physica.reconstruction.instant_ngp.network;
import physica.reconstruction.instant_ngp.sampling;
import physica.reconstruction.instant_ngp.scene;
import std;

export namespace physica::reconstruction::instant_ngp {
enum class DensityActivation : std::uint8_t {
    exponential,
};

enum class ColorActivation : std::uint8_t {
    sigmoid,
};

enum class ReconstructionLoss : std::uint8_t {
    l2,
};

struct RenderingShape final {
    DensityActivation density_activation;
    ColorActivation color_activation;
    ReconstructionLoss loss;

    constexpr bool operator==(const RenderingShape&) const = default;
};

inline constexpr RenderingShape nerf_synthetic_rendering_shape{
    .density_activation = DensityActivation::exponential,
    .color_activation = ColorActivation::sigmoid,
    .loss = ReconstructionLoss::l2,
};

struct TrainingBatch final {
    DeviceSamples samples;
    NetworkGradients gradients;
    std::uint32_t ray_count = 0u;
    std::uint32_t generated_sample_count = 0u;
    std::uint32_t requested_compacted_sample_count = 0u;
    std::uint32_t compacted_sample_count = 0u;
    double loss_sum = 0.0;
};

template <RenderingShape Shape, NetworkShape NetworkSpec>
struct Rendering final {
    static_assert(Shape.density_activation == DensityActivation::exponential, "Only exponential density activation is implemented.");
    static_assert(Shape.color_activation == ColorActivation::sigmoid, "Only sigmoid color activation is implemented.");
    static_assert(Shape.loss == ReconstructionLoss::l2, "Only L2 reconstruction loss is implemented.");
    static_assert(NetworkSpec.training_batch_size == 1u << 18u, "Only a 2^18 training batch is implemented.");
    static_assert(NetworkSpec.color.output_width == 16u, "Only network output width 16 is implemented.");

    explicit Rendering(::cuda::stream_ref stream);

    TrainingBatch loss_and_compact(TrainingSamples samples, NetworkOutput output, const DeviceFrameSet& frame_set, std::uint32_t seed, std::uint32_t step);
    void begin_evaluation();
    void accumulate_evaluation(EvaluationSamples samples, NetworkOutput output, const DeviceFrameSet& frame_set);
    double end_evaluation();

private:
    ::cuda::stream_ref stream;
    ::cuda::device_buffer<Sample> compacted_samples;
    ::cuda::device_buffer<std::uint16_t> gradients;
    ::cuda::device_buffer<std::uint32_t> requested_compacted_sample_counter;
    ::cuda::device_buffer<std::uint32_t> used_compacted_sample_counter;
    ::cuda::device_buffer<double> training_loss_sum;
    ::cuda::device_buffer<double> evaluation_loss_sum;
};

extern template struct Rendering<nerf_synthetic_rendering_shape, nerf_synthetic_network_shape>;
} // namespace physica::reconstruction::instant_ngp
