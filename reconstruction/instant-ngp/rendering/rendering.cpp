module;

#include "kernels.h"
#include <physica/cuda.h>

module physica.reconstruction.instant_ngp.rendering;

import std;
import physica.reconstruction.instant_ngp.network;
import physica.reconstruction.instant_ngp.sampling;
import physica.reconstruction.instant_ngp.scene;

namespace physica::reconstruction::instant_ngp {
template <NetworkShape NetworkSpec>
inline constexpr kernels::RenderingKernelShape rendering_kernel_shape{
    .training_batch_size = NetworkSpec.training_batch_size,
    .network_output_width = NetworkSpec.color.output_width,
};

template <RenderingShape Shape, NetworkShape NetworkSpec>
Rendering<Shape, NetworkSpec>::Rendering(const ::cuda::stream_ref source_stream)
    : stream{source_stream},
      compacted_samples{stream, ::cuda::device_default_memory_pool(stream.device()), NetworkSpec.training_batch_size, ::cuda::no_init},
      gradients{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(NetworkSpec.training_batch_size) * NetworkSpec.color.output_width, ::cuda::no_init},
      requested_compacted_sample_counter{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init},
      used_compacted_sample_counter{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init},
      training_loss_sum{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init},
      evaluation_loss_sum{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init} {}

template <RenderingShape Shape, NetworkShape NetworkSpec>
TrainingBatch Rendering<Shape, NetworkSpec>::loss_and_compact(const TrainingSamples samples, const NetworkOutput output, const DeviceFrameSet& frame_set, const std::uint32_t seed, const std::uint32_t step) {
    kernels::RenderingKernels<rendering_kernel_shape<NetworkSpec>>::compute_training_loss_and_compact_samples(stream, samples.rays_per_batch, seed, step, samples.ray_counter, frame_set.pixels.data(), output.data, requested_compacted_sample_counter.data(), used_compacted_sample_counter.data(), samples.target_pixel_indices, reinterpret_cast<const float*>(samples.rays), samples.numsteps, reinterpret_cast<const float*>(samples.samples.data), reinterpret_cast<float*>(compacted_samples.data()), gradients.data(), training_loss_sum.data());
    kernels::RenderingKernels<rendering_kernel_shape<NetworkSpec>>::pad_compacted_training_batch(stream, used_compacted_sample_counter.data(), reinterpret_cast<float*>(compacted_samples.data()), gradients.data());

    std::uint32_t ray_count = 0u;
    std::uint32_t generated_sample_count = 0u;
    std::uint32_t requested_compacted_sample_count = 0u;
    std::uint32_t compacted_sample_count = 0u;
    double loss_sum = 0.0;
    kernels::RenderingKernels<rendering_kernel_shape<NetworkSpec>>::read_training_statistics(stream, samples.ray_counter, samples.sample_counter, requested_compacted_sample_counter.data(), used_compacted_sample_counter.data(), training_loss_sum.data(), ray_count, generated_sample_count, requested_compacted_sample_count, compacted_sample_count, loss_sum);

    return {
        .samples = {.data = compacted_samples.data(), .count = NetworkSpec.training_batch_size},
        .gradients = {.data = gradients.data()},
        .ray_count = ray_count,
        .generated_sample_count = generated_sample_count,
        .requested_compacted_sample_count = requested_compacted_sample_count,
        .compacted_sample_count = compacted_sample_count,
        .loss_sum = loss_sum,
    };
}

template <RenderingShape Shape, NetworkShape NetworkSpec>
void Rendering<Shape, NetworkSpec>::begin_evaluation() {
    kernels::RenderingKernels<rendering_kernel_shape<NetworkSpec>>::begin_evaluation(stream, evaluation_loss_sum.data());
}

template <RenderingShape Shape, NetworkShape NetworkSpec>
void Rendering<Shape, NetworkSpec>::accumulate_evaluation(const EvaluationSamples samples, const NetworkOutput output, const DeviceFrameSet& frame_set) {
    kernels::RenderingKernels<rendering_kernel_shape<NetworkSpec>>::accumulate_evaluation_loss(stream, samples.pixel_count, samples.pixel_offset, samples.image_index, frame_set.extent.width, frame_set.extent.height, frame_set.pixels.data(), samples.numsteps, reinterpret_cast<const float*>(samples.samples.data), output.data, evaluation_loss_sum.data());
}

template <RenderingShape Shape, NetworkShape NetworkSpec>
double Rendering<Shape, NetworkSpec>::end_evaluation() {
    return kernels::RenderingKernels<rendering_kernel_shape<NetworkSpec>>::end_evaluation(stream, evaluation_loss_sum.data());
}

template struct Rendering<nerf_synthetic_rendering_shape, nerf_synthetic_network_shape>;
} // namespace physica::reconstruction::instant_ngp
