#ifndef PHYSICA_RECONSTRUCTION_INSTANT_NGP_RENDERING_KERNELS_H
#define PHYSICA_RECONSTRUCTION_INSTANT_NGP_RENDERING_KERNELS_H

#include <physica/cuda_stream.h>
#include <cstdint>

namespace physica::reconstruction::instant_ngp::kernels {
struct RenderingKernelShape final {
    std::uint32_t training_batch_size;
    std::uint32_t network_output_width;

    constexpr bool operator==(const RenderingKernelShape&) const = default;
};

template <RenderingKernelShape Shape>
struct RenderingKernels final {
    static void compute_training_loss_and_compact_samples(::cuda::stream_ref stream, std::uint32_t rays_per_batch, std::uint32_t seed, std::uint32_t current_step, const std::uint32_t* ray_counter, const std::uint8_t* pixels, const std::uint16_t* network_output, std::uint32_t* requested_compacted_sample_counter, std::uint32_t* used_compacted_sample_counter, const std::uint32_t* target_pixel_indices, const float* rays, std::uint32_t* numsteps, const float* sample_coords, float* compacted_sample_coords, std::uint16_t* network_output_gradients, double* loss_sum);
    static void pad_compacted_training_batch(::cuda::stream_ref stream, const std::uint32_t* compacted_sample_counter, float* compacted_sample_coords, std::uint16_t* network_output_gradients);
    static void read_training_statistics(::cuda::stream_ref stream, const std::uint32_t* ray_counter, const std::uint32_t* generated_sample_counter, const std::uint32_t* requested_compacted_sample_counter, const std::uint32_t* used_compacted_sample_counter, const double* loss_sum, std::uint32_t& ray_count, std::uint32_t& generated_sample_count, std::uint32_t& requested_compacted_sample_count, std::uint32_t& used_compacted_sample_count, double& output_loss_sum);
    static void begin_evaluation(::cuda::stream_ref stream, double* loss_sum);
    static void accumulate_evaluation_loss(::cuda::stream_ref stream, std::uint32_t tile_pixels, std::uint32_t pixel_offset, std::uint32_t image_index, std::uint32_t width, std::uint32_t height, const std::uint8_t* pixels, const std::uint32_t* numsteps, const float* sample_coords, const std::uint16_t* network_output, double* loss_sum);
    static double end_evaluation(::cuda::stream_ref stream, const double* loss_sum);
};
} // namespace physica::reconstruction::instant_ngp::kernels

#endif
