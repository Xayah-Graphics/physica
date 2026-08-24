#ifndef PHYSICA_RECONSTRUCTION_INSTANT_NGP_SAMPLING_KERNELS_H
#define PHYSICA_RECONSTRUCTION_INSTANT_NGP_SAMPLING_KERNELS_H

#include <physica/cuda_stream.h>
#include <cstdint>

namespace physica::reconstruction::instant_ngp::kernels {
struct SamplingKernelShape final {
    std::uint32_t occupancy_grid_size;
    std::uint32_t ray_step_count;
    std::uint32_t training_batch_granularity;
    std::uint32_t evaluation_tile_rays;
    std::uint32_t density_grid_warmup_steps;
    std::uint32_t density_grid_skip_interval;
    std::uint32_t density_grid_max_skip;
    std::uint32_t density_grid_warmup_samples;
    std::uint32_t density_grid_uniform_samples;
    std::uint32_t density_grid_nonuniform_samples;
    bool snap_to_pixel_centers;
    std::uint32_t training_batch_size;
    std::uint32_t sample_capacity;

    constexpr bool operator==(const SamplingKernelShape&) const = default;
};

template <SamplingKernelShape Shape>
struct SamplingKernels final {
    static void set_occupancy_grid_full(::cuda::stream_ref stream, std::uint8_t* occupancy, std::uint32_t* occupancy_grid_occupied_count);
    static void sample_training_batch(::cuda::stream_ref stream, const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, std::uint32_t seed, std::uint32_t current_step, std::uint32_t rays_per_batch, std::uint32_t sample_limit, const std::uint8_t* occupancy, float* sample_coords, float* rays, std::uint32_t* target_pixel_indices, std::uint32_t* numsteps, std::uint32_t* ray_counter, std::uint32_t* sample_counter);
    static void update_occupancy_grid_from_density_grid(::cuda::stream_ref stream, const float* density_grid_values, float* density_grid_mean, std::uint32_t* occupancy_grid_occupied_count, std::uint8_t* occupancy);
    static std::uint32_t prepare_density_grid_update(::cuda::stream_ref stream, const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, std::uint32_t seed, std::uint32_t current_step, float* sample_coords, float* density_grid_values, float* density_grid_scratch, std::uint32_t* density_grid_indices, std::uint32_t density_grid_ema_step, bool reset_density_grid);
    static void accumulate_density_grid_update(::cuda::stream_ref stream, std::uint32_t sample_count, const std::uint32_t* density_grid_indices, const std::uint16_t* density_output, float* density_grid_scratch);
    static void commit_density_grid_update(::cuda::stream_ref stream, float* density_grid_scratch, float* density_grid_values, std::uint32_t& density_grid_ema_step);
    static std::uint32_t sample_evaluation_batch(::cuda::stream_ref stream, std::uint32_t tile_pixels, std::uint32_t pixel_offset, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, const float* camera, std::uint32_t image_index, const std::uint8_t* occupancy, float* sample_coords, std::uint32_t* numsteps, std::uint32_t* sample_counter);
};
} // namespace physica::reconstruction::instant_ngp::kernels

#endif
