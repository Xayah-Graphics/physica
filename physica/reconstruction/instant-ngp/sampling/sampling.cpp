module;

#include "kernels.h"
#include <physica/cuda.h>
#include <cuda/cmath>

module physica.reconstruction.instant_ngp.sampling;

import std;
import physica.reconstruction.instant_ngp.network;
import physica.reconstruction.instant_ngp.scene;

namespace physica::reconstruction::instant_ngp {
    template <SamplingShape Shape, NetworkShape NetworkSpec>
    inline constexpr kernels::SamplingKernelShape sampling_kernel_shape{
        .occupancy_grid_size             = Shape.occupancy_grid_size,
        .ray_step_count                  = Shape.ray_step_count,
        .training_batch_granularity      = Shape.training_batch_granularity,
        .evaluation_tile_rays            = Shape.evaluation_tile_rays,
        .density_grid_warmup_steps       = Shape.density_grid_warmup_steps,
        .density_grid_skip_interval      = Shape.density_grid_skip_interval,
        .density_grid_max_skip           = Shape.density_grid_max_skip,
        .density_grid_warmup_samples     = Shape.density_grid_warmup_samples,
        .density_grid_uniform_samples    = Shape.density_grid_uniform_samples,
        .density_grid_nonuniform_samples = Shape.density_grid_nonuniform_samples,
        .snap_to_pixel_centers           = Shape.snap_to_pixel_centers,
        .training_batch_size             = NetworkSpec.training_batch_size,
        .sample_capacity                 = NetworkSpec.inference_capacity,
    };

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    Sampling<Shape, NetworkSpec>::Sampling(const ::cuda::stream_ref source_stream)
        : stream{source_stream}, samples{stream, ::cuda::device_default_memory_pool(stream.device()), NetworkSpec.inference_capacity, ::cuda::no_init}, rays{stream, ::cuda::device_default_memory_pool(stream.device()), NetworkSpec.training_batch_size, ::cuda::no_init}, target_pixel_indices{stream, ::cuda::device_default_memory_pool(stream.device()), NetworkSpec.training_batch_size, ::cuda::no_init}, numsteps{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(NetworkSpec.training_batch_size) * 2uz, ::cuda::no_init}, ray_counter{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, sample_counter{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, occupancy{stream, ::cuda::device_default_memory_pool(stream.device()), grid_cell_count / 8u, ::cuda::no_init}, density{stream, ::cuda::device_default_memory_pool(stream.device()), grid_cell_count, ::cuda::no_init},
          density_scratch{stream, ::cuda::device_default_memory_pool(stream.device()), grid_cell_count, ::cuda::no_init}, density_indices{stream, ::cuda::device_default_memory_pool(stream.device()), grid_cell_count, ::cuda::no_init}, density_mean{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, occupied_cell_counter{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, evaluation_numsteps{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(Shape.evaluation_tile_rays) * 2uz, ::cuda::no_init}, evaluation_sample_counter{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, occupied_cell_count{stream, ::cuda::pinned_default_memory_pool(), 1uz, ::cuda::no_init} {
        ::cuda::fill_bytes(stream, density, 0u);
        ::cuda::fill_bytes(stream, density_scratch, 0u);
        ::cuda::fill_bytes(stream, density_mean, 0u);
        kernels::SamplingKernels<sampling_kernel_shape<Shape, NetworkSpec>>::set_occupancy_grid_full(stream, occupancy.data(), occupied_cell_counter.data());
        ::cuda::copy_bytes(stream, ::cuda::std::span{occupied_cell_counter.data(), 1uz}, occupied_cell_count);
        stream.sync();
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    DeviceSamples Sampling<Shape, NetworkSpec>::prepare_density_update(const DeviceFrameSet& frame_set, const std::uint32_t seed, const std::uint32_t step) {
        const std::uint32_t sample_count = kernels::SamplingKernels<sampling_kernel_shape<Shape, NetworkSpec>>::prepare_density_grid_update(stream, reinterpret_cast<const float*>(frame_set.cameras.data()), frame_set.frame_count, frame_set.extent.width, frame_set.extent.height, frame_set.intrinsics.focal_x, frame_set.intrinsics.focal_y, frame_set.intrinsics.principal_x, frame_set.intrinsics.principal_y, seed, step, reinterpret_cast<float*>(samples.data()), density.data(), density_scratch.data(), density_indices.data(), density_grid_ema_step, density_grid_ema_step == 0u);
        return {.data = samples.data(), .count = sample_count};
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    void Sampling<Shape, NetworkSpec>::accumulate_density_update(const std::uint32_t offset, const NetworkOutput output) {
        kernels::SamplingKernels<sampling_kernel_shape<Shape, NetworkSpec>>::accumulate_density_grid_update(stream, output.count, density_indices.data() + offset, output.data, density_scratch.data());
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    void Sampling<Shape, NetworkSpec>::commit_density_update() {
        kernels::SamplingKernels<sampling_kernel_shape<Shape, NetworkSpec>>::commit_density_grid_update(stream, density_scratch.data(), density.data(), density_grid_ema_step);
        kernels::SamplingKernels<sampling_kernel_shape<Shape, NetworkSpec>>::update_occupancy_grid_from_density_grid(stream, density.data(), density_mean.data(), occupied_cell_counter.data(), occupancy.data());
        ::cuda::copy_bytes(stream, ::cuda::std::span{occupied_cell_counter.data(), 1uz}, occupied_cell_count);
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    TrainingSamples Sampling<Shape, NetworkSpec>::sample_training(const DeviceFrameSet& frame_set, const std::uint32_t seed, const std::uint32_t step) {
        kernels::SamplingKernels<sampling_kernel_shape<Shape, NetworkSpec>>::sample_training_batch(stream, reinterpret_cast<const float*>(frame_set.cameras.data()), frame_set.frame_count, frame_set.extent.width, frame_set.extent.height, frame_set.intrinsics.focal_x, frame_set.intrinsics.focal_y, frame_set.intrinsics.principal_x, frame_set.intrinsics.principal_y, seed, step, rays_per_batch, inference_sample_count, occupancy.data(), reinterpret_cast<float*>(samples.data()), reinterpret_cast<float*>(rays.data()), target_pixel_indices.data(), numsteps.data(), ray_counter.data(), sample_counter.data());
        return {
            .samples              = {.data = samples.data(), .count = inference_sample_count},
            .rays                 = rays.data(),
            .target_pixel_indices = target_pixel_indices.data(),
            .numsteps             = numsteps.data(),
            .ray_counter          = ray_counter.data(),
            .sample_counter       = sample_counter.data(),
            .rays_per_batch       = rays_per_batch,
        };
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    EvaluationSamples Sampling<Shape, NetworkSpec>::sample_evaluation(const DeviceFrameSet& frame_set, const std::uint32_t image_index, const std::uint32_t pixel_offset, const std::uint32_t pixel_count) {
        const std::uint32_t sample_count = kernels::SamplingKernels<sampling_kernel_shape<Shape, NetworkSpec>>::sample_evaluation_batch(stream, pixel_count, pixel_offset, frame_set.extent.width, frame_set.extent.height, frame_set.intrinsics.focal_x, frame_set.intrinsics.focal_y, frame_set.intrinsics.principal_x, frame_set.intrinsics.principal_y, reinterpret_cast<const float*>(frame_set.cameras.data()), image_index, occupancy.data(), reinterpret_cast<float*>(samples.data()), evaluation_numsteps.data(), evaluation_sample_counter.data());
        return {
            .samples      = {.data = samples.data(), .count = sample_count},
            .numsteps     = evaluation_numsteps.data(),
            .image_index  = image_index,
            .pixel_offset = pixel_offset,
            .pixel_count  = pixel_count,
        };
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    std::uint32_t Sampling<Shape, NetworkSpec>::adapt(const std::uint32_t generated_sample_count, const std::uint32_t requested_compacted_sample_count) {
        inference_sample_count          = ::cuda::round_up(std::min(generated_sample_count, NetworkSpec.inference_capacity), Shape.training_batch_granularity);
        const std::uint64_t scaled_rays = static_cast<std::uint64_t>(rays_per_batch) * NetworkSpec.training_batch_size / requested_compacted_sample_count;
        const std::uint32_t target_rays = static_cast<std::uint32_t>(std::min(scaled_rays, static_cast<std::uint64_t>(NetworkSpec.training_batch_size)));
        rays_per_batch                  = std::clamp(::cuda::round_up(target_rays, Shape.training_batch_granularity), Shape.training_batch_granularity, NetworkSpec.training_batch_size);
        return occupied_cell_count.data()[0];
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    SamplingState Sampling<Shape, NetworkSpec>::download() const {
        SamplingState state{
            .density                = std::vector<float>(density.size()),
            .density_grid_ema_step  = density_grid_ema_step,
            .rays_per_batch         = rays_per_batch,
            .inference_sample_count = inference_sample_count,
        };
        ::cuda::copy_bytes(stream, density, ::cuda::std::span{state.density.data(), state.density.size()});
        stream.sync();
        return state;
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    void Sampling<Shape, NetworkSpec>::upload(const SamplingState& state) {
        ::cuda::copy_bytes(stream, ::cuda::std::span{state.density.data(), state.density.size()}, density);
        kernels::SamplingKernels<sampling_kernel_shape<Shape, NetworkSpec>>::update_occupancy_grid_from_density_grid(stream, density.data(), density_mean.data(), occupied_cell_counter.data(), occupancy.data());
        ::cuda::copy_bytes(stream, ::cuda::std::span{occupied_cell_counter.data(), 1uz}, occupied_cell_count);
        stream.sync();
        density_grid_ema_step  = state.density_grid_ema_step;
        rays_per_batch         = state.rays_per_batch;
        inference_sample_count = state.inference_sample_count;
    }

    template <SamplingShape Shape, NetworkShape NetworkSpec>
    SamplingDeviceState Sampling<Shape, NetworkSpec>::device_state() const noexcept {
        return {.occupancy = {occupancy.data(), occupancy.size()}};
    }

    template struct Sampling<nerf_synthetic_sampling_shape, nerf_synthetic_network_shape>;
} // namespace physica::reconstruction::instant_ngp
