module;

#include "kernels.h"
#include <physica/cuda.h>

module physica.reconstruction.pinfs.sampling;

import std;
import physica.reconstruction.pinfs.network;
import physica.reconstruction.pinfs.scene;

namespace physica::reconstruction::pinfs {
    Sampling::Sampling(const ::cuda::stream_ref source_stream, const std::uint32_t source_maximum_ray_count, const std::uint32_t source_maximum_samples_per_ray, const std::uint32_t source_maximum_importance_samples)
        : stream{source_stream}, coarse_z{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray, ::cuda::no_init}, coarse_positions{stream, ::cuda::device_default_memory_pool(stream.device()), coarse_z.size(), ::cuda::no_init}, coarse_dynamic_points{stream, ::cuda::device_default_memory_pool(stream.device()), coarse_z.size(), ::cuda::no_init}, coarse_directions{stream, ::cuda::device_default_memory_pool(stream.device()), coarse_z.size(), ::cuda::no_init}, fine_z{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray, ::cuda::no_init}, fine_positions{stream, ::cuda::device_default_memory_pool(stream.device()), fine_z.size(), ::cuda::no_init}, fine_dynamic_points{stream, ::cuda::device_default_memory_pool(stream.device()), fine_z.size(), ::cuda::no_init},
          fine_directions{stream, ::cuda::device_default_memory_pool(stream.device()), fine_z.size(), ::cuda::no_init}, pdf_z{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_importance_samples, ::cuda::no_init}, upsample_z_a{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray, ::cuda::no_init}, upsample_z_b{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray, ::cuda::no_init}, upsample_sdf_a{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray, ::cuda::no_init}, upsample_sdf_b{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray, ::cuda::no_init},
          new_z{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_importance_samples, ::cuda::no_init}, new_positions{stream, ::cuda::device_default_memory_pool(stream.device()), new_z.size(), ::cuda::no_init}, new_sdf{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_importance_samples, ::cuda::no_init}, current_upsample_z{upsample_z_a.data()}, next_upsample_z{upsample_z_b.data()}, current_upsample_sdf{upsample_sdf_a.data()}, next_upsample_sdf{upsample_sdf_b.data()} {}

    RaySamples Sampling::coarse(const DeviceRays rays, const std::uint32_t samples_per_ray, const float near_distance, const float far_distance, const float time, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t ray_offset, const bool perturb) {
        kernels::sample_coarse(stream, rays.data, coarse_z.data(), coarse_positions.data(), reinterpret_cast<float*>(coarse_dynamic_points.data()), coarse_directions.data(), rays.count, samples_per_ray, near_distance, far_distance, time, seed, step, ray_offset * samples_per_ray, perturb);
        return {.z = coarse_z.data(), .positions = coarse_positions.data(), .dynamic_points = coarse_dynamic_points.data(), .directions = coarse_directions.data(), .ray_count = rays.count, .samples_per_ray = samples_per_ray};
    }

    void Sampling::warp(const RaySamples samples, const ConstDeviceTensor velocity, const float amount, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t ray_offset) {
        kernels::warp_points(stream, reinterpret_cast<float*>(samples.dynamic_points), velocity.values, samples.sample_count(), amount, seed, step, ray_offset * samples.samples_per_ray);
    }

    void Sampling::sample_pdf(const RaySamples& coarse_samples, const float* coarse_weights, const std::uint32_t importance_count, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t ray_offset, const bool deterministic) {
        kernels::sample_pdf(stream, coarse_samples.z, coarse_weights, pdf_z.data(), coarse_samples.ray_count, coarse_samples.samples_per_ray, 1u, coarse_samples.samples_per_ray - 2u, importance_count, seed, step, ray_offset, deterministic);
        kernels::sort_samples(stream, pdf_z.data(), coarse_samples.ray_count, importance_count);
    }

    void Sampling::begin_neus_upsampling(const RaySamples& coarse_samples, const ConstDeviceTensor sdf) {
        current_upsample_z    = upsample_z_a.data();
        next_upsample_z       = upsample_z_b.data();
        current_upsample_sdf  = upsample_sdf_a.data();
        next_upsample_sdf     = upsample_sdf_b.data();
        upsample_ray_count    = coarse_samples.ray_count;
        upsample_sample_count = coarse_samples.samples_per_ray;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{coarse_samples.z, static_cast<std::size_t>(upsample_ray_count) * upsample_sample_count}, ::cuda::std::span<float>{current_upsample_z, static_cast<std::size_t>(upsample_ray_count) * upsample_sample_count});
        kernels::extract_sdf(stream, sdf.values, current_upsample_sdf, upsample_ray_count * upsample_sample_count);
    }

    RaySamples Sampling::neus_upsampling_positions(const DeviceRays rays, const std::uint32_t new_sample_count, const float inverse_deviation) {
        pending_new_sample_count = new_sample_count;
        kernels::neus_pdf(stream, current_upsample_z, current_upsample_sdf, new_z.data(), upsample_ray_count, upsample_sample_count, new_sample_count, inverse_deviation);
        kernels::positions_from_z(stream, rays.data, new_z.data(), new_positions.data(), rays.count, new_sample_count);
        return {.z = new_z.data(), .positions = new_positions.data(), .ray_count = rays.count, .samples_per_ray = new_sample_count};
    }

    void Sampling::commit_neus_upsampling(const ConstDeviceTensor source_new_sdf, const bool last) {
        const float* compact_new_sdf = nullptr;
        if (!last) {
            kernels::extract_sdf(stream, source_new_sdf.values, new_sdf.data(), upsample_ray_count * pending_new_sample_count);
            compact_new_sdf = new_sdf.data();
        }
        kernels::merge_samples(stream, current_upsample_z, current_upsample_sdf, upsample_sample_count, new_z.data(), compact_new_sdf, pending_new_sample_count, next_upsample_z, last ? nullptr : next_upsample_sdf, upsample_ray_count);
        std::swap(current_upsample_z, next_upsample_z);
        std::swap(current_upsample_sdf, next_upsample_sdf);
        upsample_sample_count += pending_new_sample_count;
    }

    RaySamples Sampling::fine(const DeviceRays rays, const RaySamples& coarse_samples, const std::uint32_t pdf_sample_count, const std::uint32_t static_sample_count, const float time) {
        if (static_sample_count == 0u) kernels::merge_samples(stream, coarse_samples.z, nullptr, coarse_samples.samples_per_ray, pdf_z.data(), nullptr, pdf_sample_count, fine_z.data(), nullptr, rays.count);
        else kernels::merge_samples(stream, current_upsample_z, nullptr, coarse_samples.samples_per_ray + static_sample_count, pdf_z.data(), nullptr, pdf_sample_count, fine_z.data(), nullptr, rays.count);
        const std::uint32_t samples_per_ray = coarse_samples.samples_per_ray + pdf_sample_count + static_sample_count;
        kernels::samples_from_z(stream, rays.data, fine_z.data(), fine_positions.data(), reinterpret_cast<float*>(fine_dynamic_points.data()), fine_directions.data(), rays.count, samples_per_ray, time);
        return {.z = fine_z.data(), .positions = fine_positions.data(), .dynamic_points = fine_dynamic_points.data(), .directions = fine_directions.data(), .ray_count = rays.count, .samples_per_ray = samples_per_ray};
    }
} // namespace physica::reconstruction::pinfs
