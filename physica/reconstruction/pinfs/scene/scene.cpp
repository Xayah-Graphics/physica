module;

#include <cuda/std/random>
#include <physica/cuda.h>

module physica.reconstruction.pinfs.scene;

import std;
import physica.reconstruction.dataset.pinf;

namespace physica::reconstruction::pinfs {
    namespace {
        inline constexpr std::uint32_t scene_random_domain = 1u;

        enum class SceneRandomSequence : std::uint32_t {
            perceptual_patch,
            training_pixel,
        };

        const dataset::multiview::FrameSet& find_training_frame_set(const dataset::pinf::Dataset& dataset) {
            return *std::ranges::find_if(dataset.multiview.frame_sets, [](const dataset::multiview::FrameSet& frame_set) { return frame_set.name == "train"; });
        }

        Vector3<float> transform_point(const Matrix4<float>& matrix, const Vector3<float> point) {
            Vector3<float> result;
            for (const std::size_t row : std::views::iota(0uz, 3uz)) result[row] = matrix(row, 3uz) + matrix(row, 0uz) * point.x + matrix(row, 1uz) * point.y + matrix(row, 2uz) * point.z;
            return result;
        }

        Matrix4<float> make_world_from_voxel(const dataset::pinf::Dataset& dataset) {
            Matrix4<float> result = dataset.voxel_transform;
            for (const std::size_t row : std::views::iota(0uz, 3uz))
                for (const std::size_t column : std::views::iota(0uz, 3uz)) result(row, column) *= dataset.voxel_scale[column];
            return result;
        }

        std::vector<Vector3<float>> make_voxel_positions(const dataset::pinf::Dataset& dataset, const Matrix4<float>& world_from_voxel, const std::uint32_t volume_width) {
            const Vector3<float> scale{1.0F, dataset.voxel_scale.y / dataset.voxel_scale.x, dataset.voxel_scale.z / dataset.voxel_scale.x};
            const float minimum_scale  = std::min({scale.x, scale.y, scale.z});
            const float minimum_ratio  = 72.0F / minimum_scale;
            const std::uint32_t width  = std::max(volume_width, static_cast<std::uint32_t>(minimum_ratio * scale.x + 0.5F));
            const std::uint32_t height = static_cast<std::uint32_t>(scale.y * static_cast<float>(width) + 0.5F);
            const std::uint32_t depth  = static_cast<std::uint32_t>(scale.z * static_cast<float>(width) + 0.5F);
            std::vector<Vector3<float>> result(static_cast<std::size_t>(width) * height * depth);
            for (const std::uint32_t z : std::views::iota(0u, depth))
                for (const std::uint32_t y : std::views::iota(0u, height))
                    for (const std::uint32_t x : std::views::iota(0u, width)) result[static_cast<std::size_t>(z) * height * width + static_cast<std::size_t>(y) * width + x] = transform_point(world_from_voxel, {(static_cast<float>(x) + 0.5F) / static_cast<float>(width), (static_cast<float>(y) + 0.5F) / static_cast<float>(height), (static_cast<float>(z) + 0.5F) / static_cast<float>(depth)});
            return result;
        }

        std::size_t maximum_training_pixel_count(const dataset::multiview::FrameSet& training) {
            return std::ranges::max(training.frames | std::views::transform([](const dataset::multiview::Frame& frame) { return static_cast<std::size_t>(frame.extent.width) * frame.extent.height; }));
        }

        Ray3<float> make_ray(const dataset::multiview::Frame& frame, const std::uint32_t pixel) {
            const std::uint32_t x = pixel % frame.extent.width;
            const std::uint32_t y = pixel / frame.extent.width;
            const Vector3<float> camera_direction{(static_cast<float>(x) - frame.intrinsics.principal_x) / frame.intrinsics.focal_x, -(static_cast<float>(y) - frame.intrinsics.principal_y) / frame.intrinsics.focal_y, -1.0F};
            Ray3<float> result;
            for (const std::size_t component : std::views::iota(0uz, 3uz)) {
                result.origin[component]    = frame.world_from_camera(component, 3uz);
                result.direction[component] = frame.world_from_camera(component, 0uz) * camera_direction.x + frame.world_from_camera(component, 1uz) * camera_direction.y + frame.world_from_camera(component, 2uz) * camera_direction.z;
            }
            return result;
        }
    } // namespace

    Scene::Scene(const dataset::pinf::Dataset& source_dataset, const ::cuda::stream_ref source_stream, const std::uint32_t source_rays_per_step, const std::uint32_t inference_batch_size, const std::uint32_t volume_width, const std::uint32_t source_perceptual_stride, const std::uint32_t source_central_crop_steps, const float source_central_crop_fraction, const std::optional<AxisAlignedBox3<float>> normalized_bounds)
        : dataset{source_dataset}, training{find_training_frame_set(dataset)}, voxel_positions{}, background{source_stream, ::cuda::device_default_memory_pool(source_stream.device()), 1uz, ::cuda::no_init}, bounds{source_stream, ::cuda::device_default_memory_pool(source_stream.device()), 1uz, ::cuda::no_init}, stream{source_stream}, rays_per_step{source_rays_per_step}, perceptual_stride{source_perceptual_stride}, central_crop_steps{source_central_crop_steps}, central_crop_fraction{source_central_crop_fraction}, world_from_voxel{make_world_from_voxel(dataset)}, host_rays{stream, ::cuda::pinned_default_memory_pool(), rays_per_step, ::cuda::no_init}, host_targets{stream, ::cuda::pinned_default_memory_pool(), rays_per_step, ::cuda::no_init}, host_volume_points{stream, ::cuda::pinned_default_memory_pool(), inference_batch_size, ::cuda::no_init}, device_rays{stream, ::cuda::device_default_memory_pool(stream.device()), rays_per_step, ::cuda::no_init},
          device_targets{stream, ::cuda::device_default_memory_pool(stream.device()), rays_per_step, ::cuda::no_init}, device_volume_points{stream, ::cuda::device_default_memory_pool(stream.device()), inference_batch_size, ::cuda::no_init}, pixels(rays_per_step), candidates(maximum_training_pixel_count(training)) {
        voxel_positions = make_voxel_positions(dataset, world_from_voxel, volume_width);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const Vector3<float>>{&dataset.background, 1uz}, background);

        AxisAlignedBox3<float> world_bounds;
        if (normalized_bounds) {
            world_bounds.minimum = {(std::numeric_limits<float>::max)(), (std::numeric_limits<float>::max)(), (std::numeric_limits<float>::max)()};
            world_bounds.maximum = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
            for (const std::uint32_t corner : std::views::iota(0u, 8u)) {
                const Vector3<float> normalized{
                    corner & 1u ? normalized_bounds->maximum.x : normalized_bounds->minimum.x,
                    corner & 2u ? normalized_bounds->maximum.y : normalized_bounds->minimum.y,
                    corner & 4u ? normalized_bounds->maximum.z : normalized_bounds->minimum.z,
                };
                const Vector3<float> world = transform_point(world_from_voxel, normalized);
                for (const std::size_t component : std::views::iota(0uz, 3uz)) {
                    world_bounds.minimum[component] = std::min(world_bounds.minimum[component], world[component]);
                    world_bounds.maximum[component] = std::max(world_bounds.maximum[component], world[component]);
                }
            }
        } else {
            world_bounds.minimum = {-(std::numeric_limits<float>::max)(), -(std::numeric_limits<float>::max)(), -(std::numeric_limits<float>::max)()};
            world_bounds.maximum = {(std::numeric_limits<float>::max)(), (std::numeric_limits<float>::max)(), (std::numeric_limits<float>::max)()};
        }
        ::cuda::copy_bytes(stream, ::cuda::std::span<const AxisAlignedBox3<float>>{&world_bounds, 1uz}, bounds);
    }

    TrainingRays Scene::prepare_training_rays(const dataset::multiview::Frame& frame, const bool perceptual, const std::uint32_t seed, const std::uint32_t step) {
        if (perceptual) {
            const std::uint32_t patch_width = static_cast<std::uint32_t>(std::sqrt(static_cast<float>(rays_per_step)));
            std::uint32_t stride            = perceptual_stride + (step + 1u) % 3u - 1u;
            stride                          = std::min(stride, std::min(frame.extent.height - 10u, frame.extent.width - 10u) / patch_width);
            double total_weight{};
            double center_x{};
            double center_y{};
            for (const std::uint32_t y : std::views::iota(0u, frame.extent.height)) {
                for (const std::uint32_t x : std::views::iota(0u, frame.extent.width)) {
                    const std::size_t index = (static_cast<std::size_t>(y) * frame.extent.width + x) * 4uz;
                    const float weight      = (std::abs(static_cast<float>(frame.rgba[index]) / 255.0F - dataset.background.x) + std::abs(static_cast<float>(frame.rgba[index + 1uz]) / 255.0F - dataset.background.y) + std::abs(static_cast<float>(frame.rgba[index + 2uz]) / 255.0F - dataset.background.z)) / 3.0F;
                    total_weight += weight;
                    center_x += static_cast<double>(x) * weight;
                    center_y += static_cast<double>(y) * weight;
                }
            }
            ::cuda::std::philox4x32 random{seed};
            random.set_counter({scene_random_domain, static_cast<std::uint32_t>(SceneRandomSequence::perceptual_patch), step, 0u});
            const float uniform_x        = std::max(static_cast<float>(random() >> 8u) * 0x1.0p-24F, (std::numeric_limits<float>::min)());
            const float uniform_y        = static_cast<float>(random() >> 8u) * 0x1.0p-24F;
            const float magnitude        = std::sqrt(-2.0F * std::log(uniform_x));
            const float normal_x         = magnitude * std::cos(2.0F * std::numbers::pi_v<float> * uniform_y);
            const float normal_y         = magnitude * std::sin(2.0F * std::numbers::pi_v<float> * uniform_y);
            const float radius           = static_cast<float>(patch_width * stride) * 0.5F;
            const std::uint32_t offset_x = static_cast<std::uint32_t>(std::clamp(static_cast<float>(center_x / total_weight) + radius / 3.0F * normal_x - radius, 10.0F, static_cast<float>(frame.extent.width - patch_width * stride - 10u)));
            const std::uint32_t offset_y = static_cast<std::uint32_t>(std::clamp(static_cast<float>(center_y / total_weight) + radius / 3.0F * normal_y - radius, 10.0F, static_cast<float>(frame.extent.height - patch_width * stride - 10u)));
            for (const std::uint32_t y : std::views::iota(0u, patch_width))
                for (const std::uint32_t x : std::views::iota(0u, patch_width)) pixels[static_cast<std::size_t>(y) * patch_width + x] = (offset_y + y * stride) * frame.extent.width + offset_x + x * stride;
        } else {
            const std::uint32_t half_width      = step + 1u < central_crop_steps ? static_cast<std::uint32_t>(static_cast<float>(frame.extent.width / 2u) * central_crop_fraction) : frame.extent.width / 2u;
            const std::uint32_t half_height     = step + 1u < central_crop_steps ? static_cast<std::uint32_t>(static_cast<float>(frame.extent.height / 2u) * central_crop_fraction) : frame.extent.height / 2u;
            const std::uint32_t begin_x         = frame.extent.width / 2u - half_width;
            const std::uint32_t begin_y         = frame.extent.height / 2u - half_height;
            const std::uint32_t candidate_count = half_width * 2u * half_height * 2u;
            for (const std::uint32_t index : std::views::iota(0u, candidate_count)) candidates[index] = (begin_y + index / (half_width * 2u)) * frame.extent.width + begin_x + index % (half_width * 2u);
            for (const std::uint32_t ray : std::views::iota(0u, rays_per_step)) {
                ::cuda::std::philox4x32 random{seed};
                random.set_counter({scene_random_domain, static_cast<std::uint32_t>(SceneRandomSequence::training_pixel), step, ray});
                const std::uint32_t selected = ray + random() % (candidate_count - ray);
                std::swap(candidates[ray], candidates[selected]);
                pixels[ray] = candidates[ray];
            }
        }

        for (const std::uint32_t ray : std::views::iota(0u, rays_per_step)) {
            const std::uint32_t pixel = pixels[ray];
            host_rays.data()[ray]     = make_ray(frame, pixel);
            host_targets.data()[ray]  = {
                static_cast<float>(frame.rgba[static_cast<std::size_t>(pixel) * 4uz]) / 255.0F,
                static_cast<float>(frame.rgba[static_cast<std::size_t>(pixel) * 4uz + 1uz]) / 255.0F,
                static_cast<float>(frame.rgba[static_cast<std::size_t>(pixel) * 4uz + 2uz]) / 255.0F,
            };
        }
        ::cuda::copy_bytes(stream, host_rays, device_rays);
        ::cuda::copy_bytes(stream, host_targets, device_targets);
        return {.rays = {.data = device_rays.data(), .count = rays_per_step}, .target = device_targets.data(), .time = frame.time};
    }

    DeviceRays Scene::prepare_rendering_rays(const dataset::multiview::Frame& frame, const std::uint32_t offset, const std::uint32_t count) {
        for (const std::uint32_t ray : std::views::iota(0u, count)) host_rays.data()[ray] = make_ray(frame, offset + ray);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const Ray3<float>>{host_rays.data(), count}, ::cuda::std::span<Ray3<float>>{device_rays.data(), count});
        return {.data = device_rays.data(), .count = count};
    }

    const SpacetimePoint* Scene::prepare_volume_points(const float time, const Vector3<std::uint32_t> resolution, const std::size_t offset, const std::uint32_t count) {
        for (const std::uint32_t local : std::views::iota(0u, count)) {
            const std::size_t index          = offset + local;
            const std::uint32_t x            = static_cast<std::uint32_t>(index % resolution.x);
            const std::uint32_t y            = static_cast<std::uint32_t>(index / resolution.x % resolution.y);
            const std::uint32_t z            = static_cast<std::uint32_t>(index / resolution.x / resolution.y);
            host_volume_points.data()[local] = {
                .position = transform_point(world_from_voxel, {(static_cast<float>(x) + 0.5F) / static_cast<float>(resolution.x), (static_cast<float>(y) + 0.5F) / static_cast<float>(resolution.y), (static_cast<float>(z) + 0.5F) / static_cast<float>(resolution.z)}),
                .time     = time,
            };
        }
        ::cuda::copy_bytes(stream, ::cuda::std::span<const SpacetimePoint>{host_volume_points.data(), count}, ::cuda::std::span<SpacetimePoint>{device_volume_points.data(), count});
        return device_volume_points.data();
    }
} // namespace physica::reconstruction::pinfs
