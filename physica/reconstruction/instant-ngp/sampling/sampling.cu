#include "kernels.h"
#include <math/math.h>
#include <cuda/cmath>
#include <cuda/std/numbers>
#include <algorithm>
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/random>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace physica::reconstruction::instant_ngp {
    inline constexpr kernels::SamplingKernelShape sampling_cuda_shape{
        .occupancy_grid_size             = 128u,
        .ray_step_count                  = 1024u,
        .training_batch_granularity      = 16u * 8u,
        .evaluation_tile_rays            = 4096u,
        .density_grid_warmup_steps       = 256u,
        .density_grid_skip_interval      = 16u,
        .density_grid_max_skip           = 16u,
        .density_grid_warmup_samples     = 128u * 128u * 128u,
        .density_grid_uniform_samples    = 128u * 128u * 128u / 4u,
        .density_grid_nonuniform_samples = 128u * 128u * 128u / 4u,
        .snap_to_pixel_centers           = true,
        .training_batch_size             = 1u << 18u,
        .sample_capacity                 = (1u << 18u) * 16u,
    };

    template <kernels::SamplingKernelShape Shape>
    struct SamplingLayout final {
        inline static constexpr std::uint32_t nerf_grid_size                         = Shape.occupancy_grid_size;
        inline static constexpr std::uint32_t nerf_grid_cells                        = Shape.occupancy_grid_size * Shape.occupancy_grid_size * Shape.occupancy_grid_size;
        inline static constexpr std::uint32_t nerf_steps                             = Shape.ray_step_count;
        inline static constexpr float min_cone_stepsize                              = ::cuda::std::numbers::sqrt3_v<float> / static_cast<float>(Shape.ray_step_count);
        inline static constexpr bool snap_to_pixel_centers                           = Shape.snap_to_pixel_centers;
        inline static constexpr std::uint32_t density_grid_warmup_steps              = Shape.density_grid_warmup_steps;
        inline static constexpr std::uint32_t density_grid_skip_interval             = Shape.density_grid_skip_interval;
        inline static constexpr std::uint32_t density_grid_max_skip                  = Shape.density_grid_max_skip;
        inline static constexpr std::uint32_t density_grid_warmup_samples            = Shape.density_grid_warmup_samples;
        inline static constexpr std::uint32_t density_grid_steady_uniform_samples    = Shape.density_grid_uniform_samples;
        inline static constexpr std::uint32_t density_grid_steady_nonuniform_samples = Shape.density_grid_nonuniform_samples;
        inline static constexpr std::uint32_t evaluation_tile_rays                   = Shape.evaluation_tile_rays;
        inline static constexpr std::uint32_t network_batch_granularity              = Shape.training_batch_granularity;
    };
} // namespace physica::reconstruction::instant_ngp

namespace physica::reconstruction::instant_ngp::kernels {
    namespace {
    inline constexpr std::uint32_t sampling_random_domain = 1u;
    static __device__ constexpr AxisAlignedBox3<float> unit_box{{0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F}};

    enum class SamplingRandomSequence : std::uint32_t {
        density_grid,
        training_pixel,
        raymarch,
    };

    inline __device__ std::uint32_t morton_expand_3d(std::uint32_t value) {
        value = (value * 0x00010001u) & 0xFF0000FFu;
        value = (value * 0x00000101u) & 0x0F00F00Fu;
        value = (value * 0x00000011u) & 0xC30C30C3u;
        return (value * 0x00000005u) & 0x49249249u;
    }

    inline __device__ std::uint32_t morton_compact_3d(std::uint32_t value) {
        value &= 0x49249249u;
        value = (value | (value >> 2u)) & 0xC30C30C3u;
        value = (value | (value >> 4u)) & 0x0F00F00Fu;
        value = (value | (value >> 8u)) & 0xFF0000FFu;
        return (value | (value >> 16u)) & 0x0000FFFFu;
    }

    template <std::uint32_t GridSize>
    inline __device__ bool is_occupancy_grid_cell_occupied(const Vector3<float> pos, const std::uint8_t* occupancy) {
        const int x = static_cast<int>(pos.x * static_cast<float>(GridSize));
        const int y = static_cast<int>(pos.y * static_cast<float>(GridSize));
        const int z = static_cast<int>(pos.z * static_cast<float>(GridSize));
        if (x < 0 || x >= static_cast<int>(GridSize) || y < 0 || y >= static_cast<int>(GridSize) || z < 0 || z >= static_cast<int>(GridSize)) return false;
        const std::uint32_t index = morton_expand_3d(static_cast<std::uint32_t>(x)) | (morton_expand_3d(static_cast<std::uint32_t>(y)) << 1u) | (morton_expand_3d(static_cast<std::uint32_t>(z)) << 2u);
        return (occupancy[index / 8u] & (1u << (index % 8u))) != 0u;
    }

    template <std::uint32_t GridSize, std::uint32_t RayStepCount>
    inline __device__ float advance_to_next_density_voxel(const float t, const Vector3<float> pos, const Vector3<float> direction, const Vector3<float> inv_direction) {
        constexpr auto scale              = static_cast<float>(GridSize);
        constexpr float minimum_step_size = ::cuda::std::numbers::sqrt3_v<float> / static_cast<float>(RayStepCount);
        const Vector3<float> p                    = {(pos.x - 0.5f) * scale, (pos.y - 0.5f) * scale, (pos.z - 0.5f) * scale};
        const float tx                    = (::cuda::std::floor(p.x + 0.5f + 0.5f * ::cuda::std::copysign(1.0f, direction.x)) - p.x) * inv_direction.x;
        const float ty                    = (::cuda::std::floor(p.y + 0.5f + 0.5f * ::cuda::std::copysign(1.0f, direction.y)) - p.y) * inv_direction.y;
        const float tz                    = (::cuda::std::floor(p.z + 0.5f + 0.5f * ::cuda::std::copysign(1.0f, direction.z)) - p.z) * inv_direction.z;
        const float t_target              = t + ::cuda::std::max(::cuda::std::min(::cuda::std::min(tx, ty), tz) / scale, 0.0f);
        return t + ::cuda::std::ceil(::cuda::std::max((t_target - t) / minimum_step_size, 0.5F)) * minimum_step_size;
    }

    inline __device__ Ray3<float> make_camera_ray(const float* const camera, const float pixel_x, const float pixel_y, const float focal_x, const float focal_y, const float principal_x, const float principal_y) {
        const float ray_x     = (pixel_x - principal_x) / focal_x;
        const float ray_y     = (pixel_y - principal_y) / focal_y;
        const Vector3<float> camera_x = {camera[0], camera[1], camera[2]};
        const Vector3<float> camera_y = {camera[3], camera[4], camera[5]};
        const Vector3<float> camera_z = {camera[6], camera[7], camera[8]};
        Vector3<float> direction      = {
            camera_x.x * ray_x + camera_y.x * ray_y + camera_z.x,
            camera_x.y * ray_x + camera_y.y * ray_y + camera_z.y,
            camera_x.z * ray_x + camera_y.z * ray_y + camera_z.z,
        };
        const float length = ::cuda::std::hypot(direction.x, direction.y, direction.z);
        return {
            .origin    = {camera[9], camera[10], camera[11]},
            .direction = {direction.x / length, direction.y / length, direction.z / length},
        };
    }

    inline __device__ std::uint32_t count_occupied_samples(const Ray3<float> ray, const float start_t, const std::uint8_t* const occupancy) {
        const Vector3<float> inverse_direction = {1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z};
        std::uint32_t sample_count     = 0u;
        float t                        = start_t;
        while (sample_count < SamplingLayout<sampling_cuda_shape>::nerf_steps) {
            const Vector3<float> position = {ray.origin.x + ray.direction.x * t, ray.origin.y + ray.direction.y * t, ray.origin.z + ray.direction.z * t};
            if (!contains(unit_box, position)) break;
            if (is_occupancy_grid_cell_occupied<SamplingLayout<sampling_cuda_shape>::nerf_grid_size>(position, occupancy)) {
                ++sample_count;
                t += SamplingLayout<sampling_cuda_shape>::min_cone_stepsize;
            } else {
                t = advance_to_next_density_voxel<SamplingLayout<sampling_cuda_shape>::nerf_grid_size, SamplingLayout<sampling_cuda_shape>::nerf_steps>(t, position, ray.direction, inverse_direction);
            }
        }
        return sample_count;
    }

    inline __device__ void write_occupied_samples(const Ray3<float> ray, const float start_t, const std::uint32_t sample_count, const std::uint32_t output_offset, const std::uint8_t* const occupancy, float* const output) {
        const Vector3<float> inverse_direction = {1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z};
        const Vector3<float> warped_direction  = {(ray.direction.x + 1.0f) * 0.5f, (ray.direction.y + 1.0f) * 0.5f, (ray.direction.z + 1.0f) * 0.5f};
        std::uint32_t written          = 0u;
        float t                        = start_t;
        while (written < sample_count) {
            const Vector3<float> position = {ray.origin.x + ray.direction.x * t, ray.origin.y + ray.direction.y * t, ray.origin.z + ray.direction.z * t};
            if (!contains(unit_box, position)) break;
            if (is_occupancy_grid_cell_occupied<SamplingLayout<sampling_cuda_shape>::nerf_grid_size>(position, occupancy)) {
                float* sample = output + static_cast<std::uint64_t>(output_offset + written) * 7u;
                sample[0]     = position.x;
                sample[1]     = position.y;
                sample[2]     = position.z;
                sample[3]     = SamplingLayout<sampling_cuda_shape>::min_cone_stepsize;
                sample[4]     = warped_direction.x;
                sample[5]     = warped_direction.y;
                sample[6]     = warped_direction.z;
                ++written;
                t += SamplingLayout<sampling_cuda_shape>::min_cone_stepsize;
            } else {
                t = advance_to_next_density_voxel<SamplingLayout<sampling_cuda_shape>::nerf_grid_size, SamplingLayout<sampling_cuda_shape>::nerf_steps>(t, position, ray.direction, inverse_direction);
            }
        }
    }

    __global__ void seed_density_grid_visibility_kernel(float* __restrict__ density_grid_values, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_x, const float focal_y, const float principal_x, const float principal_y, const float* __restrict__ camera) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= SamplingLayout<sampling_cuda_shape>::nerf_grid_cells) return;

        const std::uint32_t x = morton_compact_3d(i >> 0u);
        const std::uint32_t y = morton_compact_3d(i >> 1u);
        const std::uint32_t z = morton_compact_3d(i >> 2u);

        constexpr float voxel_size      = 1.0f / static_cast<float>(SamplingLayout<sampling_cuda_shape>::nerf_grid_size);
        const Vector3<float> cell_min           = {static_cast<float>(x) * voxel_size, static_cast<float>(y) * voxel_size, static_cast<float>(z) * voxel_size};
        bool visible_to_training_camera = false;

        for (std::uint32_t frame = 0u; frame < frame_count && !visible_to_training_camera; ++frame) {
            const float* frame_camera = camera + static_cast<std::uint64_t>(frame) * 12u;
            const Vector3<float> camera_x     = {frame_camera[0], frame_camera[1], frame_camera[2]};
            const Vector3<float> camera_y     = {frame_camera[3], frame_camera[4], frame_camera[5]};
            const Vector3<float> camera_z     = {frame_camera[6], frame_camera[7], frame_camera[8]};
            const Vector3<float> origin       = {frame_camera[9], frame_camera[10], frame_camera[11]};

            for (std::uint32_t corner = 0u; corner < 8u && !visible_to_training_camera; ++corner) {
                const Vector3<float> pos = {
                    cell_min.x + ((corner & 1u) != 0u ? voxel_size : 0.0f),
                    cell_min.y + ((corner & 2u) != 0u ? voxel_size : 0.0f),
                    cell_min.z + ((corner & 4u) != 0u ? voxel_size : 0.0f),
                };
                const Vector3<float> relative = {pos.x - origin.x, pos.y - origin.y, pos.z - origin.z};
                const float distance  = ::cuda::std::hypot(relative.x, relative.y, relative.z);
                if (distance <= 0.0f) continue;

                const float local_x = relative.x * camera_x.x + relative.y * camera_x.y + relative.z * camera_x.z;
                const float local_y = relative.x * camera_y.x + relative.y * camera_y.y + relative.z * camera_y.z;
                const float local_z = relative.x * camera_z.x + relative.y * camera_z.y + relative.z * camera_z.z;
                if (local_z / distance < 1e-4f) continue;

                const float pixel_x = local_x * focal_x / local_z + principal_x;
                const float pixel_y = local_y * focal_y / local_z + principal_y;
                if (pixel_x <= 0.0f || pixel_y <= 0.0f || pixel_x >= static_cast<float>(width) || pixel_y >= static_cast<float>(height)) continue;

                const float ray_x          = (pixel_x - principal_x) / focal_x;
                const float ray_y          = (pixel_y - principal_y) / focal_y;
                const Vector3<float> ray_direction = {
                    camera_x.x * ray_x + camera_y.x * ray_y + camera_z.x,
                    camera_x.y * ray_x + camera_y.y * ray_y + camera_z.y,
                    camera_x.z * ray_x + camera_y.z * ray_y + camera_z.z,
                };
                const float ray_length = ::cuda::std::hypot(ray_direction.x, ray_direction.y, ray_direction.z);
                if (ray_length <= 0.0f) continue;

                const Vector3<float> ray_normalized = {ray_direction.x / ray_length, ray_direction.y / ray_length, ray_direction.z / ray_length};
                const Vector3<float> direction      = {relative.x / distance, relative.y / distance, relative.z / distance};
                const float direction_delta = ::cuda::std::hypot(ray_normalized.x - direction.x, ray_normalized.y - direction.y, ray_normalized.z - direction.z);
                visible_to_training_camera  = direction_delta < 1e-3f;
            }
        }

        density_grid_values[i] = visible_to_training_camera ? 0.0f : -1.0f;
    }

    __global__ void generate_density_grid_samples_kernel(const std::uint32_t sample_count, const std::uint32_t seed, const std::uint32_t density_grid_ema_step, const std::uint32_t phase, const float threshold, const float* __restrict__ density_grid_values, float* __restrict__ sample_coords, std::uint32_t* __restrict__ density_grid_indices) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= sample_count) return;

        ::cuda::std::philox4x32 random{seed};
        random.set_counter({sampling_random_domain, ::cuda::std::to_underlying(SamplingRandomSequence::density_grid), density_grid_ema_step * 2u + phase, i});
        ::cuda::std::uniform_real_distribution<float> unit_distribution;

        std::uint32_t idx = 0u;
        for (std::uint32_t j = 0u; j < 10u; ++j) {
            idx = static_cast<std::uint32_t>(((static_cast<std::uint64_t>(i) + static_cast<std::uint64_t>(density_grid_ema_step) * sample_count) * 56924617ull + static_cast<std::uint64_t>(j) * 19349663ull + 96925573ull) % SamplingLayout<sampling_cuda_shape>::nerf_grid_cells);
            if (density_grid_values[idx] > threshold) break;
        }

        const std::uint32_t x = morton_compact_3d(idx >> 0u);
        const std::uint32_t y = morton_compact_3d(idx >> 1u);
        const std::uint32_t z = morton_compact_3d(idx >> 2u);

        float* coord            = sample_coords + static_cast<std::uint64_t>(i) * 7u;
        coord[0]                = (static_cast<float>(x) + unit_distribution(random)) / static_cast<float>(SamplingLayout<sampling_cuda_shape>::nerf_grid_size);
        coord[1]                = (static_cast<float>(y) + unit_distribution(random)) / static_cast<float>(SamplingLayout<sampling_cuda_shape>::nerf_grid_size);
        coord[2]                = (static_cast<float>(z) + unit_distribution(random)) / static_cast<float>(SamplingLayout<sampling_cuda_shape>::nerf_grid_size);
        coord[3]                = SamplingLayout<sampling_cuda_shape>::min_cone_stepsize;
        coord[4]                = 0.5f;
        coord[5]                = 0.5f;
        coord[6]                = 0.5f;
        density_grid_indices[i] = idx;
    }

    __global__ void splat_density_grid_samples_kernel(const std::uint32_t sample_count, const std::uint32_t* __restrict__ density_grid_indices, const __half* __restrict__ density_output, float* __restrict__ density_grid_scratch) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= sample_count) return;

        const float thickness = ::cuda::std::exp(__half2float(density_output[i])) * SamplingLayout<sampling_cuda_shape>::min_cone_stepsize;
        atomicMax(reinterpret_cast<unsigned int*>(density_grid_scratch + density_grid_indices[i]), ::cuda::std::bit_cast<unsigned int>(thickness));
    }

    __global__ void update_density_grid_ema_kernel(const float* __restrict__ density_grid_scratch, float* __restrict__ density_grid_values) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= SamplingLayout<sampling_cuda_shape>::nerf_grid_cells) return;

        const float prev_val   = density_grid_values[i];
        const float importance = density_grid_scratch[i];
        density_grid_values[i] = prev_val < 0.0f ? prev_val : ::cuda::std::max(prev_val * 0.95F, importance);
    }

    __global__ void reduce_density_grid_mean_kernel(const float* __restrict__ density_grid_values, float* __restrict__ density_grid_mean) {
        __shared__ float sums[1024];
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        float sum             = 0.0f;
        if (i < SamplingLayout<sampling_cuda_shape>::nerf_grid_cells / 4u) {
            const float4 values = reinterpret_cast<const float4*>(density_grid_values)[i];
            sum                 = ::cuda::std::max(values.x, 0.0f) + ::cuda::std::max(values.y, 0.0f) + ::cuda::std::max(values.z, 0.0f) + ::cuda::std::max(values.w, 0.0f);
        }

        sums[threadIdx.x] = sum;
        __syncthreads();

        for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0u) atomicAdd(density_grid_mean, sums[0] / static_cast<float>(SamplingLayout<sampling_cuda_shape>::nerf_grid_cells));
    }

    __global__ void build_density_grid_bitfield_kernel(const float* __restrict__ density_grid_values, const float* __restrict__ density_grid_mean, std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ occupancy_grid_occupied_count) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= SamplingLayout<sampling_cuda_shape>::nerf_grid_cells / 8u) return;

        std::uint8_t bits     = 0u;
        const float threshold = ::cuda::std::min(0.01F, *density_grid_mean);
        for (std::uint8_t j = 0u; j < 8u; ++j) bits |= density_grid_values[i * 8u + j] > threshold ? static_cast<std::uint8_t>(1u << j) : 0u;

        occupancy[i]                 = bits;
        const std::uint32_t occupied = ::cuda::std::popcount(static_cast<std::uint32_t>(bits));
        if (occupied != 0u) atomicAdd(occupancy_grid_occupied_count, occupied);
    }

    __global__ void generate_training_samples_kernel(const std::uint32_t rays_per_batch, const std::uint32_t sample_limit, const std::uint32_t seed, const std::uint32_t current_step, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_x, const float focal_y, const float principal_x, const float principal_y, const float* __restrict__ camera, const std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ ray_counter, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ target_pixel_indices_out, float* __restrict__ rays_out, std::uint32_t* __restrict__ numsteps_out, float* __restrict__ coords_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= rays_per_batch) return;

        const std::uint32_t image = static_cast<std::uint32_t>((static_cast<std::uint64_t>(i) * frame_count) / rays_per_batch) % frame_count;
        const float* frame_camera = camera + static_cast<std::uint64_t>(image) * 12u;

        ::cuda::std::philox4x32 pixel_random{seed};
        pixel_random.set_counter({sampling_random_domain, ::cuda::std::to_underlying(SamplingRandomSequence::training_pixel), current_step, i});
        ::cuda::std::uniform_real_distribution<float> unit_distribution;
        float u                     = unit_distribution(pixel_random);
        float v                     = unit_distribution(pixel_random);
        const std::uint32_t pixel_x = ::cuda::std::min(static_cast<std::uint32_t>(u * static_cast<float>(width)), width - 1u);
        const std::uint32_t pixel_y = ::cuda::std::min(static_cast<std::uint32_t>(v * static_cast<float>(height)), height - 1u);
        if constexpr (SamplingLayout<sampling_cuda_shape>::snap_to_pixel_centers) {
            u = (static_cast<float>(pixel_x) + 0.5f) / static_cast<float>(width);
            v = (static_cast<float>(pixel_y) + 0.5f) / static_cast<float>(height);
        }
        const Ray3<float> ray = make_camera_ray(frame_camera, u * static_cast<float>(width), v * static_cast<float>(height), focal_x, focal_y, principal_x, principal_y);

        float tmin = 0.0f;
        if (!intersect(unit_box, ray, tmin)) return;

        ::cuda::std::philox4x32 raymarch_random{seed};
        raymarch_random.set_counter({sampling_random_domain, ::cuda::std::to_underlying(SamplingRandomSequence::raymarch), current_step, i});
        const float start_t          = tmin + unit_distribution(raymarch_random) * SamplingLayout<sampling_cuda_shape>::min_cone_stepsize;
        const std::uint32_t numsteps = count_occupied_samples(ray, start_t, occupancy);
        if (numsteps == 0u) return;

        const std::uint32_t base = atomicAdd(sample_counter, numsteps);
        if (base + numsteps > sample_limit) return;

        const std::uint32_t ray_index       = atomicAdd(ray_counter, 1u);
        target_pixel_indices_out[ray_index] = static_cast<std::uint32_t>(pixel_x + static_cast<std::uint64_t>(pixel_y) * width + static_cast<std::uint64_t>(image) * width * height);

        float* ray_out = rays_out + static_cast<std::uint64_t>(ray_index) * 3u;
        ray_out[0]     = ray.origin.x;
        ray_out[1]     = ray.origin.y;
        ray_out[2]     = ray.origin.z;

        numsteps_out[ray_index * 2u + 0u] = numsteps;
        numsteps_out[ray_index * 2u + 1u] = base;
        write_occupied_samples(ray, start_t, numsteps, base, occupancy, coords_out);
    }

    __global__ void generate_evaluation_samples_kernel(const std::uint32_t tile_pixels, const std::uint32_t pixel_offset, const std::uint32_t width, const std::uint32_t height, const float focal_x, const float focal_y, const float principal_x, const float principal_y, const float* __restrict__ evaluation_camera, const std::uint32_t evaluation_image_index, const std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ numsteps_out, float* __restrict__ coords_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= tile_pixels) return;

        numsteps_out[i * 2u + 0u] = 0u;
        numsteps_out[i * 2u + 1u] = 0u;

        const std::uint32_t global_pixel = pixel_offset + i;
        const std::uint32_t pixel_x      = global_pixel % width;
        const std::uint32_t pixel_y      = global_pixel / width;
        const float* frame_camera        = evaluation_camera + static_cast<std::uint64_t>(evaluation_image_index) * 12u;

        const Ray3<float> ray = make_camera_ray(frame_camera, static_cast<float>(pixel_x) + 0.5f, static_cast<float>(pixel_y) + 0.5f, focal_x, focal_y, principal_x, principal_y);

        float tmin = 0.0f;
        if (!intersect(unit_box, ray, tmin)) return;

        const float start_t          = tmin + 0.5f * SamplingLayout<sampling_cuda_shape>::min_cone_stepsize;
        const std::uint32_t numsteps = count_occupied_samples(ray, start_t, occupancy);
        if (numsteps == 0u) return;

        const std::uint32_t base  = atomicAdd(sample_counter, numsteps);
        numsteps_out[i * 2u + 0u] = numsteps;
        numsteps_out[i * 2u + 1u] = base;
        write_occupied_samples(ray, start_t, numsteps, base, occupancy, coords_out);
    }

    __global__ void pad_evaluation_rollover_coords_kernel(const std::uint32_t used_sample_count, const std::uint32_t padded_sample_count, float* __restrict__ inout) {
        const std::uint32_t i             = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t used_elements = used_sample_count * 7u;
        if (used_sample_count == 0u || i < used_elements || i >= padded_sample_count * 7u) return;
        inout[i] = inout[i % used_elements];
    }

    } // namespace
} // namespace physica::reconstruction::instant_ngp::kernels

namespace physica::reconstruction::instant_ngp::kernels {
    template <SamplingKernelShape Shape>
    void SamplingKernels<Shape>::set_occupancy_grid_full(const ::cuda::stream_ref stream, std::uint8_t* const occupancy, std::uint32_t* const occupancy_grid_occupied_count) {
        ::cuda::fill_bytes(stream, ::cuda::std::span{occupancy, SamplingLayout<Shape>::nerf_grid_cells / 8u}, 0xFFu);
        const std::uint32_t occupied_count = SamplingLayout<Shape>::nerf_grid_cells;
        ::cuda::copy_bytes(stream, ::cuda::std::span{&occupied_count, 1u}, ::cuda::std::span{occupancy_grid_occupied_count, 1u});
    }

    template <SamplingKernelShape Shape>
    void SamplingKernels<Shape>::sample_training_batch(const ::cuda::stream_ref stream, const float* const camera, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_x, const float focal_y, const float principal_x, const float principal_y, const std::uint32_t seed, const std::uint32_t current_step, const std::uint32_t rays_per_batch, const std::uint32_t sample_limit, const std::uint8_t* const occupancy, float* const sample_coords, float* const rays, std::uint32_t* const target_pixel_indices, std::uint32_t* const numsteps, std::uint32_t* const ray_counter, std::uint32_t* const sample_counter) {
        if (rays_per_batch == 0u) return;

        ::cuda::fill_bytes(stream, ::cuda::std::span{ray_counter, 1u}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{sample_counter, 1u}, 0u);

        ::cuda::launch(stream, ::cuda::distribute<128u>(rays_per_batch), generate_training_samples_kernel, rays_per_batch, sample_limit, seed, current_step, frame_count, width, height, focal_x, focal_y, principal_x, principal_y, camera, occupancy, ray_counter, sample_counter, target_pixel_indices, rays, numsteps, sample_coords);
    }

    template <SamplingKernelShape Shape>
    void SamplingKernels<Shape>::update_occupancy_grid_from_density_grid(const ::cuda::stream_ref stream, const float* const density_grid_values, float* const density_grid_mean, std::uint32_t* const occupancy_grid_occupied_count, std::uint8_t* const occupancy) {
        ::cuda::fill_bytes(stream, ::cuda::std::span{density_grid_mean, 1u}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{occupancy_grid_occupied_count, 1u}, 0u);

        ::cuda::launch(stream, ::cuda::distribute<1024>(SamplingLayout<Shape>::nerf_grid_cells / 4u), reduce_density_grid_mean_kernel, density_grid_values, density_grid_mean);

        ::cuda::launch(stream, ::cuda::distribute<128u>(SamplingLayout<Shape>::nerf_grid_cells / 8u), build_density_grid_bitfield_kernel, density_grid_values, density_grid_mean, occupancy, occupancy_grid_occupied_count);
    }

    template <SamplingKernelShape Shape>
    std::uint32_t SamplingKernels<Shape>::prepare_density_grid_update(const ::cuda::stream_ref stream, const float* const camera, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_x, const float focal_y, const float principal_x, const float principal_y, const std::uint32_t seed, const std::uint32_t current_step, float* const sample_coords, float* const density_grid_values, float* const density_grid_scratch, std::uint32_t* const density_grid_indices, const std::uint32_t density_grid_ema_step, const bool reset_density_grid) {
        const std::uint32_t density_grid_skip = std::clamp(current_step / SamplingLayout<Shape>::density_grid_skip_interval, 1u, SamplingLayout<Shape>::density_grid_max_skip);
        if (!reset_density_grid && current_step % density_grid_skip != 0u) return 0u;

        const std::uint32_t uniform_sample_count    = current_step < SamplingLayout<Shape>::density_grid_warmup_steps ? SamplingLayout<Shape>::density_grid_warmup_samples : SamplingLayout<Shape>::density_grid_steady_uniform_samples;
        const std::uint32_t nonuniform_sample_count = current_step < SamplingLayout<Shape>::density_grid_warmup_steps ? 0u : SamplingLayout<Shape>::density_grid_steady_nonuniform_samples;
        const std::uint32_t sample_count            = uniform_sample_count + nonuniform_sample_count;

        if (reset_density_grid) {
            ::cuda::launch(stream, ::cuda::distribute<128u>(SamplingLayout<Shape>::nerf_grid_cells), seed_density_grid_visibility_kernel, density_grid_values, frame_count, width, height, focal_x, focal_y, principal_x, principal_y, camera);
        }

        ::cuda::fill_bytes(stream, ::cuda::std::span{density_grid_scratch, static_cast<std::size_t>(SamplingLayout<Shape>::nerf_grid_cells)}, 0u);
        ::cuda::launch(stream, ::cuda::distribute<128u>(uniform_sample_count), generate_density_grid_samples_kernel, uniform_sample_count, seed, density_grid_ema_step, 0u, -0.01f, density_grid_values, sample_coords, density_grid_indices);

        if (nonuniform_sample_count != 0u) {
            ::cuda::launch(stream, ::cuda::distribute<128u>(nonuniform_sample_count), generate_density_grid_samples_kernel, nonuniform_sample_count, seed, density_grid_ema_step, 1u, 0.01F, density_grid_values, sample_coords + static_cast<std::uint64_t>(uniform_sample_count) * 7u, density_grid_indices + uniform_sample_count);
        }
        return sample_count;
    }

    template <SamplingKernelShape Shape>
    void SamplingKernels<Shape>::accumulate_density_grid_update(const ::cuda::stream_ref stream, const std::uint32_t sample_count, const std::uint32_t* const density_grid_indices, const std::uint16_t* const density_output, float* const density_grid_scratch) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), splat_density_grid_samples_kernel, sample_count, density_grid_indices, reinterpret_cast<const __half*>(density_output), density_grid_scratch);
    }

    template <SamplingKernelShape Shape>
    void SamplingKernels<Shape>::commit_density_grid_update(const ::cuda::stream_ref stream, float* const density_grid_scratch, float* const density_grid_values, std::uint32_t& density_grid_ema_step) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(SamplingLayout<Shape>::nerf_grid_cells), update_density_grid_ema_kernel, density_grid_scratch, density_grid_values);
        ++density_grid_ema_step;
    }

    template <SamplingKernelShape Shape>
    std::uint32_t SamplingKernels<Shape>::sample_evaluation_batch(const ::cuda::stream_ref stream, const std::uint32_t tile_pixels, const std::uint32_t pixel_offset, const std::uint32_t width, const std::uint32_t height, const float focal_x, const float focal_y, const float principal_x, const float principal_y, const float* const camera, const std::uint32_t image_index, const std::uint8_t* const occupancy, float* const sample_coords, std::uint32_t* const numsteps, std::uint32_t* const sample_counter) {
        ::cuda::fill_bytes(stream, ::cuda::std::span{numsteps, static_cast<std::size_t>(SamplingLayout<Shape>::evaluation_tile_rays) * 2u}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{sample_counter, 1u}, 0u);
        ::cuda::launch(stream, ::cuda::distribute<128u>(tile_pixels), generate_evaluation_samples_kernel, tile_pixels, pixel_offset, width, height, focal_x, focal_y, principal_x, principal_y, camera, image_index, occupancy, sample_counter, numsteps, sample_coords);

        std::uint32_t used_samples = 0u;
        ::cuda::copy_bytes(stream, ::cuda::std::span{sample_counter, 1u}, ::cuda::std::span{&used_samples, 1u});
        stream.sync();
        if (used_samples == 0u) return 0u;
        const std::uint32_t padded_sample_count = ::cuda::round_up(used_samples, SamplingLayout<Shape>::network_batch_granularity);
        const std::uint32_t coordinate_count    = padded_sample_count * 7u;
        ::cuda::launch(stream, ::cuda::distribute<128u>(coordinate_count), pad_evaluation_rollover_coords_kernel, used_samples, padded_sample_count, sample_coords);
        return padded_sample_count;
    }

    template struct SamplingKernels<sampling_cuda_shape>;

} // namespace physica::reconstruction::instant_ngp::kernels
