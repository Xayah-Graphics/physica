#include "kernels.h"
#include <cuda/launch>
#include <cuda/std/random>
#include <cuda_runtime.h>

namespace physica::reconstruction::pinfs::kernels {
    namespace {
        inline constexpr std::uint32_t sampling_random_domain = 2u;

        enum class SamplingRandomSequence : std::uint32_t {
            coarse,
            warp,
            pdf,
        };

        __device__ float random_uniform(const std::uint32_t seed, const SamplingRandomSequence sequence, const std::uint32_t step, const std::uint32_t index) {
            ::cuda::std::philox4x32 random{seed};
            random.set_counter({sampling_random_domain, static_cast<std::uint32_t>(sequence), step, index});
            return static_cast<float>(random() >> 8u) * 0x1.0p-24F;
        }

        __global__ void sample_coarse_kernel(const Ray3<float>* rays, float* z, Vector3<float>* positions, float* dynamic_points, Vector3<float>* directions, const std::uint32_t ray_count, const std::uint32_t sample_count, const float near_distance, const float far_distance, const float time, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t random_offset, const bool perturb) {
            const std::uint32_t flat_sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (flat_sample >= ray_count * sample_count) return;
            const std::uint32_t ray    = flat_sample / sample_count;
            const std::uint32_t sample = flat_sample - ray * sample_count;
            const float nominal        = near_distance + (far_distance - near_distance) * static_cast<float>(sample) / static_cast<float>(sample_count - 1u);
            float sample_z             = nominal;
            if (perturb) {
                const float previous = sample == 0u ? near_distance : near_distance + (far_distance - near_distance) * static_cast<float>(sample - 1u) / static_cast<float>(sample_count - 1u);
                const float next     = sample + 1u == sample_count ? far_distance : near_distance + (far_distance - near_distance) * static_cast<float>(sample + 1u) / static_cast<float>(sample_count - 1u);
                const float lower    = sample == 0u ? nominal : 0.5F * (previous + nominal);
                const float upper    = sample + 1u == sample_count ? nominal : 0.5F * (nominal + next);
                sample_z             = lower + (upper - lower) * random_uniform(seed, SamplingRandomSequence::coarse, step, random_offset + flat_sample);
            }
            z[flat_sample] = sample_z;
            const Ray3<float> source_ray = rays[ray];
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const float position                                                   = fmaf(source_ray.direction[component], sample_z, source_ray.origin[component]);
                positions[flat_sample][component]                                      = position;
                dynamic_points[component + static_cast<std::size_t>(flat_sample) * 4u] = position;
                directions[flat_sample][component]                                     = source_ray.direction[component];
            }
            dynamic_points[3u + static_cast<std::size_t>(flat_sample) * 4u] = time;
        }

        __global__ void warp_points_kernel(float* dynamic_points, const float* velocity, const std::uint32_t sample_count, const float amount, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t random_offset) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            const float delta = (random_uniform(seed, SamplingRandomSequence::warp, step, random_offset + sample) * 6.0F - 3.0F) * amount;
            for (std::uint32_t component = 0u; component < 3u; ++component) dynamic_points[component + static_cast<std::size_t>(sample) * 4u] = fmaf(velocity[component + static_cast<std::size_t>(sample) * 3u], delta, dynamic_points[component + static_cast<std::size_t>(sample) * 4u]);
            dynamic_points[3u + static_cast<std::size_t>(sample) * 4u] += delta;
        }

        __device__ float midpoint_bin(const float* z, const std::uint32_t ray, const std::uint32_t z_count, const std::uint32_t index) {
            return 0.5F * (z[static_cast<std::size_t>(ray) * z_count + index] + z[static_cast<std::size_t>(ray) * z_count + index + 1u]);
        }

        __global__ void sample_pdf_kernel(const float* z, const float* weights, float* output, const std::uint32_t ray_count, const std::uint32_t z_count, const std::uint32_t weight_offset, const std::uint32_t weight_count, const std::uint32_t output_count, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t ray_offset, const bool deterministic) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            float weight_sum{};
            for (std::uint32_t index = 0u; index < weight_count; ++index) weight_sum += weights[static_cast<std::size_t>(ray) * z_count + weight_offset + index] + 1.0e-5F;
            for (std::uint32_t output_index = 0u; output_index < output_count; ++output_index) {
                const float u = deterministic ? static_cast<float>(output_index) / static_cast<float>(output_count - 1u) : random_uniform(seed, SamplingRandomSequence::pdf, step, (ray_offset + ray) * output_count + output_index);
                float previous_cdf{};
                float current_cdf{};
                std::uint32_t interval{};
                for (; interval < weight_count; ++interval) {
                    current_cdf += (weights[static_cast<std::size_t>(ray) * z_count + weight_offset + interval] + 1.0e-5F) / weight_sum;
                    if (u <= current_cdf) break;
                    previous_cdf = current_cdf;
                }
                interval                                                            = interval < weight_count ? interval : weight_count - 1u;
                const float lower                                                   = midpoint_bin(z, ray, z_count, interval);
                const float upper                                                   = midpoint_bin(z, ray, z_count, interval + 1u);
                const float denominator                                             = current_cdf - previous_cdf < 1.0e-5F ? 1.0F : current_cdf - previous_cdf;
                output[static_cast<std::size_t>(ray) * output_count + output_index] = lower + (upper - lower) * (u - previous_cdf) / denominator;
            }
        }

        __global__ void sort_samples_kernel(float* samples, const std::uint32_t ray_count, const std::uint32_t sample_count) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            float* values = samples + static_cast<std::size_t>(ray) * sample_count;
            for (std::uint32_t index = 1u; index < sample_count; ++index) {
                const float value       = values[index];
                std::uint32_t insertion = index;
                while (insertion > 0u && value < values[insertion - 1u]) {
                    values[insertion] = values[insertion - 1u];
                    --insertion;
                }
                values[insertion] = value;
            }
        }

        __global__ void extract_sdf_kernel(const float* sdf_output, float* sdf, const std::uint32_t sample_count) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample < sample_count) sdf[sample] = sdf_output[static_cast<std::size_t>(sample) * 257u];
        }

        __device__ void neus_interval(const float* z, const float* sdf, const std::uint32_t base, const std::uint32_t interval, const float inverse_deviation, float& alpha) {
            const float previous_z        = z[base + interval];
            const float next_z            = z[base + interval + 1u];
            const float previous_sdf      = sdf[base + interval];
            const float next_sdf          = sdf[base + interval + 1u];
            const float cosine            = (next_sdf - previous_sdf) / (next_z - previous_z + 1.0e-5F);
            const float previous_cosine   = interval == 0u ? 0.0F : (previous_sdf - sdf[base + interval - 1u]) / (previous_z - z[base + interval - 1u] + 1.0e-5F);
            const float clamped_cosine    = fminf(fmaxf(fminf(previous_cosine, cosine), -1.0e3F), 0.0F);
            const float middle_sdf        = 0.5F * (previous_sdf + next_sdf);
            const float distance          = next_z - previous_z;
            const float previous_estimate = middle_sdf - clamped_cosine * distance * 0.5F;
            const float next_estimate     = middle_sdf + clamped_cosine * distance * 0.5F;
            const float previous_cdf      = 1.0F / (1.0F + expf(-previous_estimate * inverse_deviation));
            const float next_cdf          = 1.0F / (1.0F + expf(-next_estimate * inverse_deviation));
            alpha                         = (previous_cdf - next_cdf + 1.0e-5F) / (previous_cdf + 1.0e-5F);
        }

        __global__ void neus_pdf_kernel(const float* z, const float* sdf, float* output, const std::uint32_t ray_count, const std::uint32_t sample_count, const std::uint32_t output_count, const float inverse_deviation) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            const std::uint32_t base = ray * sample_count;
            float weight_sum{};
            float transmittance = 1.0F;
            for (std::uint32_t interval = 0u; interval + 1u < sample_count; ++interval) {
                float alpha{};
                neus_interval(z, sdf, base, interval, inverse_deviation, alpha);
                weight_sum += alpha * transmittance + 1.0e-5F;
                transmittance *= 1.0F - alpha + 1.0e-7F;
            }
            for (std::uint32_t output_index = 0u; output_index < output_count; ++output_index) {
                const float u = static_cast<float>(output_index) / static_cast<float>(output_count - 1u);
                float previous_cdf{};
                float current_cdf{};
                float current_transmittance = 1.0F;
                std::uint32_t interval{};
                for (; interval + 1u < sample_count; ++interval) {
                    float alpha{};
                    neus_interval(z, sdf, base, interval, inverse_deviation, alpha);
                    current_cdf += (alpha * current_transmittance + 1.0e-5F) / weight_sum;
                    if (u <= current_cdf) break;
                    previous_cdf = current_cdf;
                    current_transmittance *= 1.0F - alpha + 1.0e-7F;
                }
                interval                                                            = interval + 1u < sample_count ? interval : sample_count - 2u;
                const float denominator                                             = current_cdf - previous_cdf < 1.0e-5F ? 1.0F : current_cdf - previous_cdf;
                output[static_cast<std::size_t>(ray) * output_count + output_index] = z[base + interval] + (z[base + interval + 1u] - z[base + interval]) * (u - previous_cdf) / denominator;
            }
        }

        __global__ void merge_samples_kernel(const float* first_z, const float* first_sdf, const std::uint32_t first_count, const float* second_z, const float* second_sdf, const std::uint32_t second_count, float* output_z, float* output_sdf, const std::uint32_t ray_count) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            std::uint32_t first{};
            std::uint32_t second{};
            const std::uint32_t output_count = first_count + second_count;
            for (std::uint32_t output = 0u; output < output_count; ++output) {
                const bool take_first = second == second_count || first < first_count && first_z[static_cast<std::size_t>(ray) * first_count + first] <= second_z[static_cast<std::size_t>(ray) * second_count + second];
                if (take_first) {
                    output_z[static_cast<std::size_t>(ray) * output_count + output] = first_z[static_cast<std::size_t>(ray) * first_count + first];
                    if (output_sdf != nullptr) output_sdf[static_cast<std::size_t>(ray) * output_count + output] = first_sdf[static_cast<std::size_t>(ray) * first_count + first];
                    ++first;
                } else {
                    output_z[static_cast<std::size_t>(ray) * output_count + output] = second_z[static_cast<std::size_t>(ray) * second_count + second];
                    if (output_sdf != nullptr) output_sdf[static_cast<std::size_t>(ray) * output_count + output] = second_sdf[static_cast<std::size_t>(ray) * second_count + second];
                    ++second;
                }
            }
        }

        __global__ void positions_from_z_kernel(const Ray3<float>* rays, const float* z, Vector3<float>* positions, const std::uint32_t ray_count, const std::uint32_t sample_count) {
            const std::uint32_t flat_sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (flat_sample >= ray_count * sample_count) return;
            const std::uint32_t ray = flat_sample / sample_count;
            for (std::uint32_t component = 0u; component < 3u; ++component) positions[flat_sample][component] = fmaf(rays[ray].direction[component], z[flat_sample], rays[ray].origin[component]);
        }

        __global__ void samples_from_z_kernel(const Ray3<float>* rays, const float* z, Vector3<float>* positions, float* dynamic_points, Vector3<float>* directions, const std::uint32_t ray_count, const std::uint32_t sample_count, const float time) {
            const std::uint32_t flat_sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (flat_sample >= ray_count * sample_count) return;
            const std::uint32_t ray = flat_sample / sample_count;
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const float position                                                   = fmaf(rays[ray].direction[component], z[flat_sample], rays[ray].origin[component]);
                positions[flat_sample][component]                                      = position;
                dynamic_points[component + static_cast<std::size_t>(flat_sample) * 4u] = position;
                directions[flat_sample][component]                                     = rays[ray].direction[component];
            }
            dynamic_points[3u + static_cast<std::size_t>(flat_sample) * 4u] = time;
        }
    } // namespace

    void sample_coarse(const ::cuda::stream_ref stream, const Ray3<float>* rays, float* z, Vector3<float>* positions, float* dynamic_points, Vector3<float>* directions, const std::uint32_t ray_count, const std::uint32_t sample_count, const float near_distance, const float far_distance, const float time, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t random_offset, const bool perturb) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count * sample_count), sample_coarse_kernel, rays, z, positions, dynamic_points, directions, ray_count, sample_count, near_distance, far_distance, time, seed, step, random_offset, perturb);
    }

    void warp_points(const ::cuda::stream_ref stream, float* dynamic_points, const float* velocity, const std::uint32_t sample_count, const float amount, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t random_offset) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), warp_points_kernel, dynamic_points, velocity, sample_count, amount, seed, step, random_offset);
    }

    void sample_pdf(const ::cuda::stream_ref stream, const float* z, const float* weights, float* output, const std::uint32_t ray_count, const std::uint32_t z_count, const std::uint32_t weight_offset, const std::uint32_t weight_count, const std::uint32_t output_count, const std::uint32_t seed, const std::uint32_t step, const std::uint32_t ray_offset, const bool deterministic) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), sample_pdf_kernel, z, weights, output, ray_count, z_count, weight_offset, weight_count, output_count, seed, step, ray_offset, deterministic);
    }

    void sort_samples(const ::cuda::stream_ref stream, float* samples, const std::uint32_t ray_count, const std::uint32_t sample_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), sort_samples_kernel, samples, ray_count, sample_count);
    }

    void extract_sdf(const ::cuda::stream_ref stream, const float* sdf_output, float* sdf, const std::uint32_t sample_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), extract_sdf_kernel, sdf_output, sdf, sample_count);
    }

    void neus_pdf(const ::cuda::stream_ref stream, const float* z, const float* sdf, float* output, const std::uint32_t ray_count, const std::uint32_t sample_count, const std::uint32_t output_count, const float inverse_deviation) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), neus_pdf_kernel, z, sdf, output, ray_count, sample_count, output_count, inverse_deviation);
    }

    void merge_samples(const ::cuda::stream_ref stream, const float* first_z, const float* first_sdf, const std::uint32_t first_count, const float* second_z, const float* second_sdf, const std::uint32_t second_count, float* output_z, float* output_sdf, const std::uint32_t ray_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), merge_samples_kernel, first_z, first_sdf, first_count, second_z, second_sdf, second_count, output_z, output_sdf, ray_count);
    }

    void positions_from_z(const ::cuda::stream_ref stream, const Ray3<float>* rays, const float* z, Vector3<float>* positions, const std::uint32_t ray_count, const std::uint32_t sample_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count * sample_count), positions_from_z_kernel, rays, z, positions, ray_count, sample_count);
    }

    void samples_from_z(const ::cuda::stream_ref stream, const Ray3<float>* rays, const float* z, Vector3<float>* positions, float* dynamic_points, Vector3<float>* directions, const std::uint32_t ray_count, const std::uint32_t sample_count, const float time) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count * sample_count), samples_from_z_kernel, rays, z, positions, dynamic_points, directions, ray_count, sample_count, time);
    }
} // namespace physica::reconstruction::pinfs::kernels
