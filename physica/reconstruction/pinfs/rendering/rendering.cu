#include "kernels.h"
#include <cuda/launch>
#include <cuda_runtime.h>

namespace physica::reconstruction::pinfs::kernels {
    namespace {
        __device__ bool inside_aabb(const float* positions, const float* aabb, const std::uint32_t sample) {
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const float value = positions[component + static_cast<std::size_t>(sample) * 3u];
                if (value < aabb[component] || value > aabb[3u + component]) return false;
            }
            return true;
        }

        __device__ float distance(const float* z, const float* directions, const std::uint32_t ray, const std::uint32_t sample, const std::uint32_t sample_count) {
            const std::uint32_t next          = sample + 1u < sample_count ? sample + 1u : sample;
            const std::uint32_t previous      = sample + 1u < sample_count ? sample : sample - 1u;
            const float delta                 = z[static_cast<std::size_t>(ray) * sample_count + next] - z[static_cast<std::size_t>(ray) * sample_count + previous];
            const std::size_t direction_index = static_cast<std::size_t>(ray) * sample_count * 3u;
            const float x                     = directions[direction_index];
            const float y                     = directions[direction_index + 1u];
            const float z_direction           = directions[direction_index + 2u];
            return delta * sqrtf(x * x + y * y + z_direction * z_direction);
        }

        __device__ float dynamic_alpha(const float density, const float sample_distance, const bool inside) {
            return inside ? 1.0F - expf(-density * sample_distance) : 0.0F;
        }

        __device__ float static_alpha(const float sdf, const float gradient_x, const float gradient_y, const float gradient_z, const float direction_x, const float direction_y, const float direction_z, const float sample_distance, const float inverse_deviation, const bool inside) {
            if (!inside) return 0.0F;
            const float true_cosine      = direction_x * gradient_x + direction_y * gradient_y + direction_z * gradient_z;
            const float iterative_cosine = fminf(true_cosine, 0.0F);
            const float sdf_difference   = iterative_cosine * sample_distance * 0.5F;
            const float previous_cdf     = 1.0F / (1.0F + expf(-(sdf - sdf_difference) * inverse_deviation));
            const float next_cdf         = 1.0F / (1.0F + expf(-(sdf + sdf_difference) * inverse_deviation));
            return fminf(fmaxf(1.0F - next_cdf / (previous_cdf + 1.0e-5F), 0.0F), 1.0F);
        }

        __global__ void render_dynamic_kernel(const float* z, const float* positions, const float* directions, const float* dynamic, const float* background, const float* aabb, const RenderViews output, const std::uint32_t ray_count, const std::uint32_t samples_per_ray) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            float transmittance = 1.0F;
            float accumulation{};
            float rgb[3]{};
            for (std::uint32_t sample = 0u; sample < samples_per_ray; ++sample) {
                const std::uint32_t flat          = ray * samples_per_ray + sample;
                const float alpha                 = dynamic_alpha(dynamic[static_cast<std::size_t>(flat) * 4u + 3u], distance(z, directions, ray, sample, samples_per_ray), inside_aabb(positions, aabb, flat));
                const float weight                = alpha * transmittance;
                output.weights[flat]              = weight;
                output.dynamic_alpha[flat]        = alpha;
                output.shared_transmittance[flat] = transmittance;
                accumulation += weight;
                for (std::uint32_t component = 0u; component < 3u; ++component) rgb[component] = fmaf(weight, dynamic[static_cast<std::size_t>(flat) * 4u + component], rgb[component]);
                transmittance *= 1.0F - alpha + 1.0e-10F;
            }
            output.accumulation[ray]         = accumulation;
            output.dynamic_accumulation[ray] = accumulation;
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const float value                                                  = fmaf(background[component], 1.0F - accumulation, rgb[component]);
                output.rgb[static_cast<std::size_t>(ray) * 3u + component]         = value;
                output.dynamic_rgb[static_cast<std::size_t>(ray) * 3u + component] = value;
            }
        }

        __global__ void render_hybrid_kernel(const float* z, const float* positions, const float* directions, const float* dynamic, const float* static_color, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* background, const float* aabb, const RenderViews output, const std::uint32_t ray_count, const std::uint32_t samples_per_ray) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            const std::uint32_t total_samples = ray_count * samples_per_ray;
            float shared_transmittance        = 1.0F;
            float dynamic_transmittance       = 1.0F;
            float static_transmittance        = 1.0F;
            float accumulation{};
            float dynamic_accumulation{};
            float static_accumulation{};
            float rgb[3]{};
            float dynamic_rgb[3]{};
            float static_rgb_map[3]{};
            const float direction_x = directions[static_cast<std::size_t>(ray) * samples_per_ray * 3u];
            const float direction_y = directions[static_cast<std::size_t>(ray) * samples_per_ray * 3u + 1u];
            const float direction_z = directions[static_cast<std::size_t>(ray) * samples_per_ray * 3u + 2u];
            for (std::uint32_t sample = 0u; sample < samples_per_ray; ++sample) {
                const std::uint32_t flat           = ray * samples_per_ray + sample;
                const float sample_distance        = distance(z, directions, ray, sample, samples_per_ray);
                const bool inside                  = inside_aabb(positions, aabb, flat);
                const float alpha_dynamic          = dynamic_alpha(dynamic[static_cast<std::size_t>(flat) * 4u + 3u], sample_distance, inside);
                const float alpha_static           = static_alpha(sdf[static_cast<std::size_t>(flat) * 257u], sdf_derivatives[static_cast<std::size_t>(flat) * 257u], sdf_derivatives[static_cast<std::size_t>(total_samples) * 257u + static_cast<std::size_t>(flat) * 257u], sdf_derivatives[static_cast<std::size_t>(total_samples) * 514u + static_cast<std::size_t>(flat) * 257u], direction_x, direction_y, direction_z, sample_distance, inverse_deviation[0], inside);
                const float dynamic_weight         = alpha_dynamic * shared_transmittance;
                const float static_weight          = alpha_static * shared_transmittance;
                const float dynamic_self_weight    = alpha_dynamic * dynamic_transmittance;
                const float static_self_weight     = alpha_static * static_transmittance;
                output.weights[flat]               = dynamic_weight;
                output.dynamic_alpha[flat]         = alpha_dynamic;
                output.static_alpha[flat]          = alpha_static;
                output.shared_transmittance[flat]  = shared_transmittance;
                output.dynamic_transmittance[flat] = dynamic_transmittance;
                output.static_transmittance[flat]  = static_transmittance;
                accumulation += dynamic_weight + static_weight;
                dynamic_accumulation += dynamic_self_weight;
                static_accumulation += static_self_weight;
                for (std::uint32_t component = 0u; component < 3u; ++component) {
                    rgb[component]            = fmaf(dynamic_weight, dynamic[static_cast<std::size_t>(flat) * 4u + component], fmaf(static_weight, static_color[static_cast<std::size_t>(flat) * 3u + component], rgb[component]));
                    dynamic_rgb[component]    = fmaf(dynamic_self_weight, dynamic[static_cast<std::size_t>(flat) * 4u + component], dynamic_rgb[component]);
                    static_rgb_map[component] = fmaf(static_self_weight, static_color[static_cast<std::size_t>(flat) * 3u + component], static_rgb_map[component]);
                }
                shared_transmittance *= (1.0F - alpha_dynamic) * (1.0F - alpha_static) + 1.0e-9F;
                dynamic_transmittance *= 1.0F - alpha_dynamic + 1.0e-9F;
                static_transmittance *= 1.0F - alpha_static + 1.0e-9F;
            }
            output.accumulation[ray]         = accumulation;
            output.dynamic_accumulation[ray] = dynamic_accumulation;
            output.static_accumulation[ray]  = static_accumulation;
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                output.rgb[static_cast<std::size_t>(ray) * 3u + component]         = fmaf(background[component], 1.0F - accumulation, rgb[component]);
                output.dynamic_rgb[static_cast<std::size_t>(ray) * 3u + component] = fmaf(background[component], 1.0F - dynamic_accumulation, dynamic_rgb[component]);
                output.static_rgb[static_cast<std::size_t>(ray) * 3u + component]  = fmaf(background[component], 1.0F - static_accumulation, static_rgb_map[component]);
            }
        }

        __device__ double mse_loss(const float* rgb, const float* target, float* rgb_adjoint, const std::uint32_t ray, const std::uint32_t normalization_ray_count, const float weight) {
            double result{};
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const std::size_t index = static_cast<std::size_t>(ray) * 3u + component;
                const float difference  = rgb[index] - target[index];
                result += static_cast<double>(difference) * difference * weight / static_cast<double>(normalization_ray_count * 3u);
                rgb_adjoint[index] += 2.0F * difference * weight / static_cast<float>(normalization_ray_count * 3u);
            }
            return result;
        }

        __device__ double ghost_loss(const float* rgb, const float* reference, const float accumulation, float* rgb_adjoint, float& accumulation_adjoint, const std::uint32_t ray, const std::uint32_t normalization_ray_count, const float scale, const float weight, float* reference_adjoint) {
            float squared_difference{};
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const std::size_t index = static_cast<std::size_t>(ray) * 3u + component;
                const float difference  = rgb[index] - reference[index];
                squared_difference      = fmaf(difference, difference, squared_difference);
            }
            squared_difference /= 3.0F;
            const float mask   = expf(-scale * squared_difference);
            const float common = -4.0F * scale * accumulation * accumulation * mask * mask * weight / static_cast<float>(normalization_ray_count * 3u);
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const std::size_t index = static_cast<std::size_t>(ray) * 3u + component;
                const float gradient    = common * (rgb[index] - reference[index]);
                rgb_adjoint[index] += gradient;
                if (reference_adjoint != nullptr) reference_adjoint[index] -= gradient;
            }
            accumulation_adjoint += 2.0F * accumulation * mask * mask * weight / static_cast<float>(normalization_ray_count);
            return static_cast<double>(accumulation) * accumulation * mask * mask * weight / static_cast<double>(normalization_ray_count);
        }

        __global__ void initialize_loss_kernel(const RenderViews coarse, const RenderViews fine, const RenderAdjointViews coarse_adjoints, const RenderAdjointViews fine_adjoints, const float* target, const float* background, const float* fine_perceptual_adjoints, const float* coarse_perceptual_adjoints, double* losses, const std::uint32_t ray_count, const std::uint32_t normalization_ray_count, const bool hybrid, const float temporal_fading, const float ghost_weight, const float ghost_scale, const float perceptual_loss) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            coarse_adjoints.accumulation[ray]         = 0.0F;
            coarse_adjoints.dynamic_accumulation[ray] = 0.0F;
            coarse_adjoints.static_accumulation[ray]  = 0.0F;
            fine_adjoints.accumulation[ray]           = 0.0F;
            fine_adjoints.dynamic_accumulation[ray]   = 0.0F;
            fine_adjoints.static_accumulation[ray]    = 0.0F;
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const std::size_t index            = static_cast<std::size_t>(ray) * 3u + component;
                coarse_adjoints.rgb[index]         = coarse_perceptual_adjoints[index];
                coarse_adjoints.dynamic_rgb[index] = 0.0F;
                coarse_adjoints.static_rgb[index]  = 0.0F;
                fine_adjoints.rgb[index]           = fine_perceptual_adjoints[index];
                fine_adjoints.dynamic_rgb[index]   = 0.0F;
                fine_adjoints.static_rgb[index]    = 0.0F;
            }
            const double fine_image   = mse_loss(fine.rgb, target, fine_adjoints.rgb, ray, normalization_ray_count, hybrid ? temporal_fading : 1.0F) + (hybrid ? mse_loss(fine.static_rgb, target, fine_adjoints.static_rgb, ray, normalization_ray_count, 1.0F - temporal_fading) : 0.0);
            const double coarse_image = mse_loss(coarse.rgb, target, coarse_adjoints.rgb, ray, normalization_ray_count, 1.0F);
            double ghost{};
            if (ghost_weight > 0.0F) {
                ghost += ghost_loss(fine.rgb, background, fine.accumulation[ray], fine_adjoints.rgb, fine_adjoints.accumulation[ray], ray, normalization_ray_count, ghost_scale, ghost_weight, nullptr);
                if (hybrid) {
                    ghost += ghost_loss(fine.static_rgb, background, fine.static_accumulation[ray], fine_adjoints.static_rgb, fine_adjoints.static_accumulation[ray], ray, normalization_ray_count, ghost_scale, ghost_weight * 0.1F, nullptr);
                    ghost += ghost_loss(fine.dynamic_rgb, fine.static_rgb, fine.dynamic_accumulation[ray], fine_adjoints.dynamic_rgb, fine_adjoints.dynamic_accumulation[ray], ray, normalization_ray_count, ghost_scale, ghost_weight * 0.1F, fine_adjoints.static_rgb);
                }
                ghost += ghost_loss(coarse.rgb, background, coarse.accumulation[ray], coarse_adjoints.rgb, coarse_adjoints.accumulation[ray], ray, normalization_ray_count, ghost_scale, ghost_weight, nullptr);
            }
            atomicAdd(losses + 0u, fine_image + coarse_image + ghost);
            atomicAdd(losses + 1u, fine_image);
            atomicAdd(losses + 2u, coarse_image);
            atomicAdd(losses + 4u, ghost);
            if (ray == 0u) {
                atomicAdd(losses + 0u, static_cast<double>(perceptual_loss));
                atomicAdd(losses + 3u, static_cast<double>(perceptual_loss));
            }
        }

        __global__ void regularization_loss_kernel(const float* positions, const float* dynamic, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* aabb, float* dynamic_adjoints, float* sdf_adjoints, float* sdf_gradient_adjoints, float* inverse_deviation_adjoint, double* losses, const std::uint32_t sample_count, const std::uint32_t normalization_sample_count, const float overlay_weight, const float eikonal_weight, const float deviation_weight, const std::uint32_t step) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            const float inv_s = inverse_deviation[0];
            if (overlay_weight > 0.0F && inside_aabb(positions, aabb, sample)) {
                const float density        = dynamic[static_cast<std::size_t>(sample) * 4u + 3u];
                const float sdf_value      = sdf[static_cast<std::size_t>(sample) * 257u];
                const float sigmoid        = 1.0F / (1.0F + expf(inv_s * sdf_value));
                const float static_density = inv_s * sigmoid * 0.5F;
                const float denominator    = density * density + static_density * static_density + 1.0e-8F;
                const float value          = density * static_density / denominator;
                const float common         = overlay_weight / static_cast<float>(normalization_sample_count);
                dynamic_adjoints[static_cast<std::size_t>(sample) * 4u + 3u] += common * static_density * (static_density * static_density - density * density + 1.0e-8F) / (denominator * denominator);
                const float static_gradient = common * density * (density * density - static_density * static_density + 1.0e-8F) / (denominator * denominator);
                sdf_adjoints[sample] += static_gradient * -0.5F * inv_s * inv_s * sigmoid * (1.0F - sigmoid);
                const double weighted = static_cast<double>(value) * overlay_weight / static_cast<double>(normalization_sample_count);
                atomicAdd(losses + 0u, weighted);
                atomicAdd(losses + 5u, weighted);
            }
            if (eikonal_weight > 0.0F) {
                const float gradient_x = sdf_derivatives[static_cast<std::size_t>(sample) * 257u];
                const float gradient_y = sdf_derivatives[static_cast<std::size_t>(sample_count) * 257u + static_cast<std::size_t>(sample) * 257u];
                const float gradient_z = sdf_derivatives[static_cast<std::size_t>(sample_count) * 514u + static_cast<std::size_t>(sample) * 257u];
                const float norm       = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y + gradient_z * gradient_z);
                const float difference = norm - 1.0F;
                const float mask       = inside_aabb(positions, aabb, sample) ? 1.0F : 0.0F;
                const float factor     = norm == 0.0F ? 0.0F : eikonal_weight * mask * 2.0F * difference / (norm * static_cast<float>(normalization_sample_count));
                sdf_gradient_adjoints[sample] += factor * gradient_x;
                sdf_gradient_adjoints[static_cast<std::size_t>(sample_count) + sample] += factor * gradient_y;
                sdf_gradient_adjoints[static_cast<std::size_t>(sample_count) * 2u + sample] += factor * gradient_z;
                const double weighted = static_cast<double>(difference) * difference * mask * eikonal_weight / static_cast<double>(normalization_sample_count);
                atomicAdd(losses + 0u, weighted);
                atomicAdd(losses + 6u, weighted);
            }
            if (sample == 0u && step > 20'000u && inv_s < 100.0F && deviation_weight > 0.0F) {
                const float value = deviation_weight / inv_s;
                inverse_deviation_adjoint[0] -= deviation_weight / (inv_s * inv_s);
                atomicAdd(losses + 0u, static_cast<double>(value));
                atomicAdd(losses + 7u, static_cast<double>(value));
            }
        }

        __global__ void backward_dynamic_kernel(const float* z, const float* positions, const float* directions, const float* dynamic, const float* background, const float* aabb, const RenderViews output, const RenderAdjointViews output_adjoints, float* dynamic_adjoints, const std::uint32_t ray_count, const std::uint32_t samples_per_ray) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            float suffix{};
            for (std::uint32_t sample = samples_per_ray; sample-- > 0u;) {
                const std::uint32_t flat  = ray * samples_per_ray + sample;
                const float alpha         = output.dynamic_alpha[flat];
                const float transmittance = output.shared_transmittance[flat];
                const float weight        = output.weights[flat];
                float weight_adjoint      = output_adjoints.accumulation[ray];
                for (std::uint32_t component = 0u; component < 3u; ++component) weight_adjoint = fmaf(output_adjoints.rgb[static_cast<std::size_t>(ray) * 3u + component], dynamic[static_cast<std::size_t>(flat) * 4u + component] - background[component], weight_adjoint);
                for (std::uint32_t component = 0u; component < 3u; ++component) dynamic_adjoints[static_cast<std::size_t>(flat) * 4u + component] += weight * output_adjoints.rgb[static_cast<std::size_t>(ray) * 3u + component];
                const float alpha_adjoint = transmittance * weight_adjoint - suffix / (1.0F - alpha + 1.0e-10F);
                dynamic_adjoints[static_cast<std::size_t>(flat) * 4u + 3u] += alpha_adjoint * (1.0F - alpha) * distance(z, directions, ray, sample, samples_per_ray) * (inside_aabb(positions, aabb, flat) ? 1.0F : 0.0F);
                suffix = fmaf(weight_adjoint, weight, suffix);
            }
        }

        __device__ void static_alpha_backward(const float alpha_adjoint, const float sdf_value, const float gradient_x, const float gradient_y, const float gradient_z, const float direction_x, const float direction_y, const float direction_z, const float sample_distance, const float inverse_deviation, const bool inside, float& sdf_adjoint, float& gradient_x_adjoint, float& gradient_y_adjoint, float& gradient_z_adjoint, float& inverse_deviation_adjoint) {
            if (!inside) return;
            const float true_cosine       = direction_x * gradient_x + direction_y * gradient_y + direction_z * gradient_z;
            const float iterative_cosine  = fminf(true_cosine, 0.0F);
            const float sdf_difference    = iterative_cosine * sample_distance * 0.5F;
            const float previous_argument = (sdf_value - sdf_difference) * inverse_deviation;
            const float next_argument     = (sdf_value + sdf_difference) * inverse_deviation;
            const float previous_cdf      = 1.0F / (1.0F + expf(-previous_argument));
            const float next_cdf          = 1.0F / (1.0F + expf(-next_argument));
            const float unclamped_alpha   = 1.0F - next_cdf / (previous_cdf + 1.0e-5F);
            if (unclamped_alpha <= 0.0F || unclamped_alpha >= 1.0F) return;
            const float previous_adjoint          = alpha_adjoint * next_cdf / ((previous_cdf + 1.0e-5F) * (previous_cdf + 1.0e-5F));
            const float next_adjoint              = -alpha_adjoint / (previous_cdf + 1.0e-5F);
            const float previous_argument_adjoint = previous_adjoint * previous_cdf * (1.0F - previous_cdf);
            const float next_argument_adjoint     = next_adjoint * next_cdf * (1.0F - next_cdf);
            sdf_adjoint += (previous_argument_adjoint + next_argument_adjoint) * inverse_deviation;
            const float difference_adjoint = (-previous_argument_adjoint + next_argument_adjoint) * inverse_deviation;
            if (true_cosine < 0.0F) {
                const float cosine_adjoint = difference_adjoint * sample_distance * 0.5F;
                gradient_x_adjoint += cosine_adjoint * direction_x;
                gradient_y_adjoint += cosine_adjoint * direction_y;
                gradient_z_adjoint += cosine_adjoint * direction_z;
            }
            inverse_deviation_adjoint += previous_argument_adjoint * (sdf_value - sdf_difference) + next_argument_adjoint * (sdf_value + sdf_difference);
        }

        __global__ void backward_hybrid_kernel(const float* z, const float* positions, const float* directions, const float* dynamic, const float* static_color, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* background, const float* aabb, const RenderViews output, const RenderAdjointViews output_adjoints, float* dynamic_adjoints, float* static_color_adjoints, float* sdf_adjoints, float* sdf_gradient_adjoints, float* inverse_deviation_adjoint, const std::uint32_t ray_count, const std::uint32_t samples_per_ray) {
            const std::uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray >= ray_count) return;
            const std::uint32_t total_samples = ray_count * samples_per_ray;
            const float direction_x           = directions[static_cast<std::size_t>(ray) * samples_per_ray * 3u];
            const float direction_y           = directions[static_cast<std::size_t>(ray) * samples_per_ray * 3u + 1u];
            const float direction_z           = directions[static_cast<std::size_t>(ray) * samples_per_ray * 3u + 2u];
            float shared_suffix{};
            float dynamic_suffix{};
            float static_suffix{};
            for (std::uint32_t sample = samples_per_ray; sample-- > 0u;) {
                const std::uint32_t flat          = ray * samples_per_ray + sample;
                const float alpha_dynamic         = output.dynamic_alpha[flat];
                const float alpha_static          = output.static_alpha[flat];
                const float shared                = output.shared_transmittance[flat];
                const float dynamic_self          = output.dynamic_transmittance[flat];
                const float static_self           = output.static_transmittance[flat];
                const float dynamic_weight        = alpha_dynamic * shared;
                const float static_weight         = alpha_static * shared;
                const float dynamic_self_weight   = alpha_dynamic * dynamic_self;
                const float static_self_weight    = alpha_static * static_self;
                float dynamic_weight_adjoint      = output_adjoints.accumulation[ray];
                float static_weight_adjoint       = output_adjoints.accumulation[ray];
                float dynamic_self_weight_adjoint = output_adjoints.dynamic_accumulation[ray];
                float static_self_weight_adjoint  = output_adjoints.static_accumulation[ray];
                for (std::uint32_t component = 0u; component < 3u; ++component) {
                    const std::size_t map_index = static_cast<std::size_t>(ray) * 3u + component;
                    dynamic_weight_adjoint      = fmaf(output_adjoints.rgb[map_index], dynamic[static_cast<std::size_t>(flat) * 4u + component] - background[component], dynamic_weight_adjoint);
                    static_weight_adjoint       = fmaf(output_adjoints.rgb[map_index], static_color[static_cast<std::size_t>(flat) * 3u + component] - background[component], static_weight_adjoint);
                    dynamic_self_weight_adjoint = fmaf(output_adjoints.dynamic_rgb[map_index], dynamic[static_cast<std::size_t>(flat) * 4u + component] - background[component], dynamic_self_weight_adjoint);
                    static_self_weight_adjoint  = fmaf(output_adjoints.static_rgb[map_index], static_color[static_cast<std::size_t>(flat) * 3u + component] - background[component], static_self_weight_adjoint);
                    dynamic_adjoints[static_cast<std::size_t>(flat) * 4u + component] += dynamic_weight * output_adjoints.rgb[map_index] + dynamic_self_weight * output_adjoints.dynamic_rgb[map_index];
                    static_color_adjoints[static_cast<std::size_t>(flat) * 3u + component] += static_weight * output_adjoints.rgb[map_index] + static_self_weight * output_adjoints.static_rgb[map_index];
                }
                const float beta            = (1.0F - alpha_dynamic) * (1.0F - alpha_static) + 1.0e-9F;
                float dynamic_alpha_adjoint = shared * dynamic_weight_adjoint - shared_suffix * (1.0F - alpha_static) / beta;
                float static_alpha_adjoint  = shared * static_weight_adjoint - shared_suffix * (1.0F - alpha_dynamic) / beta;
                dynamic_alpha_adjoint += dynamic_self * dynamic_self_weight_adjoint - dynamic_suffix / (1.0F - alpha_dynamic + 1.0e-9F);
                static_alpha_adjoint += static_self * static_self_weight_adjoint - static_suffix / (1.0F - alpha_static + 1.0e-9F);
                const bool inside = inside_aabb(positions, aabb, flat);
                dynamic_adjoints[static_cast<std::size_t>(flat) * 4u + 3u] += dynamic_alpha_adjoint * (1.0F - alpha_dynamic) * distance(z, directions, ray, sample, samples_per_ray) * (inside ? 1.0F : 0.0F);
                float gradient_x_adjoint{};
                float gradient_y_adjoint{};
                float gradient_z_adjoint{};
                float inv_s_adjoint{};
                static_alpha_backward(static_alpha_adjoint, sdf[static_cast<std::size_t>(flat) * 257u], sdf_derivatives[static_cast<std::size_t>(flat) * 257u], sdf_derivatives[static_cast<std::size_t>(total_samples) * 257u + static_cast<std::size_t>(flat) * 257u], sdf_derivatives[static_cast<std::size_t>(total_samples) * 514u + static_cast<std::size_t>(flat) * 257u], direction_x, direction_y, direction_z, distance(z, directions, ray, sample, samples_per_ray), inverse_deviation[0], inside, sdf_adjoints[flat], gradient_x_adjoint, gradient_y_adjoint, gradient_z_adjoint, inv_s_adjoint);
                sdf_gradient_adjoints[flat] += gradient_x_adjoint;
                sdf_gradient_adjoints[static_cast<std::size_t>(total_samples) + flat] += gradient_y_adjoint;
                sdf_gradient_adjoints[static_cast<std::size_t>(total_samples) * 2u + flat] += gradient_z_adjoint;
                atomicAdd(inverse_deviation_adjoint, inv_s_adjoint);
                shared_suffix += dynamic_weight_adjoint * dynamic_weight + static_weight_adjoint * static_weight;
                dynamic_suffix = fmaf(dynamic_self_weight_adjoint, dynamic_self_weight, dynamic_suffix);
                static_suffix  = fmaf(static_self_weight_adjoint, static_self_weight, static_suffix);
            }
        }
    } // namespace

    void render_dynamic(const ::cuda::stream_ref stream, const float* z, const float* positions, const float* directions, const float* dynamic, const float* background, const float* aabb, const RenderViews output, const std::uint32_t ray_count, const std::uint32_t samples_per_ray) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), render_dynamic_kernel, z, positions, directions, dynamic, background, aabb, output, ray_count, samples_per_ray);
    }

    void render_hybrid(const ::cuda::stream_ref stream, const float* z, const float* positions, const float* directions, const float* dynamic, const float* static_color, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* background, const float* aabb, const RenderViews output, const std::uint32_t ray_count, const std::uint32_t samples_per_ray) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), render_hybrid_kernel, z, positions, directions, dynamic, static_color, sdf, sdf_derivatives, inverse_deviation, background, aabb, output, ray_count, samples_per_ray);
    }

    void initialize_loss(const ::cuda::stream_ref stream, const RenderViews coarse, const RenderViews fine, const RenderAdjointViews coarse_adjoints, const RenderAdjointViews fine_adjoints, const float* target, const float* background, const float* fine_perceptual_adjoints, const float* coarse_perceptual_adjoints, double* losses, const std::uint32_t ray_count, const std::uint32_t normalization_ray_count, const bool hybrid, const float temporal_fading, const float ghost_weight, const float ghost_scale, const float perceptual_loss) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), initialize_loss_kernel, coarse, fine, coarse_adjoints, fine_adjoints, target, background, fine_perceptual_adjoints, coarse_perceptual_adjoints, losses, ray_count, normalization_ray_count, hybrid, temporal_fading, ghost_weight, ghost_scale, perceptual_loss);
    }

    void regularization_loss(const ::cuda::stream_ref stream, const float* positions, const float* dynamic, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* aabb, float* dynamic_adjoints, float* sdf_adjoints, float* sdf_gradient_adjoints, float* inverse_deviation_adjoint, double* losses, const std::uint32_t sample_count, const std::uint32_t normalization_sample_count, const float overlay_weight, const float eikonal_weight, const float deviation_weight, const std::uint32_t step) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), regularization_loss_kernel, positions, dynamic, sdf, sdf_derivatives, inverse_deviation, aabb, dynamic_adjoints, sdf_adjoints, sdf_gradient_adjoints, inverse_deviation_adjoint, losses, sample_count, normalization_sample_count, overlay_weight, eikonal_weight, deviation_weight, step);
    }

    void backward_dynamic(const ::cuda::stream_ref stream, const float* z, const float* positions, const float* directions, const float* dynamic, const float* background, const float* aabb, const RenderViews output, const RenderAdjointViews output_adjoints, float* dynamic_adjoints, const std::uint32_t ray_count, const std::uint32_t samples_per_ray) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), backward_dynamic_kernel, z, positions, directions, dynamic, background, aabb, output, output_adjoints, dynamic_adjoints, ray_count, samples_per_ray);
    }

    void backward_hybrid(const ::cuda::stream_ref stream, const float* z, const float* positions, const float* directions, const float* dynamic, const float* static_color, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* background, const float* aabb, const RenderViews output, const RenderAdjointViews output_adjoints, float* dynamic_adjoints, float* static_color_adjoints, float* sdf_adjoints, float* sdf_gradient_adjoints, float* inverse_deviation_adjoint, const std::uint32_t ray_count, const std::uint32_t samples_per_ray) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(ray_count), backward_hybrid_kernel, z, positions, directions, dynamic, static_color, sdf, sdf_derivatives, inverse_deviation, background, aabb, output, output_adjoints, dynamic_adjoints, static_color_adjoints, sdf_adjoints, sdf_gradient_adjoints, inverse_deviation_adjoint, ray_count, samples_per_ray);
    }
} // namespace physica::reconstruction::pinfs::kernels
