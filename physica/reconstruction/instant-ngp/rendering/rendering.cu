#include "kernels.h"
#include <math/math.h>
#include <cuda/algorithm>
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/std/cmath>
#include <cuda/std/random>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace physica::reconstruction::instant_ngp {
    inline constexpr kernels::RenderingKernelShape rendering_cuda_shape{
        .training_batch_size  = 1u << 18u,
        .network_output_width = 16u,
    };

    template <kernels::RenderingKernelShape Shape>
    struct RenderingLayout final {
        inline static constexpr std::uint32_t network_batch_size   = Shape.training_batch_size;
        inline static constexpr std::uint32_t network_output_width = Shape.network_output_width;
    };
} // namespace physica::reconstruction::instant_ngp

namespace physica::reconstruction::instant_ngp::kernels {
    namespace {
    inline constexpr std::uint32_t rendering_random_domain = 2u;

    enum class RenderingRandomSequence : std::uint32_t {
        background,
    };

    inline __device__ float sigmoid(const float value) {
        return 1.0f / (1.0f + ::cuda::std::exp(-value));
    }

    inline __device__ float rgb_activation_derivative(const float value) {
        const float rgb = sigmoid(value);
        return rgb * (1.0f - rgb);
    }

    inline __device__ Vector3<float> srgb_to_linear(const Vector3<float> value) {
        return {
            value.x <= 0.04045f ? value.x / 12.92f : ::cuda::std::pow((value.x + 0.055f) / 1.055f, 2.4f),
            value.y <= 0.04045f ? value.y / 12.92f : ::cuda::std::pow((value.y + 0.055f) / 1.055f, 2.4f),
            value.z <= 0.04045f ? value.z / 12.92f : ::cuda::std::pow((value.z + 0.055f) / 1.055f, 2.4f),
        };
    }

    inline __device__ Vector3<float> linear_to_srgb(const Vector3<float> value) {
        constexpr float inverse_gamma = 1.0f / 2.4f;
        return {
            value.x < 0.0031308f ? 12.92f * value.x : 1.055f * ::cuda::std::pow(value.x, inverse_gamma) - 0.055f,
            value.y < 0.0031308f ? 12.92f * value.y : 1.055f * ::cuda::std::pow(value.y, inverse_gamma) - 0.055f,
            value.z < 0.0031308f ? 12.92f * value.z : 1.055f * ::cuda::std::pow(value.z, inverse_gamma) - 0.055f,
        };
    }

    inline __device__ float4 read_premultiplied_linear_rgba(const std::uint32_t pixel_index, const std::uint8_t* pixels) {
        const uchar4 rgba = reinterpret_cast<const uchar4*>(pixels)[pixel_index];
        float4 result     = {
            static_cast<float>(rgba.x) * (1.0f / 255.0f),
            static_cast<float>(rgba.y) * (1.0f / 255.0f),
            static_cast<float>(rgba.z) * (1.0f / 255.0f),
            static_cast<float>(rgba.w) * (1.0f / 255.0f),
        };
        const Vector3<float> linear_rgb = srgb_to_linear({result.x, result.y, result.z});
        result.x                = linear_rgb.x * result.w;
        result.y                = linear_rgb.y * result.w;
        result.z                = linear_rgb.z * result.w;
        return result;
    }

    __global__ void accumulate_evaluation_loss_kernel(const std::uint32_t tile_pixels, const std::uint32_t pixel_offset, const std::uint32_t evaluation_image_index, const std::uint32_t width, const std::uint32_t height, const std::uint8_t* __restrict__ evaluation_pixels, const std::uint32_t* __restrict__ numsteps_in, const float* __restrict__ coords_in, const __half* __restrict__ network_output, double* __restrict__ evaluation_loss_sum) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        double squared_error  = 0.0;

        if (i < tile_pixels) {
            const std::uint32_t global_pixel = pixel_offset + i;
            const std::uint32_t pixel_x      = global_pixel % width;
            const std::uint32_t pixel_y      = global_pixel / width;
            const std::uint32_t numsteps     = numsteps_in[i * 2u + 0u];
            const std::uint32_t base         = numsteps_in[i * 2u + 1u];
            float transmittance              = 1.0f;
            Vector3<float> rgb_ray                   = {};

            const float* coord   = coords_in + static_cast<std::uint64_t>(base) * 7u;
            const __half* output = network_output + static_cast<std::uint64_t>(base) * RenderingLayout<rendering_cuda_shape>::network_output_width;

            for (std::uint32_t j = 0u; j < numsteps; ++j) {
                const float rgb_x   = sigmoid(__half2float(output[0u]));
                const float rgb_y   = sigmoid(__half2float(output[1u]));
                const float rgb_z   = sigmoid(__half2float(output[2u]));
                const float density = ::cuda::std::exp(__half2float(output[3u]));
                const float alpha   = 1.0f - __expf(-density * coord[3u]);
                const float weight  = alpha * transmittance;
                rgb_ray.x += weight * rgb_x;
                rgb_ray.y += weight * rgb_y;
                rgb_ray.z += weight * rgb_z;
                transmittance *= 1.0f - alpha;
                if (transmittance < 1e-4F) break;

                coord += 7u;
                output += RenderingLayout<rendering_cuda_shape>::network_output_width;
            }

            const std::uint32_t target_pixel_index = static_cast<std::uint32_t>(pixel_x + static_cast<std::uint64_t>(pixel_y) * width + static_cast<std::uint64_t>(evaluation_image_index) * width * height);
            const float4 texel                     = read_premultiplied_linear_rgba(target_pixel_index, evaluation_pixels);
            const Vector3<float> rgb_target                = linear_to_srgb({texel.x, texel.y, texel.z});
            const float prediction_r               = ::cuda::std::min(::cuda::std::max(rgb_ray.x, 0.0f), 1.0f);
            const float prediction_g               = ::cuda::std::min(::cuda::std::max(rgb_ray.y, 0.0f), 1.0f);
            const float prediction_b               = ::cuda::std::min(::cuda::std::max(rgb_ray.z, 0.0f), 1.0f);
            const float target_r                   = ::cuda::std::min(::cuda::std::max(rgb_target.x, 0.0f), 1.0f);
            const float target_g                   = ::cuda::std::min(::cuda::std::max(rgb_target.y, 0.0f), 1.0f);
            const float target_b                   = ::cuda::std::min(::cuda::std::max(rgb_target.z, 0.0f), 1.0f);
            const double diff_r                    = static_cast<double>(prediction_r) - static_cast<double>(target_r);
            const double diff_g                    = static_cast<double>(prediction_g) - static_cast<double>(target_g);
            const double diff_b                    = static_cast<double>(prediction_b) - static_cast<double>(target_b);
            squared_error                          = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
        }

        __shared__ double sums[128u];
        sums[threadIdx.x] = squared_error;
        __syncthreads();

        for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0u) atomicAdd(evaluation_loss_sum, sums[0]);
    }

    __global__ void compute_training_loss_and_compact_kernel(const std::uint32_t rays_per_batch, const std::uint32_t seed, const std::uint32_t current_step, const std::uint32_t* __restrict__ ray_counter, const std::uint8_t* __restrict__ pixels, const __half* __restrict__ network_output, std::uint32_t* __restrict__ requested_compacted_sample_counter, std::uint32_t* __restrict__ used_compacted_sample_counter, const std::uint32_t* __restrict__ target_pixel_indices_in, const float* __restrict__ rays_in, std::uint32_t* __restrict__ numsteps_in, const float* __restrict__ coords_in, float* __restrict__ coords_out, __half* __restrict__ dloss_doutput, double* __restrict__ loss_sum) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        double ray_loss       = 0.0;

        if (i < *ray_counter) {
            const std::uint32_t numsteps     = numsteps_in[i * 2u + 0u];
            const std::uint32_t base         = numsteps_in[i * 2u + 1u];
            const float* coord_in            = coords_in + static_cast<std::uint64_t>(base) * 7u;
            const __half* output             = network_output + static_cast<std::uint64_t>(base) * RenderingLayout<rendering_cuda_shape>::network_output_width;
            float transmittance              = 1.0f;
            Vector3<float> rgb_ray                   = {};
            std::uint32_t compacted_numsteps = 0u;
            const float* ray                 = rays_in + static_cast<std::uint64_t>(i) * 3u;
            const Vector3<float> ray_origin          = {ray[0], ray[1], ray[2]};

            for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
                if (transmittance < 1e-4F) break;
                const float rgb_x   = sigmoid(__half2float(output[0u]));
                const float rgb_y   = sigmoid(__half2float(output[1u]));
                const float rgb_z   = sigmoid(__half2float(output[2u]));
                const float density = ::cuda::std::exp(__half2float(output[3u]));
                const float alpha   = 1.0f - __expf(-density * coord_in[3u]);
                const float weight  = alpha * transmittance;
                rgb_ray.x += weight * rgb_x;
                rgb_ray.y += weight * rgb_y;
                rgb_ray.z += weight * rgb_z;
                transmittance *= 1.0f - alpha;
                output += RenderingLayout<rendering_cuda_shape>::network_output_width;
                coord_in += 7u;
            }

            const std::uint32_t target_pixel_index = target_pixel_indices_in[i];
            ::cuda::std::philox4x32 background_random{seed};
            background_random.set_counter({rendering_random_domain, ::cuda::std::to_underlying(RenderingRandomSequence::background), current_step, target_pixel_index});
            ::cuda::std::uniform_real_distribution<float> unit_distribution;
            const Vector3<float> background_color  = {unit_distribution(background_random), unit_distribution(background_random), unit_distribution(background_random)};
            const float4 texel             = read_premultiplied_linear_rgba(target_pixel_index, pixels);
            const Vector3<float> background_linear = srgb_to_linear(background_color);
            const Vector3<float> rgb_target        = linear_to_srgb({texel.x + (1.0f - texel.w) * background_linear.x, texel.y + (1.0f - texel.w) * background_linear.y, texel.z + (1.0f - texel.w) * background_linear.z});

            if (compacted_numsteps == numsteps) {
                rgb_ray.x += transmittance * background_color.x;
                rgb_ray.y += transmittance * background_color.y;
                rgb_ray.z += transmittance * background_color.z;
            }

            output -= static_cast<std::uint64_t>(compacted_numsteps) * RenderingLayout<rendering_cuda_shape>::network_output_width;
            coord_in -= static_cast<std::uint64_t>(compacted_numsteps) * 7u;
            const std::uint32_t compacted_base  = atomicAdd(requested_compacted_sample_counter, compacted_numsteps);
            const std::uint32_t remaining_slots = compacted_base < RenderingLayout<rendering_cuda_shape>::network_batch_size ? RenderingLayout<rendering_cuda_shape>::network_batch_size - compacted_base : 0u;
            compacted_numsteps                  = ::cuda::std::min(compacted_numsteps, remaining_slots);
            numsteps_in[i * 2u + 0u]            = compacted_numsteps;
            numsteps_in[i * 2u + 1u]            = compacted_base;

            if (compacted_numsteps != 0u) {
                atomicAdd(used_compacted_sample_counter, compacted_numsteps);
                coords_out += static_cast<std::uint64_t>(compacted_base) * 7u;
                dloss_doutput += static_cast<std::uint64_t>(compacted_base) * RenderingLayout<rendering_cuda_shape>::network_output_width;

                const Vector3<float> difference = {rgb_ray.x - rgb_target.x, rgb_ray.y - rgb_target.y, rgb_ray.z - rgb_target.z};
                const Vector3<float> gradient   = {2.0f * difference.x, 2.0f * difference.y, 2.0f * difference.z};
                ray_loss                = static_cast<double>(difference.x * difference.x + difference.y * difference.y + difference.z * difference.z) / (3.0 * static_cast<double>(rays_per_batch));
                const float scaled_loss = 128.0F / static_cast<float>(rays_per_batch);
                Vector3<float> rgb_ray2         = {};
                transmittance           = 1.0f;

                for (std::uint32_t j = 0u; j < compacted_numsteps; ++j) {
                    float* coord_out   = coords_out + static_cast<std::uint64_t>(j) * 7u;
                    const float* coord = coord_in + static_cast<std::uint64_t>(j) * 7u;
                    for (std::uint32_t k = 0u; k < 7u; ++k) coord_out[k] = coord[k];

                    const Vector3<float> pos        = {coord[0], coord[1], coord[2]};
                    const float depth       = ::cuda::std::hypot(pos.x - ray_origin.x, pos.y - ray_origin.y, pos.z - ray_origin.z);
                    const float dt          = coord[3u];
                    const float mlp_rgb_x   = __half2float(output[0u]);
                    const float mlp_rgb_y   = __half2float(output[1u]);
                    const float mlp_rgb_z   = __half2float(output[2u]);
                    const float mlp_density = __half2float(output[3u]);
                    const Vector3<float> rgb        = {sigmoid(mlp_rgb_x), sigmoid(mlp_rgb_y), sigmoid(mlp_rgb_z)};
                    const float density     = ::cuda::std::exp(mlp_density);
                    const float alpha       = 1.0f - __expf(-density * dt);
                    const float weight      = alpha * transmittance;
                    rgb_ray2.x += weight * rgb.x;
                    rgb_ray2.y += weight * rgb.y;
                    rgb_ray2.z += weight * rgb.z;
                    transmittance *= 1.0f - alpha;

                    const Vector3<float> suffix        = {rgb_ray.x - rgb_ray2.x, rgb_ray.y - rgb_ray2.y, rgb_ray.z - rgb_ray2.z};
                    const Vector3<float> dloss_by_drgb = {weight * gradient.x, weight * gradient.y, weight * gradient.z};
                    dloss_doutput[0u]          = __float2half(scaled_loss * (dloss_by_drgb.x * rgb_activation_derivative(mlp_rgb_x)));
                    dloss_doutput[1u]          = __float2half(scaled_loss * (dloss_by_drgb.y * rgb_activation_derivative(mlp_rgb_y)));
                    dloss_doutput[2u]          = __float2half(scaled_loss * (dloss_by_drgb.z * rgb_activation_derivative(mlp_rgb_z)));

                    const float density_derivative = ::cuda::std::exp(::cuda::std::clamp(mlp_density, static_cast<float>(-15.0F), static_cast<float>(15.0F)));
                    const float dloss_by_dmlp      = density_derivative * (dt * (gradient.x * (transmittance * rgb.x - suffix.x) + gradient.y * (transmittance * rgb.y - suffix.y) + gradient.z * (transmittance * rgb.z - suffix.z)));
                    dloss_doutput[3u]              = __float2half(scaled_loss * dloss_by_dmlp + (mlp_density > -10.0F && depth < 0.1F ? 1e-4F : 0.0f));
                    dloss_doutput += RenderingLayout<rendering_cuda_shape>::network_output_width;
                    output += RenderingLayout<rendering_cuda_shape>::network_output_width;
                }
            }
        }

        __shared__ double block_loss[128u];
        block_loss[threadIdx.x] = ray_loss;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) block_loss[threadIdx.x] += block_loss[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0u) atomicAdd(loss_sum, block_loss[0]);
    }

    __global__ void pad_rollover_coords_kernel(const std::uint32_t* __restrict__ input_count, float* __restrict__ inout) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t n = *input_count;
        if (i < n * 7u || i >= RenderingLayout<rendering_cuda_shape>::network_batch_size * 7u || n == 0u) return;
        inout[i] = inout[i % (n * 7u)];
    }

    __global__ void pad_rollover_network_output_gradients_kernel(const std::uint32_t* __restrict__ input_count, __half* __restrict__ inout) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t n = *input_count;
        if (i < n * RenderingLayout<rendering_cuda_shape>::network_output_width || i >= RenderingLayout<rendering_cuda_shape>::network_batch_size * RenderingLayout<rendering_cuda_shape>::network_output_width || n == 0u) return;
        inout[i] = __float2half(__half2float(inout[i % (n * RenderingLayout<rendering_cuda_shape>::network_output_width)]) * static_cast<float>(n) / static_cast<float>(RenderingLayout<rendering_cuda_shape>::network_batch_size));
    }

    } // namespace
} // namespace physica::reconstruction::instant_ngp::kernels

namespace physica::reconstruction::instant_ngp::kernels {
    template <RenderingKernelShape Shape>
    void RenderingKernels<Shape>::compute_training_loss_and_compact_samples(const ::cuda::stream_ref stream, const std::uint32_t rays_per_batch, const std::uint32_t seed, const std::uint32_t current_step, const std::uint32_t* const ray_counter, const std::uint8_t* const pixels, const std::uint16_t* const network_output, std::uint32_t* const requested_compacted_sample_counter, std::uint32_t* const used_compacted_sample_counter, const std::uint32_t* const target_pixel_indices, const float* const rays, std::uint32_t* const numsteps, const float* const sample_coords, float* const compacted_sample_coords, std::uint16_t* const network_output_gradients, double* const loss_sum) {
        if (rays_per_batch == 0u) return;

        ::cuda::fill_bytes(stream, ::cuda::std::span{requested_compacted_sample_counter, 1u}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{used_compacted_sample_counter, 1u}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span{loss_sum, 1u}, 0u);

        const std::uint32_t blocks = ::cuda::ceil_div(rays_per_batch, 128u);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims<128u>())), compute_training_loss_and_compact_kernel, rays_per_batch, seed, current_step, ray_counter, pixels, reinterpret_cast<const __half*>(network_output), requested_compacted_sample_counter, used_compacted_sample_counter, target_pixel_indices, rays, numsteps, sample_coords, compacted_sample_coords, reinterpret_cast<__half*>(network_output_gradients), loss_sum);
    }

    template <RenderingKernelShape Shape>
    void RenderingKernels<Shape>::pad_compacted_training_batch(const ::cuda::stream_ref stream, const std::uint32_t* const compacted_sample_counter, float* const compacted_sample_coords, std::uint16_t* const network_output_gradients) {

        constexpr std::uint32_t gradient_elements = RenderingLayout<Shape>::network_batch_size * RenderingLayout<Shape>::network_output_width;
        ::cuda::launch(stream, ::cuda::distribute<128u>(gradient_elements), pad_rollover_network_output_gradients_kernel, compacted_sample_counter, reinterpret_cast<__half*>(network_output_gradients));

        constexpr std::uint32_t coord_elements = RenderingLayout<Shape>::network_batch_size * 7u;
        ::cuda::launch(stream, ::cuda::distribute<128u>(coord_elements), pad_rollover_coords_kernel, compacted_sample_counter, compacted_sample_coords);
    }

    template <RenderingKernelShape Shape>
    void RenderingKernels<Shape>::read_training_statistics(const ::cuda::stream_ref stream, const std::uint32_t* const ray_counter, const std::uint32_t* const generated_sample_counter, const std::uint32_t* const requested_compacted_sample_counter, const std::uint32_t* const used_compacted_sample_counter, const double* const loss_sum, std::uint32_t& ray_count, std::uint32_t& generated_sample_count, std::uint32_t& requested_compacted_sample_count, std::uint32_t& used_compacted_sample_count, double& output_loss_sum) {
        ::cuda::copy_bytes(stream, ::cuda::std::span{ray_counter, 1u}, ::cuda::std::span{&ray_count, 1u});
        ::cuda::copy_bytes(stream, ::cuda::std::span{generated_sample_counter, 1u}, ::cuda::std::span{&generated_sample_count, 1u});
        ::cuda::copy_bytes(stream, ::cuda::std::span{requested_compacted_sample_counter, 1u}, ::cuda::std::span{&requested_compacted_sample_count, 1u});
        ::cuda::copy_bytes(stream, ::cuda::std::span{used_compacted_sample_counter, 1u}, ::cuda::std::span{&used_compacted_sample_count, 1u});
        ::cuda::copy_bytes(stream, ::cuda::std::span{loss_sum, 1u}, ::cuda::std::span{&output_loss_sum, 1u});
        stream.sync();
    }

    template <RenderingKernelShape Shape>
    void RenderingKernels<Shape>::begin_evaluation(const ::cuda::stream_ref stream, double* const loss_sum) {
        ::cuda::fill_bytes(stream, ::cuda::std::span{loss_sum, 1u}, 0u);
    }

    template <RenderingKernelShape Shape>
    void RenderingKernels<Shape>::accumulate_evaluation_loss(const ::cuda::stream_ref stream, const std::uint32_t tile_pixels, const std::uint32_t pixel_offset, const std::uint32_t image_index, const std::uint32_t width, const std::uint32_t height, const std::uint8_t* const pixels, const std::uint32_t* const numsteps, const float* const sample_coords, const std::uint16_t* const network_output, double* const loss_sum) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(tile_pixels), accumulate_evaluation_loss_kernel, tile_pixels, pixel_offset, image_index, width, height, pixels, numsteps, sample_coords, reinterpret_cast<const __half*>(network_output), loss_sum);
    }

    template <RenderingKernelShape Shape>
    double RenderingKernels<Shape>::end_evaluation(const ::cuda::stream_ref stream, const double* const loss_sum) {
        double result = 0.0;
        ::cuda::copy_bytes(stream, ::cuda::std::span{loss_sum, 1u}, ::cuda::std::span{&result, 1u});
        stream.sync();
        return result;
    }

    template struct RenderingKernels<rendering_cuda_shape>;

} // namespace physica::reconstruction::instant_ngp::kernels
