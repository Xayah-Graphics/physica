#include "transformer-kernels.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <physica/cuda.h>

namespace physica::neural::kernels {
    namespace {
        constexpr std::uint32_t thread_count               = 256u;
        constexpr std::uint32_t attention_head_width       = 32u;
        constexpr std::uint32_t attention_tile_width       = 32u;
        constexpr std::uint32_t attention_group_width      = 8u;
        constexpr std::uint32_t attention_query_tile_width = thread_count / attention_group_width;

        __device__ __forceinline__ float2 load_pair(const float* const values, const std::size_t offset) {
            return *reinterpret_cast<const float2*>(values + offset);
        }

        __device__ __forceinline__ void store_pair(float* const values, const std::size_t offset, const float2 value) {
            *reinterpret_cast<float2*>(values + offset) = value;
        }

        __device__ __forceinline__ float attention_group_sum(float value) {
            for (std::uint32_t offset = attention_group_width / 2u; offset != 0u; offset /= 2u) value += __shfl_down_sync(0xffffffffu, value, offset, attention_group_width);
            return __shfl_sync(0xffffffffu, value, 0u, attention_group_width);
        }

        __global__ void adaln_forward_kernel(const float* const input, const float* const modulation, float* const output, float* const means, float* const inverse_standard_deviations, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group) {
            __shared__ float reduction[thread_count];
            const std::uint32_t row     = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            const std::uint32_t sample  = row / sequence;
            float value{};
            if (feature < width) value = input[static_cast<std::size_t>(row) * width + feature];
            reduction[feature] = value;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (feature < stride) reduction[feature] += reduction[feature + stride];
                __syncthreads();
            }
            const float mean = reduction[0] / static_cast<float>(width);
            __syncthreads();
            reduction[feature] = feature < width ? (value - mean) * (value - mean) : 0.0F;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (feature < stride) reduction[feature] += reduction[feature + stride];
                __syncthreads();
            }
            const float inverse_standard_deviation = rsqrtf(reduction[0] / static_cast<float>(width) + 1.0e-6F);
            if (feature == 0u) {
                means[row]                       = mean;
                inverse_standard_deviations[row] = inverse_standard_deviation;
            }
            if (feature < width) {
                const std::size_t modulation_offset                     = static_cast<std::size_t>(sample) * 6u * width + static_cast<std::size_t>(modulation_group) * width;
                const float shift                                       = modulation[modulation_offset + feature];
                const float scale                                       = modulation[modulation_offset + width + feature];
                output[static_cast<std::size_t>(row) * width + feature] = (value - mean) * inverse_standard_deviation * (1.0F + scale) + shift;
            }
        }

        __global__ void residual_forward_kernel(const float* const input, const float* const branch, const float* const modulation, float* const output, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::uint32_t sample    = static_cast<std::uint32_t>(index / (static_cast<std::size_t>(sequence) * width));
            const std::uint32_t feature   = static_cast<std::uint32_t>(index % width);
            const std::size_t gate_offset = static_cast<std::size_t>(sample) * 6u * width + (static_cast<std::size_t>(modulation_group) + 2u) * width;
            output[index]                 = input[index] + modulation[gate_offset + feature] * branch[index];
        }

        __global__ void residual_branch_backward_kernel(const float* const output_gradient, const float* const modulation, float* const branch_gradient, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::uint32_t sample    = static_cast<std::uint32_t>(index / (static_cast<std::size_t>(sequence) * width));
            const std::uint32_t feature   = static_cast<std::uint32_t>(index % width);
            const std::size_t gate_offset = static_cast<std::size_t>(sample) * 6u * width + (static_cast<std::size_t>(modulation_group) + 2u) * width;
            branch_gradient[index]        = output_gradient[index] * modulation[gate_offset + feature];
        }

        __global__ void residual_gate_backward_kernel(const float* const output_gradient, const float* const branch, float* const modulation_gradient, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group) {
            const std::uint32_t sample  = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            if (feature >= width) return;
            float gradient{};
            for (std::uint32_t token = 0u; token < sequence; ++token) {
                const std::size_t index = (static_cast<std::size_t>(sample) * sequence + token) * width + feature;
                gradient                = fmaf(output_gradient[index], branch[index], gradient);
            }
            const std::size_t gate_offset              = static_cast<std::size_t>(sample) * 6u * width + (static_cast<std::size_t>(modulation_group) + 2u) * width;
            modulation_gradient[gate_offset + feature] = gradient;
        }

        __global__ void adaln_input_backward_kernel(const float* const input, const float* const modulation, const float* const output_gradient, const float* const residual_gradient, const float* const means, const float* const inverse_standard_deviations, float* const input_gradient, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group) {
            __shared__ float gradient_sum[thread_count];
            __shared__ float normalized_gradient_sum[thread_count];
            const std::uint32_t row                = blockIdx.x;
            const std::uint32_t feature            = threadIdx.x;
            const std::uint32_t sample             = row / sequence;
            const float mean                       = means[row];
            const float inverse_standard_deviation = inverse_standard_deviations[row];
            float normalized{};
            float normalized_gradient{};
            if (feature < width) {
                const float value                   = input[static_cast<std::size_t>(row) * width + feature];
                normalized                          = (value - mean) * inverse_standard_deviation;
                const std::size_t modulation_offset = static_cast<std::size_t>(sample) * 6u * width + static_cast<std::size_t>(modulation_group) * width;
                normalized_gradient                 = output_gradient[static_cast<std::size_t>(row) * width + feature] * (1.0F + modulation[modulation_offset + width + feature]);
            }
            gradient_sum[feature]            = normalized_gradient;
            normalized_gradient_sum[feature] = normalized_gradient * normalized;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (feature < stride) {
                    gradient_sum[feature] += gradient_sum[feature + stride];
                    normalized_gradient_sum[feature] += normalized_gradient_sum[feature + stride];
                }
                __syncthreads();
            }
            if (feature < width) {
                const std::size_t index         = static_cast<std::size_t>(row) * width + feature;
                const float layer_norm_gradient = inverse_standard_deviation * (normalized_gradient - (gradient_sum[0] + normalized * normalized_gradient_sum[0]) / static_cast<float>(width));
                input_gradient[index]           = layer_norm_gradient + residual_gradient[index];
            }
        }

        __global__ void adaln_modulation_backward_kernel(const float* const input, const float* const output_gradient, const float* const means, const float* const inverse_standard_deviations, float* const modulation_gradient, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group) {
            const std::uint32_t sample  = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            if (feature >= width) return;
            float shift_gradient{};
            float scale_gradient{};
            for (std::uint32_t token = 0u; token < sequence; ++token) {
                const std::size_t row  = static_cast<std::size_t>(sample) * sequence + token;
                const float normalized = (input[row * width + feature] - means[row]) * inverse_standard_deviations[row];
                const float gradient   = output_gradient[row * width + feature];
                shift_gradient += gradient;
                scale_gradient = fmaf(gradient, normalized, scale_gradient);
            }
            const std::size_t modulation_offset                      = static_cast<std::size_t>(sample) * 6u * width + static_cast<std::size_t>(modulation_group) * width;
            modulation_gradient[modulation_offset + feature]         = shift_gradient;
            modulation_gradient[modulation_offset + width + feature] = scale_gradient;
        }

        __global__ void sdpa_forward_kernel(const float* const qkv, float* const output, float* const log_sum_exp, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t head_count, const std::uint32_t query_tile_count) {
            __shared__ float2 key_tile[attention_tile_width][attention_head_width / 2u];
            __shared__ float2 value_tile[attention_tile_width][attention_head_width / 2u];
            const std::uint32_t group               = threadIdx.x / attention_group_width;
            const std::uint32_t feature_group       = threadIdx.x % attention_group_width;
            const std::uint32_t first_feature_pair  = feature_group * 2u;
            const std::uint32_t second_feature_pair = first_feature_pair + 1u;
            const std::uint32_t query_tile          = blockIdx.x % query_tile_count;
            const std::uint32_t sample_head         = blockIdx.x / query_tile_count;
            const std::uint32_t sample              = sample_head / head_count;
            const std::uint32_t head                = sample_head % head_count;
            const std::uint32_t query_token         = query_tile * attention_query_tile_width + group;
            const bool active                       = query_token < sequence;
            const std::size_t qkv_stride            = static_cast<std::size_t>(3u) * width;
            const std::size_t query_offset          = (static_cast<std::size_t>(sample) * sequence + query_token) * qkv_stride + static_cast<std::size_t>(head) * attention_head_width + first_feature_pair * 2u;
            const float2 first_query                = active ? load_pair(qkv, query_offset) : make_float2(0.0F, 0.0F);
            const float2 second_query               = active ? load_pair(qkv, query_offset + 2u) : make_float2(0.0F, 0.0F);
            float2 first_accumulator                = make_float2(0.0F, 0.0F);
            float2 second_accumulator               = make_float2(0.0F, 0.0F);
            float maximum                           = -INFINITY;
            float denominator{};
            constexpr float scale = 0.17677669529663688110F;
            for (std::uint32_t tile = 0u; tile < sequence; tile += attention_tile_width) {
                const std::uint32_t tile_width = min(sequence - tile, attention_tile_width);
                for (std::uint32_t item = threadIdx.x; item < attention_tile_width * attention_head_width / 2u; item += blockDim.x) {
                    const std::uint32_t tile_token = item / (attention_head_width / 2u);
                    const std::uint32_t pair       = item % (attention_head_width / 2u);
                    if (tile_token < tile_width) {
                        const std::size_t source     = (static_cast<std::size_t>(sample) * sequence + tile + tile_token) * qkv_stride + static_cast<std::size_t>(head) * attention_head_width + pair * 2u;
                        key_tile[tile_token][pair]   = load_pair(qkv, source + width);
                        value_tile[tile_token][pair] = load_pair(qkv, source + 2u * width);
                    } else {
                        key_tile[tile_token][pair]   = make_float2(0.0F, 0.0F);
                        value_tile[tile_token][pair] = make_float2(0.0F, 0.0F);
                    }
                }
                __syncthreads();
                for (std::uint32_t key = 0u; key < tile_width; ++key) {
                    const float2 first_key  = key_tile[key][first_feature_pair];
                    const float2 second_key = key_tile[key][second_feature_pair];
                    const float score       = attention_group_sum(fmaf(first_query.x, first_key.x, fmaf(first_query.y, first_key.y, fmaf(second_query.x, second_key.x, second_query.y * second_key.y)))) * scale;
                    float previous_scale{};
                    float probability{};
                    if (feature_group == 0u) {
                        if (score > maximum) {
                            previous_scale = expf(maximum - score);
                            probability    = 1.0F;
                            maximum        = score;
                        } else {
                            previous_scale = 1.0F;
                            probability    = expf(score - maximum);
                        }
                        denominator = fmaf(denominator, previous_scale, probability);
                    }
                    previous_scale            = __shfl_sync(0xffffffffu, previous_scale, 0u, attention_group_width);
                    probability               = __shfl_sync(0xffffffffu, probability, 0u, attention_group_width);
                    const float2 first_value  = value_tile[key][first_feature_pair];
                    const float2 second_value = value_tile[key][second_feature_pair];
                    first_accumulator.x       = fmaf(probability, first_value.x, first_accumulator.x * previous_scale);
                    first_accumulator.y       = fmaf(probability, first_value.y, first_accumulator.y * previous_scale);
                    second_accumulator.x      = fmaf(probability, second_value.x, second_accumulator.x * previous_scale);
                    second_accumulator.y      = fmaf(probability, second_value.y, second_accumulator.y * previous_scale);
                }
                __syncthreads();
            }
            float inverse_denominator = feature_group == 0u ? 1.0F / denominator : 0.0F;
            inverse_denominator       = __shfl_sync(0xffffffffu, inverse_denominator, 0u, attention_group_width);
            if (active) {
                const std::size_t output_offset = (static_cast<std::size_t>(sample) * sequence + query_token) * width + static_cast<std::size_t>(head) * attention_head_width + first_feature_pair * 2u;
                store_pair(output, output_offset, make_float2(first_accumulator.x * inverse_denominator, first_accumulator.y * inverse_denominator));
                store_pair(output, output_offset + 2u, make_float2(second_accumulator.x * inverse_denominator, second_accumulator.y * inverse_denominator));
                if (feature_group == 0u) log_sum_exp[(static_cast<std::size_t>(sample) * head_count + head) * sequence + query_token] = maximum + logf(denominator);
            }
        }

        __global__ void sdpa_query_backward_kernel(const float* const qkv, const float* const output, const float* const output_gradient, const float* const log_sum_exp, float* const delta_values, float* const qkv_gradient, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t head_count, const std::uint32_t query_tile_count) {
            __shared__ float2 key_tile[attention_tile_width][attention_head_width / 2u];
            __shared__ float2 value_tile[attention_tile_width][attention_head_width / 2u];
            const std::uint32_t group               = threadIdx.x / attention_group_width;
            const std::uint32_t feature_group       = threadIdx.x % attention_group_width;
            const std::uint32_t first_feature_pair  = feature_group * 2u;
            const std::uint32_t second_feature_pair = first_feature_pair + 1u;
            const std::uint32_t query_tile          = blockIdx.x % query_tile_count;
            const std::uint32_t sample_head         = blockIdx.x / query_tile_count;
            const std::uint32_t sample              = sample_head / head_count;
            const std::uint32_t head                = sample_head % head_count;
            const std::uint32_t query_token         = query_tile * attention_query_tile_width + group;
            const bool active                       = query_token < sequence;
            const std::size_t qkv_stride            = static_cast<std::size_t>(3u) * width;
            const std::size_t query_offset          = (static_cast<std::size_t>(sample) * sequence + query_token) * qkv_stride + static_cast<std::size_t>(head) * attention_head_width + first_feature_pair * 2u;
            const std::size_t output_offset         = (static_cast<std::size_t>(sample) * sequence + query_token) * width + static_cast<std::size_t>(head) * attention_head_width + first_feature_pair * 2u;
            const float2 first_query                = active ? load_pair(qkv, query_offset) : make_float2(0.0F, 0.0F);
            const float2 second_query               = active ? load_pair(qkv, query_offset + 2u) : make_float2(0.0F, 0.0F);
            const float2 first_gradient             = active ? load_pair(output_gradient, output_offset) : make_float2(0.0F, 0.0F);
            const float2 second_gradient            = active ? load_pair(output_gradient, output_offset + 2u) : make_float2(0.0F, 0.0F);
            const float2 first_output               = active ? load_pair(output, output_offset) : make_float2(0.0F, 0.0F);
            const float2 second_output              = active ? load_pair(output, output_offset + 2u) : make_float2(0.0F, 0.0F);
            const float delta                       = attention_group_sum(fmaf(first_gradient.x, first_output.x, fmaf(first_gradient.y, first_output.y, fmaf(second_gradient.x, second_output.x, second_gradient.y * second_output.y))));
            if (active && feature_group == 0u) delta_values[(static_cast<std::size_t>(sample) * head_count + head) * sequence + query_token] = delta;
            const float lse              = active ? log_sum_exp[(static_cast<std::size_t>(sample) * head_count + head) * sequence + query_token] : 0.0F;
            float2 first_query_gradient  = make_float2(0.0F, 0.0F);
            float2 second_query_gradient = make_float2(0.0F, 0.0F);
            constexpr float scale        = 0.17677669529663688110F;
            for (std::uint32_t tile = 0u; tile < sequence; tile += attention_tile_width) {
                const std::uint32_t tile_width = min(sequence - tile, attention_tile_width);
                for (std::uint32_t item = threadIdx.x; item < attention_tile_width * attention_head_width / 2u; item += blockDim.x) {
                    const std::uint32_t tile_token = item / (attention_head_width / 2u);
                    const std::uint32_t pair       = item % (attention_head_width / 2u);
                    if (tile_token < tile_width) {
                        const std::size_t source     = (static_cast<std::size_t>(sample) * sequence + tile + tile_token) * qkv_stride + static_cast<std::size_t>(head) * attention_head_width + pair * 2u;
                        key_tile[tile_token][pair]   = load_pair(qkv, source + width);
                        value_tile[tile_token][pair] = load_pair(qkv, source + 2u * width);
                    } else {
                        key_tile[tile_token][pair]   = make_float2(0.0F, 0.0F);
                        value_tile[tile_token][pair] = make_float2(0.0F, 0.0F);
                    }
                }
                __syncthreads();
                for (std::uint32_t key = 0u; key < tile_width; ++key) {
                    const float2 first_key           = key_tile[key][first_feature_pair];
                    const float2 second_key          = key_tile[key][second_feature_pair];
                    const float2 first_value         = value_tile[key][first_feature_pair];
                    const float2 second_value        = value_tile[key][second_feature_pair];
                    const float score                = attention_group_sum(fmaf(first_query.x, first_key.x, fmaf(first_query.y, first_key.y, fmaf(second_query.x, second_key.x, second_query.y * second_key.y)))) * scale;
                    const float probability_gradient = attention_group_sum(fmaf(first_gradient.x, first_value.x, fmaf(first_gradient.y, first_value.y, fmaf(second_gradient.x, second_value.x, second_gradient.y * second_value.y))));
                    float score_gradient             = feature_group == 0u ? expf(score - lse) * (probability_gradient - delta) * scale : 0.0F;
                    score_gradient                   = __shfl_sync(0xffffffffu, score_gradient, 0u, attention_group_width);
                    first_query_gradient.x           = fmaf(score_gradient, first_key.x, first_query_gradient.x);
                    first_query_gradient.y           = fmaf(score_gradient, first_key.y, first_query_gradient.y);
                    second_query_gradient.x          = fmaf(score_gradient, second_key.x, second_query_gradient.x);
                    second_query_gradient.y          = fmaf(score_gradient, second_key.y, second_query_gradient.y);
                }
                __syncthreads();
            }
            if (active) {
                store_pair(qkv_gradient, query_offset, first_query_gradient);
                store_pair(qkv_gradient, query_offset + 2u, second_query_gradient);
            }
        }

        __global__ void sdpa_key_value_backward_kernel(const float* const qkv, const float* const output_gradient, const float* const log_sum_exp, const float* const delta_values, float* const qkv_gradient, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t head_count, const std::uint32_t key_tile_count) {
            const std::uint32_t group              = threadIdx.x / attention_group_width;
            const std::uint32_t feature_group      = threadIdx.x % attention_group_width;
            const std::uint32_t first_feature_pair = feature_group * 2u;
            const std::uint32_t key_tile           = blockIdx.x % key_tile_count;
            const std::uint32_t sample_head        = blockIdx.x / key_tile_count;
            const std::uint32_t sample             = sample_head / head_count;
            const std::uint32_t head               = sample_head % head_count;
            const std::uint32_t key_token          = key_tile * attention_query_tile_width + group;
            const bool active                      = key_token < sequence;
            const std::size_t qkv_stride           = static_cast<std::size_t>(3u) * width;
            const std::size_t key_offset           = (static_cast<std::size_t>(sample) * sequence + key_token) * qkv_stride + static_cast<std::size_t>(head) * attention_head_width + first_feature_pair * 2u;
            const float2 first_key                 = active ? load_pair(qkv, key_offset + width) : make_float2(0.0F, 0.0F);
            const float2 second_key                = active ? load_pair(qkv, key_offset + width + 2u) : make_float2(0.0F, 0.0F);
            const float2 first_value               = active ? load_pair(qkv, key_offset + 2u * width) : make_float2(0.0F, 0.0F);
            const float2 second_value              = active ? load_pair(qkv, key_offset + 2u * width + 2u) : make_float2(0.0F, 0.0F);
            float2 first_key_gradient              = make_float2(0.0F, 0.0F);
            float2 second_key_gradient             = make_float2(0.0F, 0.0F);
            float2 first_value_gradient            = make_float2(0.0F, 0.0F);
            float2 second_value_gradient           = make_float2(0.0F, 0.0F);
            constexpr float scale                  = 0.17677669529663688110F;
            for (std::uint32_t query_token = 0u; query_token < sequence; ++query_token) {
                const std::size_t query_offset   = (static_cast<std::size_t>(sample) * sequence + query_token) * qkv_stride + static_cast<std::size_t>(head) * attention_head_width + first_feature_pair * 2u;
                const std::size_t output_offset  = (static_cast<std::size_t>(sample) * sequence + query_token) * width + static_cast<std::size_t>(head) * attention_head_width + first_feature_pair * 2u;
                const float2 first_query         = load_pair(qkv, query_offset);
                const float2 second_query        = load_pair(qkv, query_offset + 2u);
                const float2 first_gradient      = load_pair(output_gradient, output_offset);
                const float2 second_gradient     = load_pair(output_gradient, output_offset + 2u);
                const float score                = attention_group_sum(fmaf(first_query.x, first_key.x, fmaf(first_query.y, first_key.y, fmaf(second_query.x, second_key.x, second_query.y * second_key.y)))) * scale;
                const float probability_gradient = attention_group_sum(fmaf(first_gradient.x, first_value.x, fmaf(first_gradient.y, first_value.y, fmaf(second_gradient.x, second_value.x, second_gradient.y * second_value.y))));
                float probability                = feature_group == 0u ? expf(score - log_sum_exp[(static_cast<std::size_t>(sample) * head_count + head) * sequence + query_token]) : 0.0F;
                probability                      = __shfl_sync(0xffffffffu, probability, 0u, attention_group_width);
                const float score_gradient       = probability * (probability_gradient - delta_values[(static_cast<std::size_t>(sample) * head_count + head) * sequence + query_token]) * scale;
                first_key_gradient.x             = fmaf(score_gradient, first_query.x, first_key_gradient.x);
                first_key_gradient.y             = fmaf(score_gradient, first_query.y, first_key_gradient.y);
                second_key_gradient.x            = fmaf(score_gradient, second_query.x, second_key_gradient.x);
                second_key_gradient.y            = fmaf(score_gradient, second_query.y, second_key_gradient.y);
                first_value_gradient.x           = fmaf(probability, first_gradient.x, first_value_gradient.x);
                first_value_gradient.y           = fmaf(probability, first_gradient.y, first_value_gradient.y);
                second_value_gradient.x          = fmaf(probability, second_gradient.x, second_value_gradient.x);
                second_value_gradient.y          = fmaf(probability, second_gradient.y, second_value_gradient.y);
            }
            if (active) {
                store_pair(qkv_gradient, key_offset + width, first_key_gradient);
                store_pair(qkv_gradient, key_offset + width + 2u, second_key_gradient);
                store_pair(qkv_gradient, key_offset + 2u * width, first_value_gradient);
                store_pair(qkv_gradient, key_offset + 2u * width + 2u, second_value_gradient);
            }
        }
    } // namespace

    void adaln_forward(const ::cuda::stream_ref stream, const float* const input, const float* const modulation, float* const output, float* const means, float* const inverse_standard_deviations, const std::uint32_t batch, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group) {
        adaln_forward_kernel<<<batch * sequence, thread_count, 0u, stream.get()>>>(input, modulation, output, means, inverse_standard_deviations, sequence, width, modulation_group);
    }

    void residual_forward(const ::cuda::stream_ref stream, const float* const input, const float* const branch, const float* const modulation, float* const output, const std::uint32_t batch, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group) {
        const std::size_t count = static_cast<std::size_t>(batch) * sequence * width;
        residual_forward_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(input, branch, modulation, output, sequence, width, modulation_group, count);
    }

    void residual_backward(const ::cuda::stream_ref stream, const float* const output_gradient, const float* const branch, const float* const modulation, float* const branch_gradient, float* const modulation_gradient, const std::uint32_t batch, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group) {
        const std::size_t count = static_cast<std::size_t>(batch) * sequence * width;
        residual_branch_backward_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(output_gradient, modulation, branch_gradient, sequence, width, modulation_group, count);
        residual_gate_backward_kernel<<<batch, thread_count, 0u, stream.get()>>>(output_gradient, branch, modulation_gradient, sequence, width, modulation_group);
    }

    void adaln_backward(const ::cuda::stream_ref stream, const float* const input, const float* const modulation, const float* const output_gradient, const float* const residual_gradient, const float* const means, const float* const inverse_standard_deviations, float* const input_gradient, float* const modulation_gradient, const std::uint32_t batch, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t modulation_group) {
        adaln_input_backward_kernel<<<batch * sequence, thread_count, 0u, stream.get()>>>(input, modulation, output_gradient, residual_gradient, means, inverse_standard_deviations, input_gradient, sequence, width, modulation_group);
        adaln_modulation_backward_kernel<<<batch, thread_count, 0u, stream.get()>>>(input, output_gradient, means, inverse_standard_deviations, modulation_gradient, sequence, width, modulation_group);
    }

    void sdpa_forward(const ::cuda::stream_ref stream, const float* const qkv, float* const output, float* const log_sum_exp, const std::uint32_t batch, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t head_count) {
        const std::uint32_t query_tile_count = ::cuda::ceil_div(sequence, attention_query_tile_width);
        sdpa_forward_kernel<<<batch * head_count * query_tile_count, thread_count, 0u, stream.get()>>>(qkv, output, log_sum_exp, sequence, width, head_count, query_tile_count);
    }

    void sdpa_backward(const ::cuda::stream_ref stream, const float* const qkv, const float* const output, const float* const output_gradient, const float* const log_sum_exp, float* const delta, float* const qkv_gradient, const std::uint32_t batch, const std::uint32_t sequence, const std::uint32_t width, const std::uint32_t head_count) {
        const std::uint32_t tile_count = ::cuda::ceil_div(sequence, attention_query_tile_width);
        sdpa_query_backward_kernel<<<batch * head_count * tile_count, thread_count, 0u, stream.get()>>>(qkv, output, output_gradient, log_sum_exp, delta, qkv_gradient, sequence, width, head_count, tile_count);
        sdpa_key_value_backward_kernel<<<batch * head_count * tile_count, thread_count, 0u, stream.get()>>>(qkv, output_gradient, log_sum_exp, delta, qkv_gradient, sequence, width, head_count, tile_count);
    }
} // namespace physica::neural::kernels
