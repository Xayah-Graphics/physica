#include "kernels.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda/launch>
#include <curand_kernel.h>
#include <physica/cuda.h>

namespace physica::generative::flow_matching::kernels {
    namespace {
        constexpr std::uint32_t thread_count        = 256u;
        constexpr std::uint32_t image_element_count = 32u * 32u * 3u;
        constexpr std::uint32_t sequence            = 256u;
        constexpr std::uint32_t patch_width         = 12u;

        __device__ unsigned long long subsequence(const std::uint64_t step, const std::uint32_t batch, const std::uint32_t sample, const std::uint32_t domain) {
            return (step * batch + sample) * 8ull + domain;
        }

        __global__ void make_training_batch_kernel(const std::uint8_t* const images, const std::uint8_t* const dataset_labels, float* const path, float* const target, float* const times, std::uint8_t* const labels, const std::uint64_t* const step, const std::uint64_t* const seed, const std::uint32_t batch) {
            __shared__ std::uint32_t image_index;
            __shared__ std::uint32_t flip;
            __shared__ float time;
            const std::uint32_t sample = blockIdx.x;
            if (threadIdx.x == 0u) {
                curandStatePhilox4_32_10_t state{};
                curand_init(*seed, subsequence(*step, batch, sample, 0u), 0ull, &state);
                image_index = static_cast<std::uint32_t>(static_cast<std::uint64_t>(curand(&state)) * 50'000u >> 32u);
                flip        = curand(&state) & 1u;
                curand_init(*seed, subsequence(*step, batch, sample, 1u), 0ull, &state);
                time = static_cast<float>(curand(&state) >> 8u) * 0x1p-24F;
                curand_init(*seed, subsequence(*step, batch, sample, 2u), 0ull, &state);
                labels[sample] = static_cast<float>(curand(&state) >> 8u) * 0x1p-24F < 0.1F ? 10u : dataset_labels[image_index];
                times[sample]  = time;
            }
            __syncthreads();
            for (std::uint32_t group = threadIdx.x; group < image_element_count / 4u; group += blockDim.x) {
                curandStatePhilox4_32_10_t state{};
                curand_init(*seed, subsequence(*step, batch, sample, 3u), static_cast<unsigned long long>(group) * 4ull, &state);
                const float4 noise = curand_normal4(&state);
                const float gaussian[4]{noise.x, noise.y, noise.z, noise.w};
                for (std::uint32_t lane = 0u; lane < 4u; ++lane) {
                    const std::uint32_t patch_index   = group * 4u + lane;
                    const std::uint32_t token         = patch_index / patch_width;
                    const std::uint32_t patch_element = patch_index % patch_width;
                    const std::uint32_t patch_y       = token / 16u;
                    const std::uint32_t patch_x       = token % 16u;
                    const std::uint32_t pixel         = patch_element / 3u;
                    const std::uint32_t channel       = patch_element % 3u;
                    const std::uint32_t y             = patch_y * 2u + pixel / 2u;
                    const std::uint32_t unflipped_x   = patch_x * 2u + pixel % 2u;
                    const std::uint32_t x             = flip == 0u ? unflipped_x : 31u - unflipped_x;
                    const std::size_t source          = static_cast<std::size_t>(image_index) * image_element_count + static_cast<std::size_t>(channel) * 1024u + y * 32u + x;
                    const float data                  = static_cast<float>(images[source]) * (2.0F / 255.0F) - 1.0F;
                    const std::size_t destination     = static_cast<std::size_t>(sample) * image_element_count + patch_index;
                    path[destination]                 = fmaf(time, data - gaussian[lane], gaussian[lane]);
                    target[destination]               = data - gaussian[lane];
                }
            }
        }

        __global__ void make_sampling_noise_kernel(float* const state, const std::uint64_t seed, const std::uint32_t batch) {
            const std::uint32_t sample = blockIdx.x;
            for (std::uint32_t group = threadIdx.x; group < image_element_count / 4u; group += blockDim.x) {
                curandStatePhilox4_32_10_t random{};
                curand_init(seed, subsequence(0u, batch, sample, 4u), static_cast<unsigned long long>(group) * 4ull, &random);
                const float4 noise      = curand_normal4(&random);
                const std::size_t index = static_cast<std::size_t>(sample) * image_element_count + group * 4u;
                state[index]            = noise.x;
                state[index + 1u]       = noise.y;
                state[index + 2u]       = noise.z;
                state[index + 3u]       = noise.w;
            }
        }

        __global__ void make_sampling_time_kernel(float* const times, const float time, const std::uint32_t batch) {
            const std::uint32_t sample = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (sample < batch) times[sample] = time;
        }

        __global__ void time_embedding_kernel(const float* const times, float* const embedding, const std::uint32_t width, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::uint32_t feature         = static_cast<std::uint32_t>(index % width);
            const std::uint32_t frequency_index = feature % (width / 2u);
            const float frequency               = expf(-logf(10'000.0F) * static_cast<float>(frequency_index) / static_cast<float>(width / 2u));
            const float angle                   = times[index / width] * frequency;
            embedding[index]                    = feature < width / 2u ? cosf(angle) : sinf(angle);
        }

        __global__ void silu_forward_kernel(const float* const input, float* const output, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const float value = input[index];
            output[index]     = value / (1.0F + expf(-value));
        }

        __global__ void silu_backward_kernel(const float* const input, const float* const output_gradient, float* const input_gradient, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const float value     = input[index];
            const float sigmoid   = 1.0F / (1.0F + expf(-value));
            input_gradient[index] = output_gradient[index] * sigmoid * (1.0F + value * (1.0F - sigmoid));
        }

        __global__ void add_position_kernel(float* const tokens, const float* const position, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) tokens[index] += position[index % (sequence * 256u)];
        }

        __global__ void make_condition_kernel(float* const time_condition, const float* const class_embedding, const std::uint8_t* const labels, const std::uint32_t width, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::uint32_t feature = static_cast<std::uint32_t>(index % width);
            time_condition[index] += class_embedding[static_cast<std::size_t>(labels[index / width]) * width + feature];
        }

        __global__ void class_embedding_backward_kernel(const float* const condition_gradient, const std::uint8_t* const labels, float* const class_embedding_gradient, const std::uint32_t batch, const std::uint32_t width) {
            const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= 11u * width) return;
            const std::uint32_t label   = index / width;
            const std::uint32_t feature = index % width;
            float gradient{};
            for (std::uint32_t sample = 0u; sample < batch; ++sample)
                if (labels[sample] == label) gradient += condition_gradient[static_cast<std::size_t>(sample) * width + feature];
            class_embedding_gradient[index] = gradient;
        }

        __global__ void final_adaln_forward_kernel(const float* const input, const float* const modulation, float* const output, float* const means, float* const inverse_standard_deviations, const std::uint32_t sequence_size, const std::uint32_t width) {
            __shared__ float reduction[thread_count];
            const std::uint32_t row     = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            const std::uint32_t sample  = row / sequence_size;
            const float value           = feature < width ? input[static_cast<std::size_t>(row) * width + feature] : 0.0F;
            reduction[feature]          = value;
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
                const std::size_t modulation_offset                     = static_cast<std::size_t>(sample) * 2u * width;
                output[static_cast<std::size_t>(row) * width + feature] = (value - mean) * inverse_standard_deviation * (1.0F + modulation[modulation_offset + width + feature]) + modulation[modulation_offset + feature];
            }
        }

        __global__ void final_adaln_input_backward_kernel(const float* const input, const float* const modulation, const float* const output_gradient, const float* const means, const float* const inverse_standard_deviations, float* const input_gradient, const std::uint32_t sequence_size, const std::uint32_t width) {
            __shared__ float gradient_sum[thread_count];
            __shared__ float normalized_gradient_sum[thread_count];
            const std::uint32_t row     = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            const std::uint32_t sample  = row / sequence_size;
            float normalized{};
            float gradient{};
            if (feature < width) {
                normalized = (input[static_cast<std::size_t>(row) * width + feature] - means[row]) * inverse_standard_deviations[row];
                gradient   = output_gradient[static_cast<std::size_t>(row) * width + feature] * (1.0F + modulation[static_cast<std::size_t>(sample) * 2u * width + width + feature]);
            }
            gradient_sum[feature]            = gradient;
            normalized_gradient_sum[feature] = gradient * normalized;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (feature < stride) {
                    gradient_sum[feature] += gradient_sum[feature + stride];
                    normalized_gradient_sum[feature] += normalized_gradient_sum[feature + stride];
                }
                __syncthreads();
            }
            if (feature < width) input_gradient[static_cast<std::size_t>(row) * width + feature] = inverse_standard_deviations[row] * (gradient - (gradient_sum[0] + normalized * normalized_gradient_sum[0]) / static_cast<float>(width));
        }

        __global__ void final_adaln_modulation_backward_kernel(const float* const input, const float* const output_gradient, const float* const means, const float* const inverse_standard_deviations, float* const modulation_gradient, const std::uint32_t sequence_size, const std::uint32_t width) {
            const std::uint32_t sample  = blockIdx.x;
            const std::uint32_t feature = threadIdx.x;
            if (feature >= width) return;
            float shift_gradient{};
            float scale_gradient{};
            for (std::uint32_t token = 0u; token < sequence_size; ++token) {
                const std::size_t row = static_cast<std::size_t>(sample) * sequence_size + token;
                const float gradient  = output_gradient[row * width + feature];
                shift_gradient += gradient;
                scale_gradient = fmaf(gradient, (input[row * width + feature] - means[row]) * inverse_standard_deviations[row], scale_gradient);
            }
            const std::size_t offset                      = static_cast<std::size_t>(sample) * 2u * width;
            modulation_gradient[offset + feature]         = shift_gradient;
            modulation_gradient[offset + width + feature] = scale_gradient;
        }

        __global__ void flow_matching_sample_loss_kernel(const float* const prediction, const float* const target, float* const prediction_gradient, float* const sample_loss) {
            __shared__ float reduction[thread_count];
            const std::uint32_t sample = blockIdx.x;
            float loss{};
            for (std::uint32_t element = threadIdx.x; element < image_element_count; element += blockDim.x) {
                const std::size_t index    = static_cast<std::size_t>(sample) * image_element_count + element;
                const float difference     = prediction[index] - target[index];
                loss                       = fmaf(difference, difference, loss);
                prediction_gradient[index] = 2.0F * difference / static_cast<float>(gridDim.x * image_element_count);
            }
            reduction[threadIdx.x] = loss;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0u) sample_loss[sample] = reduction[0];
        }

        __global__ void flow_matching_loss_kernel(const float* const sample_loss, float* const loss, const std::uint32_t batch) {
            __shared__ float reduction[thread_count];
            reduction[threadIdx.x] = threadIdx.x < batch ? sample_loss[threadIdx.x] : 0.0F;
            __syncthreads();
            for (std::uint32_t stride = thread_count / 2u; stride != 0u; stride /= 2u) {
                if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0u) *loss = reduction[0] / static_cast<float>(batch * image_element_count);
        }

        __global__ void add_loss_kernel(const float* const loss, float* const loss_sum) {
            *loss_sum += *loss;
        }

        __global__ void advance_training_state_kernel(std::uint64_t* const step, std::uint64_t* const processed_samples, const std::uint32_t samples_per_step) {
            ++*step;
            *processed_samples += samples_per_step;
        }

        __global__ void make_labels_kernel(std::uint8_t* const labels, const std::uint32_t batch, const std::uint32_t class_index) {
            const std::uint32_t sample = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (sample < batch) labels[sample] = class_index == 10u ? sample % 10u : static_cast<std::uint8_t>(class_index);
        }

        __global__ void combine_guidance_kernel(const float* const conditional, const float* const unconditional, float* const output, const float guidance, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) output[index] = fmaf(guidance, conditional[index] - unconditional[index], unconditional[index]);
        }

        __global__ void euler_step_kernel(float* const state, const float* const velocity, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) state[index] = fmaf(step_size, velocity[index], state[index]);
        }

        __global__ void heun_predict_kernel(const float* const state, const float* const velocity, float* const prediction, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) prediction[index] = fmaf(step_size, velocity[index], state[index]);
        }

        __global__ void heun_step_kernel(float* const state, const float* const first_velocity, const float* const second_velocity, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) state[index] = fmaf(0.5F * step_size, first_velocity[index] + second_velocity[index], state[index]);
        }

        __global__ void rk4_intermediate_kernel(const float* const state, const float* const velocity, float* const intermediate, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) intermediate[index] = fmaf(step_size, velocity[index], state[index]);
        }

        __global__ void rk4_step_kernel(float* const state, const float* const first, const float* const second, const float* const third, const float* const fourth, const float step_size, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) state[index] += step_size * (first[index] + 2.0F * second[index] + 2.0F * third[index] + fourth[index]) / 6.0F;
        }

        __global__ void unpatchify_kernel(const float* const patches, std::uint8_t* const rgba, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const std::uint32_t sample      = static_cast<std::uint32_t>(index / (32u * 32u));
            const std::uint32_t pixel       = static_cast<std::uint32_t>(index % (32u * 32u));
            const std::uint32_t y           = pixel / 32u;
            const std::uint32_t x           = pixel % 32u;
            const std::uint32_t token       = (y / 2u) * 16u + x / 2u;
            const std::uint32_t patch_pixel = (y % 2u) * 2u + x % 2u;
            for (std::uint32_t channel = 0u; channel < 3u; ++channel) {
                const float value          = fminf(fmaxf(patches[static_cast<std::size_t>(sample) * image_element_count + token * patch_width + patch_pixel * 3u + channel], -1.0F), 1.0F);
                rgba[index * 4u + channel] = static_cast<std::uint8_t>(rintf((value + 1.0F) * 127.5F));
            }
            rgba[index * 4u + 3u] = 255u;
        }
    } // namespace

    void make_training_batch(const ::cuda::stream_ref stream, const std::uint8_t* const images, const std::uint8_t* const dataset_labels, float* const path, float* const target, float* const times, std::uint8_t* const labels, const std::uint64_t* const step, const std::uint64_t* const seed, const std::uint32_t batch) {
        make_training_batch_kernel<<<batch, thread_count, 0u, stream.get()>>>(images, dataset_labels, path, target, times, labels, step, seed, batch);
    }

    void make_sampling_noise(const ::cuda::stream_ref stream, float* const state, const std::uint64_t seed, const std::uint32_t batch) {
        make_sampling_noise_kernel<<<batch, thread_count, 0u, stream.get()>>>(state, seed, batch);
    }

    void make_sampling_time(const ::cuda::stream_ref stream, float* const times, const float time, const std::uint32_t batch) {
        make_sampling_time_kernel<<<::cuda::ceil_div(batch, thread_count), thread_count, 0u, stream.get()>>>(times, time, batch);
    }

    void make_time_embedding(const ::cuda::stream_ref stream, const float* const times, float* const embedding, const std::uint32_t batch, const std::uint32_t width) {
        const std::size_t count = static_cast<std::size_t>(batch) * width;
        time_embedding_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(times, embedding, width, count);
    }

    void silu_forward(const ::cuda::stream_ref stream, const float* const input, float* const output, const std::size_t count) {
        silu_forward_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(input, output, count);
    }

    void silu_backward(const ::cuda::stream_ref stream, const float* const input, const float* const output_gradient, float* const input_gradient, const std::size_t count) {
        silu_backward_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(input, output_gradient, input_gradient, count);
    }

    void add_position(const ::cuda::stream_ref stream, float* const tokens, const float* const position, const std::size_t count) {
        add_position_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(tokens, position, count);
    }

    void make_condition(const ::cuda::stream_ref stream, float* const time_condition, const float* const class_embedding, const std::uint8_t* const labels, const std::uint32_t batch, const std::uint32_t width) {
        const std::size_t count = static_cast<std::size_t>(batch) * width;
        make_condition_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(time_condition, class_embedding, labels, width, count);
    }

    void class_embedding_backward(const ::cuda::stream_ref stream, const float* const condition_gradient, const std::uint8_t* const labels, float* const class_embedding_gradient, const std::uint32_t batch, const std::uint32_t width) {
        class_embedding_backward_kernel<<<::cuda::ceil_div(11u * width, thread_count), thread_count, 0u, stream.get()>>>(condition_gradient, labels, class_embedding_gradient, batch, width);
    }

    void final_adaln_forward(const ::cuda::stream_ref stream, const float* const input, const float* const modulation, float* const output, float* const means, float* const inverse_standard_deviations, const std::uint32_t batch, const std::uint32_t sequence_size, const std::uint32_t width) {
        final_adaln_forward_kernel<<<batch * sequence_size, thread_count, 0u, stream.get()>>>(input, modulation, output, means, inverse_standard_deviations, sequence_size, width);
    }

    void final_adaln_backward(const ::cuda::stream_ref stream, const float* const input, const float* const modulation, const float* const output_gradient, const float* const means, const float* const inverse_standard_deviations, float* const input_gradient, float* const modulation_gradient, const std::uint32_t batch, const std::uint32_t sequence_size, const std::uint32_t width) {
        final_adaln_input_backward_kernel<<<batch * sequence_size, thread_count, 0u, stream.get()>>>(input, modulation, output_gradient, means, inverse_standard_deviations, input_gradient, sequence_size, width);
        final_adaln_modulation_backward_kernel<<<batch, thread_count, 0u, stream.get()>>>(input, output_gradient, means, inverse_standard_deviations, modulation_gradient, sequence_size, width);
    }

    void flow_matching_loss(const ::cuda::stream_ref stream, const float* const prediction, const float* const target, float* const prediction_gradient, float* const sample_loss, float* const loss, const std::uint32_t batch) {
        flow_matching_sample_loss_kernel<<<batch, thread_count, 0u, stream.get()>>>(prediction, target, prediction_gradient, sample_loss);
        flow_matching_loss_kernel<<<1u, thread_count, 0u, stream.get()>>>(sample_loss, loss, batch);
    }

    void add_loss(const ::cuda::stream_ref stream, const float* const loss, float* const loss_sum) {
        add_loss_kernel<<<1u, 1u, 0u, stream.get()>>>(loss, loss_sum);
    }

    void advance_training_state(const ::cuda::stream_ref stream, std::uint64_t* const step, std::uint64_t* const processed_samples, const std::uint32_t samples_per_step) {
        advance_training_state_kernel<<<1u, 1u, 0u, stream.get()>>>(step, processed_samples, samples_per_step);
    }

    void make_labels(const ::cuda::stream_ref stream, std::uint8_t* const labels, const std::uint32_t batch, const std::uint32_t class_index) {
        make_labels_kernel<<<::cuda::ceil_div(batch, thread_count), thread_count, 0u, stream.get()>>>(labels, batch, class_index);
    }

    void combine_guidance(const ::cuda::stream_ref stream, const float* const conditional, const float* const unconditional, float* const output, const float guidance, const std::size_t count) {
        combine_guidance_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(conditional, unconditional, output, guidance, count);
    }

    void euler_step(const ::cuda::stream_ref stream, float* const state, const float* const velocity, const float step_size, const std::size_t count) {
        euler_step_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(state, velocity, step_size, count);
    }

    void heun_predict(const ::cuda::stream_ref stream, const float* const state, const float* const velocity, float* const prediction, const float step_size, const std::size_t count) {
        heun_predict_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(state, velocity, prediction, step_size, count);
    }

    void heun_step(const ::cuda::stream_ref stream, float* const state, const float* const first_velocity, const float* const second_velocity, const float step_size, const std::size_t count) {
        heun_step_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(state, first_velocity, second_velocity, step_size, count);
    }

    void rk4_intermediate(const ::cuda::stream_ref stream, const float* const state, const float* const velocity, float* const intermediate, const float step_size, const std::size_t count) {
        rk4_intermediate_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(state, velocity, intermediate, step_size, count);
    }

    void rk4_step(const ::cuda::stream_ref stream, float* const state, const float* const first, const float* const second, const float* const third, const float* const fourth, const float step_size, const std::size_t count) {
        rk4_step_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(state, first, second, third, fourth, step_size, count);
    }

    void unpatchify(const ::cuda::stream_ref stream, const float* const patches, std::uint8_t* const rgba, const std::uint32_t batch) {
        const std::size_t count = static_cast<std::size_t>(batch) * 32u * 32u;
        unpatchify_kernel<<<::cuda::ceil_div(count, static_cast<std::size_t>(thread_count)), thread_count, 0u, stream.get()>>>(patches, rgba, count);
    }
} // namespace physica::generative::flow_matching::kernels
