#include "kernels.h"
#include <cuda/launch>
#include <cuda/std/cmath>
#include <cuda_runtime.h>

namespace physica::reconstruction::pinfs::kernels {
    namespace {
        __device__ float activation_value(const Activation activation, const float frequency, const float value, const std::uint32_t feature) {
            if (activation == Activation::linear) return value;
            if (activation == Activation::sine) return sinf(frequency * value);
            if (activation == Activation::relu) return fmaxf(value, 0.0F);
            if (activation == Activation::sigmoid || activation == Activation::rgb_density && feature < 3u) return 1.0F / (1.0F + expf(-value));
            if (activation == Activation::rgb_density) return fmaxf(value, 0.0F);
            if (value > 0.0F) return value + log1pf(expf(-frequency * value)) / frequency;
            return log1pf(expf(frequency * value)) / frequency;
        }

        __device__ float activation_first_derivative(const Activation activation, const float frequency, const float value, const std::uint32_t feature) {
            if (activation == Activation::linear) return 1.0F;
            if (activation == Activation::sine) return frequency * cosf(frequency * value);
            if (activation == Activation::relu || activation == Activation::rgb_density && feature >= 3u) return value > 0.0F ? 1.0F : 0.0F;
            const float sigmoid = 1.0F / (1.0F + expf(-(activation == Activation::softplus ? frequency : 1.0F) * value));
            if (activation == Activation::sigmoid || activation == Activation::rgb_density) return sigmoid * (1.0F - sigmoid);
            return sigmoid;
        }

        __device__ float activation_second_derivative(const Activation activation, const float frequency, const float value, const std::uint32_t feature) {
            if (activation == Activation::linear || activation == Activation::relu || activation == Activation::rgb_density && feature >= 3u) return 0.0F;
            if (activation == Activation::sine) return -frequency * frequency * sinf(frequency * value);
            const float multiplier = activation == Activation::softplus ? frequency : 1.0F;
            const float sigmoid    = 1.0F / (1.0F + expf(-multiplier * value));
            if (activation == Activation::sigmoid || activation == Activation::rgb_density) {
                const float first = sigmoid * (1.0F - sigmoid);
                return first * (1.0F - 2.0F * sigmoid);
            }
            return frequency * sigmoid * (1.0F - sigmoid);
        }

        __global__ void weight_normalization_forward_kernel(const float* weights, const float* scales, float* normalized, const std::uint32_t input_width, const std::uint32_t output_width) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= output_width) return;
            float squared_norm{};
            for (std::uint32_t column = 0u; column < input_width; ++column) {
                const float weight = weights[row + static_cast<std::size_t>(column) * output_width];
                squared_norm       = fmaf(weight, weight, squared_norm);
            }
            const float factor = scales[row] * rsqrtf(squared_norm);
            for (std::uint32_t column = 0u; column < input_width; ++column) normalized[row + static_cast<std::size_t>(column) * output_width] = weights[row + static_cast<std::size_t>(column) * output_width] * factor;
        }

        __global__ void weight_normalization_backward_kernel(const float* weights, const float* scales, const float* normalized_gradients, float* weight_gradients, float* scale_gradients, const std::uint32_t input_width, const std::uint32_t output_width) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= output_width) return;
            float squared_norm{};
            float dot{};
            for (std::uint32_t column = 0u; column < input_width; ++column) {
                const std::size_t index = row + static_cast<std::size_t>(column) * output_width;
                squared_norm            = fmaf(weights[index], weights[index], squared_norm);
                dot                     = fmaf(normalized_gradients[index], weights[index], dot);
            }
            const float inverse_norm = rsqrtf(squared_norm);
            const float projection   = dot / squared_norm;
            const float factor       = scales[row] * inverse_norm;
            for (std::uint32_t column = 0u; column < input_width; ++column) {
                const std::size_t index = row + static_cast<std::size_t>(column) * output_width;
                weight_gradients[index] += factor * (normalized_gradients[index] - weights[index] * projection);
            }
            scale_gradients[row] += dot * inverse_norm;
        }

        __global__ void activation_forward_kernel(const Activation activation, const float frequency, const float* biases, float* linear, const float* linear_derivatives, float* output, float* output_derivatives, const std::uint32_t width, const std::uint32_t sample_count, const std::uint32_t derivative_count) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t count = width * sample_count;
            if (index >= count) return;
            const float value = linear[index] + biases[index % width];
            linear[index]     = value;
            output[index]     = activation_value(activation, frequency, value, index % width);
            const float first = activation_first_derivative(activation, frequency, value, index % width);
            for (std::uint32_t derivative = 0u; derivative < derivative_count; ++derivative) output_derivatives[index + static_cast<std::size_t>(derivative) * count] = first * linear_derivatives[index + static_cast<std::size_t>(derivative) * count];
        }

        __global__ void activation_backward_kernel(const Activation activation, const float frequency, const float* linear, const float* linear_derivatives, const float* output_adjoints, const float* output_derivative_adjoints, float* linear_adjoints, float* linear_derivative_adjoints, float* bias_gradients, const std::uint32_t width, const std::uint32_t sample_count, const std::uint32_t derivative_count) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t count = width * sample_count;
            if (index >= count) return;
            const float first  = activation_first_derivative(activation, frequency, linear[index], index % width);
            const float second = activation_second_derivative(activation, frequency, linear[index], index % width);
            float adjoint      = output_adjoints[index] * first;
            for (std::uint32_t derivative = 0u; derivative < derivative_count; ++derivative) {
                const std::size_t derivative_index = index + static_cast<std::size_t>(derivative) * count;
                const float derivative_adjoint     = output_derivative_adjoints[derivative_index];
                adjoint += derivative_adjoint * second * linear_derivatives[derivative_index];
                linear_derivative_adjoints[derivative_index] = derivative_adjoint * first;
            }
            linear_adjoints[index] = adjoint;
            atomicAdd(bias_gradients + index % width, adjoint);
        }

        __global__ void concatenate_forward_kernel(const float* first, const float* first_derivatives, const std::uint32_t first_width, const float first_scale, const float* second, const float* second_derivatives, const std::uint32_t second_width, const float second_scale, float* output, float* output_derivatives, const std::uint32_t sample_count, const std::uint32_t derivative_count) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            const std::uint32_t output_width = first_width + second_width;
            for (std::uint32_t feature = 0u; feature < first_width; ++feature) output[feature + static_cast<std::size_t>(sample) * output_width] = first[feature + static_cast<std::size_t>(sample) * first_width] * first_scale;
            for (std::uint32_t feature = 0u; feature < second_width; ++feature) output[first_width + feature + static_cast<std::size_t>(sample) * output_width] = second[feature + static_cast<std::size_t>(sample) * second_width] * second_scale;
            for (std::uint32_t derivative = 0u; derivative < derivative_count; ++derivative) {
                const std::size_t output_offset = static_cast<std::size_t>(derivative) * output_width * sample_count;
                const std::size_t first_offset  = static_cast<std::size_t>(derivative) * first_width * sample_count;
                const std::size_t second_offset = static_cast<std::size_t>(derivative) * second_width * sample_count;
                for (std::uint32_t feature = 0u; feature < first_width; ++feature) output_derivatives[output_offset + feature + static_cast<std::size_t>(sample) * output_width] = first_derivatives[first_offset + feature + static_cast<std::size_t>(sample) * first_width] * first_scale;
                for (std::uint32_t feature = 0u; feature < second_width; ++feature) output_derivatives[output_offset + first_width + feature + static_cast<std::size_t>(sample) * output_width] = second_derivatives[second_offset + feature + static_cast<std::size_t>(sample) * second_width] * second_scale;
            }
        }

        __global__ void split_adjoint_kernel(const float* source, const float* source_derivatives, const std::uint32_t first_width, const float first_scale, const std::uint32_t second_width, const float second_scale, float* first, float* first_derivatives, float* second, float* second_derivatives, const std::uint32_t sample_count, const std::uint32_t derivative_count) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            const std::uint32_t source_width = first_width + second_width;
            for (std::uint32_t feature = 0u; feature < first_width; ++feature) first[feature + static_cast<std::size_t>(sample) * first_width] = source[feature + static_cast<std::size_t>(sample) * source_width] * first_scale;
            for (std::uint32_t feature = 0u; feature < second_width; ++feature) atomicAdd(second + feature + static_cast<std::size_t>(sample) * second_width, source[first_width + feature + static_cast<std::size_t>(sample) * source_width] * second_scale);
            for (std::uint32_t derivative = 0u; derivative < derivative_count; ++derivative) {
                const std::size_t source_offset = static_cast<std::size_t>(derivative) * source_width * sample_count;
                const std::size_t first_offset  = static_cast<std::size_t>(derivative) * first_width * sample_count;
                const std::size_t second_offset = static_cast<std::size_t>(derivative) * second_width * sample_count;
                for (std::uint32_t feature = 0u; feature < first_width; ++feature) first_derivatives[first_offset + feature + static_cast<std::size_t>(sample) * first_width] = source_derivatives[source_offset + feature + static_cast<std::size_t>(sample) * source_width] * first_scale;
                for (std::uint32_t feature = 0u; feature < second_width; ++feature) atomicAdd(second_derivatives + second_offset + feature + static_cast<std::size_t>(sample) * second_width, source_derivatives[source_offset + first_width + feature + static_cast<std::size_t>(sample) * source_width] * second_scale);
            }
        }

        __global__ void add_kernel(float* destination, const float* source, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += source[index];
        }

        __global__ void scaled_add_kernel(float* destination, const float* source, const float scale, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] = fmaf(scale, source[index], destination[index]);
        }

        __global__ void adam_kernel(float* parameters, const float* gradients, float* first_moments, float* second_moments, const std::size_t count, const float learning_rate, const std::uint32_t step) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const float gradient = gradients[index];
            const float first = first_moments[index] = fmaf(0.1F, gradient, 0.9F * first_moments[index]);
            const float second = second_moments[index] = fmaf(0.001F, gradient * gradient, 0.999F * second_moments[index]);
            const float corrected_first                = first / (1.0F - powf(0.9F, static_cast<float>(step)));
            const float corrected_second               = second / (1.0F - powf(0.999F, static_cast<float>(step)));
            parameters[index] -= learning_rate * corrected_first / (sqrtf(corrected_second) + 1.0e-8F);
        }

    } // namespace

    void weight_normalization_forward(const ::cuda::stream_ref stream, const float* weights, const float* scales, float* normalized, const std::uint32_t input_width, const std::uint32_t output_width) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(output_width), weight_normalization_forward_kernel, weights, scales, normalized, input_width, output_width);
    }

    void weight_normalization_backward(const ::cuda::stream_ref stream, const float* weights, const float* scales, const float* normalized_gradients, float* weight_gradients, float* scale_gradients, const std::uint32_t input_width, const std::uint32_t output_width) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(output_width), weight_normalization_backward_kernel, weights, scales, normalized_gradients, weight_gradients, scale_gradients, input_width, output_width);
    }

    void activation_forward(const ::cuda::stream_ref stream, const Activation activation, const float frequency, const float* biases, float* linear, const float* linear_derivatives, float* output, float* output_derivatives, const std::uint32_t width, const std::uint32_t sample_count, const std::uint32_t derivative_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(width * sample_count), activation_forward_kernel, activation, frequency, biases, linear, linear_derivatives, output, output_derivatives, width, sample_count, derivative_count);
    }

    void activation_backward(const ::cuda::stream_ref stream, const Activation activation, const float frequency, const float* linear, const float* linear_derivatives, const float* output_adjoints, const float* output_derivative_adjoints, float* linear_adjoints, float* linear_derivative_adjoints, float* bias_gradients, const std::uint32_t width, const std::uint32_t sample_count, const std::uint32_t derivative_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(width * sample_count), activation_backward_kernel, activation, frequency, linear, linear_derivatives, output_adjoints, output_derivative_adjoints, linear_adjoints, linear_derivative_adjoints, bias_gradients, width, sample_count, derivative_count);
    }

    void concatenate_forward(const ::cuda::stream_ref stream, const float* first, const float* first_derivatives, const std::uint32_t first_width, const float first_scale, const float* second, const float* second_derivatives, const std::uint32_t second_width, const float second_scale, float* output, float* output_derivatives, const std::uint32_t sample_count, const std::uint32_t derivative_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), concatenate_forward_kernel, first, first_derivatives, first_width, first_scale, second, second_derivatives, second_width, second_scale, output, output_derivatives, sample_count, derivative_count);
    }

    void split_adjoint(const ::cuda::stream_ref stream, const float* source, const float* source_derivatives, const std::uint32_t first_width, const float first_scale, const std::uint32_t second_width, const float second_scale, float* first, float* first_derivatives, float* second, float* second_derivatives, const std::uint32_t sample_count, const std::uint32_t derivative_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), split_adjoint_kernel, source, source_derivatives, first_width, first_scale, second_width, second_scale, first, first_derivatives, second, second_derivatives, sample_count, derivative_count);
    }

    void add(const ::cuda::stream_ref stream, float* destination, const float* source, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(count), add_kernel, destination, source, count);
    }

    void scaled_add(const ::cuda::stream_ref stream, float* destination, const float* source, const float scale, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(count), scaled_add_kernel, destination, source, scale, count);
    }

    void adam(const ::cuda::stream_ref stream, float* parameters, const float* gradients, float* first_moments, float* second_moments, const std::size_t count, const float learning_rate, const std::uint32_t step) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(count), adam_kernel, parameters, gradients, first_moments, second_moments, count, learning_rate, step);
    }
} // namespace physica::reconstruction::pinfs::kernels
