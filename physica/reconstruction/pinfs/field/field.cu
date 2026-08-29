#include "kernels.h"
#include <cuda/launch>
#include <cuda_runtime.h>

namespace physica::reconstruction::pinfs::kernels {
    namespace {
        __global__ void extract_density_kernel(const float* field, float* density, const std::uint32_t sample_count) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample < sample_count) density[sample] = field[static_cast<std::size_t>(sample) * 4u + 3u];
        }

        __global__ void positional_encoding_kernel(const Vector3<float>* input, const float* weights, float* output, float* output_derivatives, const std::uint32_t sample_count, const std::uint32_t frequency_count) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            const std::uint32_t output_width = 3u * (1u + 2u * frequency_count);
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const float value                                                   = input[sample][component];
                output[component + static_cast<std::size_t>(sample) * output_width] = value * weights[0];
                if (output_derivatives != nullptr)
                    for (std::uint32_t derivative = 0u; derivative < 3u; ++derivative) output_derivatives[component + static_cast<std::size_t>(sample) * output_width + static_cast<std::size_t>(derivative) * output_width * sample_count] = component == derivative ? weights[0] : 0.0F;
                for (std::uint32_t frequency_index = 0u; frequency_index < frequency_count; ++frequency_index) {
                    const float frequency                                                 = exp2f(static_cast<float>(frequency_index));
                    const std::uint32_t sin_feature                                       = 3u + frequency_index * 6u + component;
                    const std::uint32_t cos_feature                                       = sin_feature + 3u;
                    output[sin_feature + static_cast<std::size_t>(sample) * output_width] = sinf(frequency * value) * weights[frequency_index + 1u];
                    output[cos_feature + static_cast<std::size_t>(sample) * output_width] = cosf(frequency * value) * weights[frequency_index + 1u];
                    if (output_derivatives != nullptr)
                        for (std::uint32_t derivative = 0u; derivative < 3u; ++derivative) {
                            output_derivatives[sin_feature + static_cast<std::size_t>(sample) * output_width + static_cast<std::size_t>(derivative) * output_width * sample_count] = component == derivative ? frequency * cosf(frequency * value) * weights[frequency_index + 1u] : 0.0F;
                            output_derivatives[cos_feature + static_cast<std::size_t>(sample) * output_width + static_cast<std::size_t>(derivative) * output_width * sample_count] = component == derivative ? -frequency * sinf(frequency * value) * weights[frequency_index + 1u] : 0.0F;
                        }
                }
            }
        }

        __global__ void static_color_input_kernel(const Vector3<float>* positions, const Vector3<float>* directions, const float* sdf_output, const float* sdf_derivatives, float* color_input, const std::uint32_t sample_count) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            constexpr std::uint32_t color_width = 265u;
            constexpr std::uint32_t sdf_width   = 257u;
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                color_input[component + static_cast<std::size_t>(sample) * color_width]      = positions[sample][component];
                color_input[3u + component + static_cast<std::size_t>(sample) * color_width] = directions[sample][component];
                color_input[6u + component + static_cast<std::size_t>(sample) * color_width] = sdf_derivatives[static_cast<std::size_t>(component) * sdf_width * sample_count + static_cast<std::size_t>(sample) * sdf_width];
            }
            for (std::uint32_t feature = 0u; feature < 256u; ++feature) color_input[9u + feature + static_cast<std::size_t>(sample) * color_width] = sdf_output[1u + feature + static_cast<std::size_t>(sample) * sdf_width];
        }

        __global__ void static_sdf_adjoint_kernel(const float* color_input_adjoints, const float* source_sdf_adjoints, const float* source_gradient_adjoints, float* output_adjoints, float* output_derivative_adjoints, const std::uint32_t sample_count) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            constexpr std::uint32_t color_width                           = 265u;
            constexpr std::uint32_t sdf_width                             = 257u;
            output_adjoints[static_cast<std::size_t>(sample) * sdf_width] = source_sdf_adjoints[sample];
            for (std::uint32_t feature = 0u; feature < 256u; ++feature) output_adjoints[1u + feature + static_cast<std::size_t>(sample) * sdf_width] = color_input_adjoints[9u + feature + static_cast<std::size_t>(sample) * color_width];
            for (std::uint32_t derivative = 0u; derivative < 3u; ++derivative) {
                const std::size_t derivative_offset                                                          = static_cast<std::size_t>(derivative) * sdf_width * sample_count;
                output_derivative_adjoints[derivative_offset + static_cast<std::size_t>(sample) * sdf_width] = color_input_adjoints[6u + derivative + static_cast<std::size_t>(sample) * color_width] + source_gradient_adjoints[static_cast<std::size_t>(derivative) * sample_count + sample];
                for (std::uint32_t feature = 1u; feature < sdf_width; ++feature) output_derivative_adjoints[derivative_offset + feature + static_cast<std::size_t>(sample) * sdf_width] = 0.0F;
            }
        }

        __global__ void inverse_deviation_forward_kernel(const float* deviation, float* inverse_deviation) {
            inverse_deviation[0] = fminf(fmaxf(expf(deviation[0] * 10.0F), 1.0e-6F), 1.0e6F);
        }

        __global__ void inverse_deviation_backward_kernel(const float* inverse_deviation, const float* inverse_deviation_adjoint, float* deviation_gradient) {
            deviation_gradient[0] += inverse_deviation_adjoint[0] * inverse_deviation[0] * 10.0F;
        }

    } // namespace

    void extract_density(const ::cuda::stream_ref stream, const float* field, float* density, const std::uint32_t sample_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), extract_density_kernel, field, density, sample_count);
    }

    void positional_encoding(const ::cuda::stream_ref stream, const Vector3<float>* input, const float* weights, float* output, float* output_derivatives, const std::uint32_t sample_count, const std::uint32_t frequency_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), positional_encoding_kernel, input, weights, output, output_derivatives, sample_count, frequency_count);
    }

    void static_color_input(const ::cuda::stream_ref stream, const Vector3<float>* positions, const Vector3<float>* directions, const float* sdf_output, const float* sdf_derivatives, float* color_input, const std::uint32_t sample_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), static_color_input_kernel, positions, directions, sdf_output, sdf_derivatives, color_input, sample_count);
    }

    void static_sdf_adjoint(const ::cuda::stream_ref stream, const float* color_input_adjoints, const float* source_sdf_adjoints, const float* source_gradient_adjoints, float* output_adjoints, float* output_derivative_adjoints, const std::uint32_t sample_count) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), static_sdf_adjoint_kernel, color_input_adjoints, source_sdf_adjoints, source_gradient_adjoints, output_adjoints, output_derivative_adjoints, sample_count);
    }

    void inverse_deviation_forward(const ::cuda::stream_ref stream, const float* deviation, float* inverse_deviation) {
        ::cuda::launch(stream, ::cuda::distribute<1u>(1u), inverse_deviation_forward_kernel, deviation, inverse_deviation);
    }

    void inverse_deviation_backward(const ::cuda::stream_ref stream, const float* inverse_deviation, const float* inverse_deviation_adjoint, float* deviation_gradient) {
        ::cuda::launch(stream, ::cuda::distribute<1u>(1u), inverse_deviation_backward_kernel, inverse_deviation, inverse_deviation_adjoint, deviation_gradient);
    }

} // namespace physica::reconstruction::pinfs::kernels
