#ifndef PHYSICA_RECONSTRUCTION_PINFS_NETWORK_KERNELS_H
#define PHYSICA_RECONSTRUCTION_PINFS_NETWORK_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::reconstruction::pinfs::kernels {
    enum class Activation : std::uint8_t {
        linear,
        sine,
        relu,
        sigmoid,
        softplus,
        rgb_density,
    };

    void weight_normalization_forward(::cuda::stream_ref stream, const float* weights, const float* scales, float* normalized, std::uint32_t input_width, std::uint32_t output_width);
    void weight_normalization_backward(::cuda::stream_ref stream, const float* weights, const float* scales, const float* normalized_gradients, float* weight_gradients, float* scale_gradients, std::uint32_t input_width, std::uint32_t output_width);
    void activation_forward(::cuda::stream_ref stream, Activation activation, float frequency, const float* biases, float* linear, const float* linear_derivatives, float* output, float* output_derivatives, std::uint32_t width, std::uint32_t sample_count, std::uint32_t derivative_count);
    void activation_backward(::cuda::stream_ref stream, Activation activation, float frequency, const float* linear, const float* linear_derivatives, const float* output_adjoints, const float* output_derivative_adjoints, float* linear_adjoints, float* linear_derivative_adjoints, float* bias_gradients, std::uint32_t width, std::uint32_t sample_count, std::uint32_t derivative_count);
    void concatenate_forward(::cuda::stream_ref stream, const float* first, const float* first_derivatives, std::uint32_t first_width, float first_scale, const float* second, const float* second_derivatives, std::uint32_t second_width, float second_scale, float* output, float* output_derivatives, std::uint32_t sample_count, std::uint32_t derivative_count);
    void split_adjoint(::cuda::stream_ref stream, const float* source, const float* source_derivatives, std::uint32_t first_width, float first_scale, std::uint32_t second_width, float second_scale, float* first, float* first_derivatives, float* second, float* second_derivatives, std::uint32_t sample_count, std::uint32_t derivative_count);
    void add(::cuda::stream_ref stream, float* destination, const float* source, std::size_t count);
    void scaled_add(::cuda::stream_ref stream, float* destination, const float* source, float scale, std::size_t count);
    void adam(::cuda::stream_ref stream, float* parameters, const float* gradients, float* first_moments, float* second_moments, std::size_t count, float learning_rate, std::uint32_t step);
} // namespace physica::reconstruction::pinfs::kernels

#endif
