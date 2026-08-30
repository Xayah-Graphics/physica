#ifndef PHYSICA_NEURAL_TRANSFORMER_KERNELS_H
#define PHYSICA_NEURAL_TRANSFORMER_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::neural::kernels {
    void adaln_forward(::cuda::stream_ref stream, const float* input, const float* modulation, float* output, float* means, float* inverse_standard_deviations, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width, std::uint32_t modulation_group);
    void residual_forward(::cuda::stream_ref stream, const float* input, const float* branch, const float* modulation, float* output, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width, std::uint32_t modulation_group);
    void residual_backward(::cuda::stream_ref stream, const float* output_gradient, const float* branch, const float* modulation, float* branch_gradient, float* modulation_gradient, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width, std::uint32_t modulation_group);
    void adaln_backward(::cuda::stream_ref stream, const float* input, const float* modulation, const float* output_gradient, const float* residual_gradient, const float* means, const float* inverse_standard_deviations, float* input_gradient, float* modulation_gradient, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width, std::uint32_t modulation_group);
    void sdpa_forward(::cuda::stream_ref stream, const float* qkv, float* output, float* log_sum_exp, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width, std::uint32_t head_count);
    void sdpa_backward(::cuda::stream_ref stream, const float* qkv, const float* output, const float* output_gradient, const float* log_sum_exp, float* delta, float* qkv_gradient, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width, std::uint32_t head_count);
} // namespace physica::neural::kernels

#endif
