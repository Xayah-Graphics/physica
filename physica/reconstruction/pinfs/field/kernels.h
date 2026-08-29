#ifndef PHYSICA_RECONSTRUCTION_PINFS_FIELD_KERNELS_H
#define PHYSICA_RECONSTRUCTION_PINFS_FIELD_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <physica/cuda_stream.h>
#include <physica/math.h>

namespace physica::reconstruction::pinfs::kernels {
    void extract_density(::cuda::stream_ref stream, const float* field, float* density, std::uint32_t sample_count);
    void positional_encoding(::cuda::stream_ref stream, const Vector3<float>* input, const float* weights, float* output, float* output_derivatives, std::uint32_t sample_count, std::uint32_t frequency_count);
    void static_color_input(::cuda::stream_ref stream, const Vector3<float>* positions, const Vector3<float>* directions, const float* sdf_output, const float* sdf_derivatives, float* color_input, std::uint32_t sample_count);
    void static_sdf_adjoint(::cuda::stream_ref stream, const float* color_input_adjoints, const float* source_sdf_adjoints, const float* source_gradient_adjoints, float* output_adjoints, float* output_derivative_adjoints, std::uint32_t sample_count);
    void inverse_deviation_forward(::cuda::stream_ref stream, const float* deviation, float* inverse_deviation);
    void inverse_deviation_backward(::cuda::stream_ref stream, const float* inverse_deviation, const float* inverse_deviation_adjoint, float* deviation_gradient);
} // namespace physica::reconstruction::pinfs::kernels

#endif
