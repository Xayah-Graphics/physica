#ifndef PHYSICA_RECONSTRUCTION_PINFS_PERCEPTUAL_KERNELS_H
#define PHYSICA_RECONSTRUCTION_PINFS_PERCEPTUAL_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::reconstruction::pinfs::kernels {
    void vgg_normalize(::cuda::stream_ref stream, const float* fine, const float* coarse, const float* target, float* output, std::uint32_t width);
    void vgg_im2col(::cuda::stream_ref stream, const float* input, float* columns, std::uint32_t channels, std::uint32_t width, bool replicate_padding);
    void vgg_bias_relu(::cuda::stream_ref stream, float* activation, const float* biases, std::uint32_t channels, std::uint32_t width, bool relu);
    void vgg_pool(::cuda::stream_ref stream, const float* input, float* output, std::uint8_t* indices, std::uint32_t channels, std::uint32_t width);
    void vgg_feature_loss(::cuda::stream_ref stream, const float* features, float* feature_adjoints, double* loss, std::uint32_t channels, std::uint32_t width, float weight);
    void vgg_add(::cuda::stream_ref stream, float* destination, const float* source, std::size_t count);
    void vgg_relu_backward(::cuda::stream_ref stream, const float* activation, float* adjoints, std::size_t count);
    void vgg_pool_backward(::cuda::stream_ref stream, const float* output_adjoints, const std::uint8_t* indices, float* input_adjoints, std::uint32_t channels, std::uint32_t width);
    void vgg_col2im(::cuda::stream_ref stream, const float* columns, float* input, std::uint32_t channels, std::uint32_t width, bool replicate_padding);
    void vgg_input_backward(::cuda::stream_ref stream, const float* normalized_adjoints, float* fine_adjoints, float* coarse_adjoints, std::uint32_t width);
} // namespace physica::reconstruction::pinfs::kernels

#endif
