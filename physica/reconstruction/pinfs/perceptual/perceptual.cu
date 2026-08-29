#include "kernels.h"
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/cmath>
#include <cuda/std/span>
#include <cuda_runtime.h>

namespace physica::reconstruction::pinfs::kernels {
    namespace {
        __global__ void normalize_kernel(const float* fine, const float* coarse, const float* target, float* output, const std::uint32_t width) {
            const std::uint32_t index             = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t image_value_count = width * width * 3u;
            if (index >= image_value_count * 3u) return;
            constexpr float means[]{0.485F, 0.456F, 0.406F};
            constexpr float deviations[]{0.229F, 0.224F, 0.225F};
            const std::uint32_t image_index = index % image_value_count;
            const std::uint32_t image       = index / image_value_count;
            const float value               = image == 0u ? target[image_index] : image == 1u ? fine[image_index] : coarse[image_index];
            output[index]                   = (value - means[image_index % 3u]) / deviations[image_index % 3u];
        }

        __global__ void im2col_kernel(const float* input, float* columns, const std::uint32_t channels, const std::uint32_t width, const bool replicate_padding) {
            const std::uint32_t index          = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t position_count = 3u * width * width;
            const std::uint32_t reduction      = channels * 9u;
            if (index >= position_count * reduction) return;
            const std::uint32_t position       = index / reduction;
            const std::uint32_t kernel_element = index % reduction;
            const std::uint32_t channel        = kernel_element % channels;
            const std::int32_t kernel_x        = static_cast<std::int32_t>((kernel_element / channels) % 3u) - 1;
            const std::int32_t kernel_y        = static_cast<std::int32_t>(kernel_element / channels / 3u) - 1;
            const std::uint32_t image          = position / (width * width);
            const std::uint32_t pixel          = position % (width * width);
            std::int32_t x                     = static_cast<std::int32_t>(pixel % width) + kernel_x;
            std::int32_t y                     = static_cast<std::int32_t>(pixel / width) + kernel_y;
            if (replicate_padding) {
                x = max(0, min(static_cast<std::int32_t>(width) - 1, x));
                y = max(0, min(static_cast<std::int32_t>(width) - 1, y));
            }
            columns[index] = x < 0 || y < 0 || x >= static_cast<std::int32_t>(width) || y >= static_cast<std::int32_t>(width) ? 0.0F : input[(image * width * width + static_cast<std::uint32_t>(y) * width + static_cast<std::uint32_t>(x)) * channels + channel];
        }

        __global__ void bias_relu_kernel(float* activation, const float* biases, const std::uint32_t channels, const std::uint32_t width, const bool relu) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t count = 3u * width * width * channels;
            if (index >= count) return;
            if (relu) activation[index] = fmaxf(activation[index], 0.0F);
            else activation[index] += biases[index % channels];
        }

        __global__ void pool_kernel(const float* input, float* output, std::uint8_t* indices, const std::uint32_t channels, const std::uint32_t width) {
            const std::uint32_t index        = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t output_width = width / 2u;
            const std::uint32_t count        = 3u * output_width * output_width * channels;
            if (index >= count) return;
            const std::uint32_t channel  = index % channels;
            const std::uint32_t position = index / channels;
            const std::uint32_t image    = position / (output_width * output_width);
            const std::uint32_t pixel    = position % (output_width * output_width);
            const std::uint32_t x        = (pixel % output_width) * 2u;
            const std::uint32_t y        = (pixel / output_width) * 2u;
            float maximum                = input[(image * width * width + y * width + x) * channels + channel];
            std::uint8_t maximum_index{};
            for (std::uint8_t offset = 1u; offset < 4u; ++offset) {
                const float value = input[(image * width * width + (y + offset / 2u) * width + x + offset % 2u) * channels + channel];
                if (value <= maximum) continue;
                maximum       = value;
                maximum_index = offset;
            }
            output[index]  = maximum;
            indices[index] = maximum_index;
        }

        __global__ void feature_loss_kernel(const float* features, float* feature_adjoints, double* loss, const std::uint32_t channels, const std::uint32_t width, const float weight) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= 2u * channels * width) return;
            const std::uint32_t predicted_image = row / (channels * width) + 1u;
            const std::uint32_t feature_row     = row % (channels * width);
            const std::uint32_t channel         = feature_row / width;
            const std::uint32_t y               = feature_row % width;
            const std::size_t image_value_count = static_cast<std::size_t>(channels) * width * width;
            float reference_squared_norm        = 1.0e-12F;
            float predicted_squared_norm        = 1.0e-12F;
            float product{};
            for (std::uint32_t x = 0u; x < width; ++x) {
                const std::size_t reference_index = (static_cast<std::size_t>(y) * width + x) * channels + channel;
                const std::size_t predicted_index = static_cast<std::size_t>(predicted_image) * image_value_count + reference_index;
                reference_squared_norm            = fmaf(features[reference_index], features[reference_index], reference_squared_norm);
                predicted_squared_norm            = fmaf(features[predicted_index], features[predicted_index], predicted_squared_norm);
                product                           = fmaf(features[reference_index], features[predicted_index], product);
            }
            const float reference_norm = sqrtf(reference_squared_norm);
            const float predicted_norm = sqrtf(predicted_squared_norm);
            const float inverse_count  = 1.0F / static_cast<float>(channels * width);
            const float scale          = -weight * inverse_count / (reference_norm * predicted_norm);
            const float projection     = product / predicted_squared_norm;
            for (std::uint32_t x = 0u; x < width; ++x) {
                const std::size_t reference_index = (static_cast<std::size_t>(y) * width + x) * channels + channel;
                const std::size_t predicted_index = static_cast<std::size_t>(predicted_image) * image_value_count + reference_index;
                feature_adjoints[predicted_index] = scale * (features[reference_index] - projection * features[predicted_index]);
            }
            atomicAdd(loss, static_cast<double>(weight * inverse_count * (1.0F - product / (reference_norm * predicted_norm))));
        }

        __global__ void add_kernel(float* destination, const float* source, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += source[index];
        }

        __global__ void relu_backward_kernel(const float* activation, float* adjoints, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count && activation[index] <= 0.0F) adjoints[index] = 0.0F;
        }

        __global__ void pool_backward_kernel(const float* output_adjoints, const std::uint8_t* indices, float* input_adjoints, const std::uint32_t channels, const std::uint32_t width) {
            const std::uint32_t index        = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t output_width = width / 2u;
            const std::uint32_t count        = 3u * output_width * output_width * channels;
            if (index >= count) return;
            const std::uint32_t channel                                                  = index % channels;
            const std::uint32_t position                                                 = index / channels;
            const std::uint32_t image                                                    = position / (output_width * output_width);
            const std::uint32_t pixel                                                    = position % (output_width * output_width);
            const std::uint32_t x                                                        = (pixel % output_width) * 2u + indices[index] % 2u;
            const std::uint32_t y                                                        = (pixel / output_width) * 2u + indices[index] / 2u;
            input_adjoints[(image * width * width + y * width + x) * channels + channel] = output_adjoints[index];
        }

        __global__ void col2im_kernel(const float* columns, float* input, const std::uint32_t channels, const std::uint32_t width, const bool replicate_padding) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t count = 3u * width * width * channels;
            if (index >= count) return;
            const std::uint32_t channel  = index % channels;
            const std::uint32_t position = index / channels;
            const std::uint32_t image    = position / (width * width);
            const std::uint32_t pixel    = position % (width * width);
            const std::int32_t x         = static_cast<std::int32_t>(pixel % width);
            const std::int32_t y         = static_cast<std::int32_t>(pixel / width);
            float result{};
            for (std::int32_t kernel_y = -1; kernel_y <= 1; ++kernel_y) {
                for (std::int32_t kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                    std::int32_t output_x = x - kernel_x;
                    std::int32_t output_y = y - kernel_y;
                    if (!replicate_padding && (output_x < 0 || output_y < 0 || output_x >= static_cast<std::int32_t>(width) || output_y >= static_cast<std::int32_t>(width))) continue;
                    if (replicate_padding) {
                        for (std::int32_t candidate_y = max(0, y - 1); candidate_y <= min(static_cast<std::int32_t>(width) - 1, y + 1); ++candidate_y) {
                            for (std::int32_t candidate_x = max(0, x - 1); candidate_x <= min(static_cast<std::int32_t>(width) - 1, x + 1); ++candidate_x) {
                                const std::int32_t mapped_x = max(0, min(static_cast<std::int32_t>(width) - 1, candidate_x + kernel_x));
                                const std::int32_t mapped_y = max(0, min(static_cast<std::int32_t>(width) - 1, candidate_y + kernel_y));
                                if (mapped_x != x || mapped_y != y) continue;
                                const std::uint32_t output_position = image * width * width + static_cast<std::uint32_t>(candidate_y) * width + static_cast<std::uint32_t>(candidate_x);
                                const std::uint32_t reduction_index = static_cast<std::uint32_t>((kernel_y + 1) * 3 + kernel_x + 1) * channels + channel;
                                result += columns[static_cast<std::size_t>(output_position) * channels * 9u + reduction_index];
                            }
                        }
                        continue;
                    }
                    const std::uint32_t output_position = image * width * width + static_cast<std::uint32_t>(output_y) * width + static_cast<std::uint32_t>(output_x);
                    const std::uint32_t reduction_index = static_cast<std::uint32_t>((kernel_y + 1) * 3 + kernel_x + 1) * channels + channel;
                    result += columns[static_cast<std::size_t>(output_position) * channels * 9u + reduction_index];
                }
            }
            input[index] = result;
        }

        __global__ void input_backward_kernel(const float* normalized_adjoints, float* fine_adjoints, float* coarse_adjoints, const std::uint32_t width) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t count = width * width * 3u;
            if (index >= count) return;
            constexpr float deviations[]{0.229F, 0.224F, 0.225F};
            fine_adjoints[index]   = normalized_adjoints[count + index] / deviations[index % 3u];
            coarse_adjoints[index] = normalized_adjoints[2u * count + index] / deviations[index % 3u];
        }
    } // namespace

    void vgg_normalize(const ::cuda::stream_ref stream, const float* fine, const float* coarse, const float* target, float* output, const std::uint32_t width) {
        const std::uint32_t count = width * width * 9u;
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), normalize_kernel, fine, coarse, target, output, width);
    }

    void vgg_im2col(const ::cuda::stream_ref stream, const float* input, float* columns, const std::uint32_t channels, const std::uint32_t width, const bool replicate_padding) {
        const std::uint32_t count = 3u * width * width * channels * 9u;
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), im2col_kernel, input, columns, channels, width, replicate_padding);
    }

    void vgg_bias_relu(const ::cuda::stream_ref stream, float* activation, const float* biases, const std::uint32_t channels, const std::uint32_t width, const bool relu) {
        const std::uint32_t count = 3u * width * width * channels;
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), bias_relu_kernel, activation, biases, channels, width, relu);
    }

    void vgg_pool(const ::cuda::stream_ref stream, const float* input, float* output, std::uint8_t* indices, const std::uint32_t channels, const std::uint32_t width) {
        const std::uint32_t count = 3u * width * width * channels / 4u;
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), pool_kernel, input, output, indices, channels, width);
    }

    void vgg_feature_loss(const ::cuda::stream_ref stream, const float* features, float* feature_adjoints, double* loss, const std::uint32_t channels, const std::uint32_t width, const float weight) {
        ::cuda::launch(stream, ::cuda::distribute<256u>(2u * channels * width), feature_loss_kernel, features, feature_adjoints, loss, channels, width, weight);
    }

    void vgg_add(const ::cuda::stream_ref stream, float* destination, const float* source, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), add_kernel, destination, source, count);
    }

    void vgg_relu_backward(const ::cuda::stream_ref stream, const float* activation, float* adjoints, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), relu_backward_kernel, activation, adjoints, count);
    }

    void vgg_pool_backward(const ::cuda::stream_ref stream, const float* output_adjoints, const std::uint8_t* indices, float* input_adjoints, const std::uint32_t channels, const std::uint32_t width) {
        const std::uint32_t output_count = 3u * width * width * channels / 4u;
        const std::uint32_t input_count  = 3u * width * width * channels;
        ::cuda::fill_bytes(stream, ::cuda::std::span<float>{input_adjoints, input_count}, 0u);
        ::cuda::launch(stream, ::cuda::distribute<256u>(output_count), pool_backward_kernel, output_adjoints, indices, input_adjoints, channels, width);
    }

    void vgg_col2im(const ::cuda::stream_ref stream, const float* columns, float* input, const std::uint32_t channels, const std::uint32_t width, const bool replicate_padding) {
        const std::uint32_t count = 3u * width * width * channels;
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), col2im_kernel, columns, input, channels, width, replicate_padding);
    }

    void vgg_input_backward(const ::cuda::stream_ref stream, const float* normalized_adjoints, float* fine_adjoints, float* coarse_adjoints, const std::uint32_t width) {
        const std::uint32_t count = width * width * 3u;
        ::cuda::launch(stream, ::cuda::distribute<256u>(count), input_backward_kernel, normalized_adjoints, fine_adjoints, coarse_adjoints, width);
    }
} // namespace physica::reconstruction::pinfs::kernels
