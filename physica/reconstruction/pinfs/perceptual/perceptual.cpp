module;

#include "kernels.h"
#include <cublas_v2.h>
#include <physica/cuda.h>

module physica.reconstruction.pinfs.perceptual;

import std;

namespace physica::reconstruction::pinfs {
    namespace {
        void matrix_product(const cublasHandle_t handle, const cublasOperation_t first_operation, const cublasOperation_t second_operation, const std::uint32_t rows, const std::uint32_t columns, const std::uint32_t reduction, const float* first, const std::uint32_t first_leading_dimension, const float* second, const std::uint32_t second_leading_dimension, float* output, const std::uint32_t output_leading_dimension) {
            constexpr float alpha{1.0F};
            constexpr float beta{};
            const cublasStatus_t status = cublasSgemm(handle, first_operation, second_operation, static_cast<int>(rows), static_cast<int>(columns), static_cast<int>(reduction), &alpha, first, static_cast<int>(first_leading_dimension), second, static_cast<int>(second_leading_dimension), &beta, output, static_cast<int>(output_leading_dimension));
            if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cublasSgemm failed: {}", cublasGetStatusString(status))};
        }
    } // namespace

    PerceptualLoss::PerceptualLoss(const ::cuda::stream_ref source_stream, const std::uint32_t source_image_width, const std::filesystem::path& weights_path) : stream{source_stream}, image_width{source_image_width}, normalized_input{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(image_width) * image_width * 9uz, ::cuda::no_init}, column_workspace{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(image_width) * image_width * 3uz * 64uz * 9uz, ::cuda::no_init}, adjoints_a{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(image_width) * image_width * 192uz, ::cuda::no_init}, adjoints_b{stream, ::cuda::device_default_memory_pool(stream.device()), adjoints_a.size(), ::cuda::no_init}, loss{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init} {
        if (const cublasStatus_t status = cublasCreate(std::out_ptr(cublas)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cublasCreate failed: {}", cublasGetStatusString(status))};
        if (const cublasStatus_t status = cublasSetStream(cublas.get(), stream.get()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cublasSetStream failed: {}", cublasGetStatusString(status))};

        constexpr std::array<std::uint32_t, 13> input_channels{3u, 64u, 64u, 128u, 128u, 256u, 256u, 256u, 256u, 512u, 512u, 512u, 512u};
        constexpr std::array<std::uint32_t, 13> output_channels{64u, 64u, 128u, 128u, 256u, 256u, 256u, 256u, 512u, 512u, 512u, 512u, 512u};
        constexpr std::array<bool, 13> pools{false, true, false, true, false, false, false, true, false, false, false, true, false};
        constexpr std::array<bool, 13> captures{false, false, true, false, true, false, false, false, true, false, false, false, true};
        std::uint32_t width = image_width;
        for (const std::size_t index : std::views::iota(0uz, input_channels.size())) {
            convolutions.emplace_back(stream, input_channels[index], output_channels[index], width, pools[index], captures[index]);
            if (pools[index]) width /= 2u;
        }

        std::ifstream input{weights_path, std::ios::binary};
        std::array<char, 8> magic;
        std::uint32_t version;
        input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        for (Convolution& convolution : convolutions) {
            std::uint32_t stored_input_channels;
            std::uint32_t stored_output_channels;
            input.read(reinterpret_cast<char*>(&stored_input_channels), sizeof(stored_input_channels));
            input.read(reinterpret_cast<char*>(&stored_output_channels), sizeof(stored_output_channels));
            std::vector<float> host_weights(convolution.weights.size());
            std::vector<float> host_biases(convolution.biases.size());
            input.read(reinterpret_cast<char*>(host_weights.data()), static_cast<std::streamsize>(host_weights.size() * sizeof(float)));
            input.read(reinterpret_cast<char*>(host_biases.data()), static_cast<std::streamsize>(host_biases.size() * sizeof(float)));
            ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{host_weights.data(), host_weights.size()}, convolution.weights);
            ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{host_biases.data(), host_biases.size()}, convolution.biases);
        }
    }

    PerceptualLoss::~PerceptualLoss() noexcept = default;

    float PerceptualLoss::loss_and_backward(const Vector3<float>* fine, const Vector3<float>* coarse, const Vector3<float>* target, const std::array<float, 4>& layer_weights, Vector3<float>* fine_adjoints, Vector3<float>* coarse_adjoints) {
        ::cuda::fill_bytes(stream, loss, 0u);
        kernels::vgg_normalize(stream, reinterpret_cast<const float*>(fine), reinterpret_cast<const float*>(coarse), reinterpret_cast<const float*>(target), normalized_input.data(), image_width);
        const float* input = normalized_input.data();
        std::size_t capture_index{};
        for (const std::size_t index : std::views::iota(0uz, convolutions.size())) {
            Convolution& convolution = convolutions[index];
            kernels::vgg_im2col(stream, input, column_workspace.data(), convolution.input_channels, convolution.width, index == 0uz);
            matrix_product(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, convolution.output_channels, 3u * convolution.width * convolution.width, convolution.input_channels * 9u, convolution.weights.data(), convolution.output_channels, column_workspace.data(), convolution.input_channels * 9u, convolution.activation.data(), convolution.output_channels);
            kernels::vgg_bias_relu(stream, convolution.activation.data(), convolution.biases.data(), convolution.output_channels, convolution.width, false);
            if (convolution.capture) {
                ::cuda::fill_bytes(stream, convolution.capture_adjoints, 0u);
                kernels::vgg_feature_loss(stream, convolution.activation.data(), convolution.capture_adjoints.data(), loss.data(), convolution.output_channels, convolution.width, layer_weights[capture_index++]);
            }
            if (index + 1uz != convolutions.size()) kernels::vgg_bias_relu(stream, convolution.activation.data(), convolution.biases.data(), convolution.output_channels, convolution.width, true);
            if (convolution.pool) {
                kernels::vgg_pool(stream, convolution.activation.data(), convolution.pooled.data(), convolution.pool_indices.data(), convolution.output_channels, convolution.width);
                input = convolution.pooled.data();
            } else input = convolution.activation.data();
        }

        ::cuda::fill_bytes(stream, adjoints_a, 0u);
        float* current = adjoints_a.data();
        float* next    = adjoints_b.data();
        for (std::size_t index = convolutions.size(); index-- > 0uz;) {
            Convolution& convolution = convolutions[index];
            if (convolution.pool) {
                kernels::vgg_pool_backward(stream, current, convolution.pool_indices.data(), next, convolution.output_channels, convolution.width);
                std::swap(current, next);
            }
            if (index + 1uz != convolutions.size()) kernels::vgg_relu_backward(stream, convolution.activation.data(), current, convolution.activation.size());
            if (convolution.capture) kernels::vgg_add(stream, current, convolution.capture_adjoints.data(), convolution.activation.size());
            matrix_product(cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N, convolution.input_channels * 9u, 3u * convolution.width * convolution.width, convolution.output_channels, convolution.weights.data(), convolution.output_channels, current, convolution.output_channels, column_workspace.data(), convolution.input_channels * 9u);
            kernels::vgg_col2im(stream, column_workspace.data(), next, convolution.input_channels, convolution.width, index == 0uz);
            std::swap(current, next);
        }
        kernels::vgg_input_backward(stream, current, reinterpret_cast<float*>(fine_adjoints), reinterpret_cast<float*>(coarse_adjoints), image_width);
        double host_loss{};
        ::cuda::copy_bytes(stream, loss, ::cuda::std::span<double>{&host_loss, 1uz});
        stream.sync();
        return static_cast<float>(host_loss);
    }

    PerceptualLoss::Convolution::Convolution(const ::cuda::stream_ref stream, const std::uint32_t source_input_channels, const std::uint32_t source_output_channels, const std::uint32_t source_width, const bool source_pool, const bool source_capture)
        : input_channels{source_input_channels}, output_channels{source_output_channels}, width{source_width}, pool{source_pool}, capture{source_capture}, weights{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(input_channels) * output_channels * 9uz, ::cuda::no_init}, biases{stream, ::cuda::device_default_memory_pool(stream.device()), output_channels, ::cuda::no_init}, activation{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(3u) * width * width * output_channels, ::cuda::no_init}, pooled{stream, ::cuda::device_default_memory_pool(stream.device()), pool ? activation.size() / 4uz : 0uz, ::cuda::no_init}, pool_indices{stream, ::cuda::device_default_memory_pool(stream.device()), pooled.size(), ::cuda::no_init}, capture_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), capture ? activation.size() : 0uz, ::cuda::no_init} {}

    void PerceptualLoss::CublasDeleter::operator()(cublasContext* handle) const noexcept {
        cublasDestroy(handle);
    }
} // namespace physica::reconstruction::pinfs
