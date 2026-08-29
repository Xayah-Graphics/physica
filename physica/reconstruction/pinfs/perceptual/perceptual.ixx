module;

#include <cublas_v2.h>
#include <physica/cuda.h>

export module physica.reconstruction.pinfs.perceptual;

import std;
export import physica.math;

export namespace physica::reconstruction::pinfs {
    struct PerceptualLoss final {
        PerceptualLoss(::cuda::stream_ref stream, std::uint32_t image_width, const std::filesystem::path& weights_path);
        ~PerceptualLoss() noexcept;

        PerceptualLoss(const PerceptualLoss&)            = delete;
        PerceptualLoss& operator=(const PerceptualLoss&) = delete;
        PerceptualLoss(PerceptualLoss&&)                 = delete;
        PerceptualLoss& operator=(PerceptualLoss&&)      = delete;

        float loss_and_backward(const Vector3<float>* fine, const Vector3<float>* coarse, const Vector3<float>* target, const std::array<float, 4>& layer_weights, Vector3<float>* fine_adjoints, Vector3<float>* coarse_adjoints);

    private:
        struct Convolution final {
            Convolution(::cuda::stream_ref stream, std::uint32_t input_channels, std::uint32_t output_channels, std::uint32_t width, bool pool, bool capture);

            std::uint32_t input_channels{};
            std::uint32_t output_channels{};
            std::uint32_t width{};
            bool pool{};
            bool capture{};
            ::cuda::device_buffer<float> weights;
            ::cuda::device_buffer<float> biases;
            ::cuda::device_buffer<float> activation;
            ::cuda::device_buffer<float> pooled;
            ::cuda::device_buffer<std::uint8_t> pool_indices;
            ::cuda::device_buffer<float> capture_adjoints;
        };

        struct CublasDeleter final {
            void operator()(cublasContext* handle) const noexcept;
        };

        ::cuda::stream_ref stream;
        std::uint32_t image_width{};
        std::unique_ptr<cublasContext, CublasDeleter> cublas;
        std::vector<Convolution> convolutions;
        ::cuda::device_buffer<float> normalized_input;
        ::cuda::device_buffer<float> column_workspace;
        ::cuda::device_buffer<float> adjoints_a;
        ::cuda::device_buffer<float> adjoints_b;
        ::cuda::device_buffer<double> loss;
    };
} // namespace physica::reconstruction::pinfs
