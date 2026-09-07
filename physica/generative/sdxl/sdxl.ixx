module;
#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <physica/cuda.h>

export module physica.generative.sdxl;

import std;
import physica.neural.inference_runtime;
import physica.generative.sdxl.inputs;
import physica.generative.sdxl.weights;
import physica.generative.sdxl.workspace;
import physica.generative.sdxl.text;
import physica.generative.sdxl.unet;
import physica.generative.sdxl.vae;

export namespace physica::generative::sdxl {
    struct Parameters final {
        std::string positive;
        std::string negative;
        int width{1024};
        int height{1536};
        int steps{50};
        float cfg{4.5F};
    };

    struct Model final {
        Model(::cuda::stream_ref stream, const std::filesystem::path& checkpoint, const std::filesystem::path& cache_directory);

    private:
        friend struct Inference;
        neural::InferenceRuntime runtime;
        Weights weights;
        Checkpoint checkpoint;
        Tokenizer tokenizer;
        Clip clip_l;
        Clip clip_g;
        UNet unet;
        VAE vae;
        ::cuda::device_buffer<float> training;
    };

    struct Output final {
        ::cuda::host_buffer<std::uint8_t> pixels;
        ::cuda::host_buffer<float> latent;
        std::uint64_t seed{};
        int width;
        int height;
        double initialize_seconds{};
        double sample_seconds{};
        double decode_seconds{};
        double transfer_seconds{};

        Output(::cuda::stream_ref stream, int width, int height);
    };

    struct Inference final {
        const Parameters parameters;
        double prepare_seconds{};
        std::size_t resident_bytes{};
        std::size_t cache_hits{};
        std::size_t cache_misses{};
        double text_seconds{};
        double precompute_seconds{};
        double tuning_seconds{};

        Inference(Model& model, Parameters parameters);
        ~Inference();
        Inference(const Inference&)            = delete;
        Inference& operator=(const Inference&) = delete;
        // Two pinned outputs alternate. Finish reading a result before the
        // second subsequent generate() call reuses its storage.
        const Output& generate(std::uint64_t seed);

    private:
        Model& model;
        UNetWorkspaceLayout unet_layout;
        VAEWorkspaceLayout vae_layout;
        ::cuda::device_buffer<std::byte> workspace;
        UNetState unet;
        ::cuda::device_buffer<kernels::SamplingStep> schedule;
        ::cuda::device_buffer<std::uint64_t> seed;
        ::cuda::device_buffer<std::byte> operator_workspace;
        ::cuda::device_buffer<float> latent;
        ::cuda::device_buffer<int> step;
        ::cuda::device_buffer<__half> input;
        ::cuda::device_buffer<__half> epsilon;
        ::cuda::device_buffer<__nv_bfloat16> decoder;
        ::cuda::device_buffer<__nv_bfloat16> decoded;
        ::cuda::device_buffer<std::uint8_t> image;
        ::cuda::device_buffer<float> scaled_latent;
        std::array<Output, 2> outputs;
        std::size_t output_index{};
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        cudaGraphConditionalHandle loop{};
        cudaEvent_t begin{};
        cudaEvent_t initialized{};
        cudaEvent_t sampled{};
        cudaEvent_t decoded_event{};
        cudaEvent_t copied{};

        void denoise();
        void decode();
    };
} // namespace physica::generative::sdxl
