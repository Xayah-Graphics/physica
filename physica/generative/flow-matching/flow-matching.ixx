module;

#include <cuda_runtime_api.h>
#include <physica/cuda.h>

export module physica.generative.flow_matching;

import std;
import physica.generative.dataset.cifar10;
import physica.generative.flow_matching.model;
export import physica.generative.flow_matching.sampling;
import physica.neural.matmul;
import physica.neural.training_state;

export namespace physica::generative::flow_matching {
    struct TrainingState final {
        std::uint64_t step{};
        std::uint64_t processed_samples{};
        std::uint64_t seed{};
        double elapsed_seconds{};
    };

    enum class ParameterSource : std::uint8_t {
        parameters,
        exponential_average,
    };

    struct TrainingStatistics final {
        std::uint64_t step;
        float average_loss;
        double samples_per_second;
        double elapsed_seconds;
    };

    struct Trainer final {
        TrainingState state;

        Trainer(const Cifar10TrainingSet& training_set, int device_ordinal, std::uint64_t seed);
        ~Trainer() noexcept;

        Trainer(const Trainer&) = delete;
        Trainer& operator=(const Trainer&) = delete;
        Trainer(Trainer&&) = delete;
        Trainer& operator=(Trainer&&) = delete;

        TrainingStatistics optimize(std::uint64_t iterations);
        SamplingResult sample(const SamplingRequest& request, ParameterSource source = ParameterSource::exponential_average);
        void save(const std::filesystem::path& path) const;
        void load(const std::filesystem::path& path);

    private:
        inline static constexpr std::uint32_t batch = 256u;
        inline static constexpr std::size_t value_count = static_cast<std::size_t>(batch) * 256uz * 12uz;

        ::cuda::stream stream;
        ::cuda::device_buffer<std::uint8_t> dataset_images;
        ::cuda::device_buffer<std::uint8_t> dataset_labels;
        neural::MatmulRuntime matmul;
        FlowDiT model;
        neural::ParameterBuffer parameter_buffer;
        neural::TrainingConfiguration training_configuration;
        FlowDiTWorkspaceLayout model_workspace_layout;
        ::cuda::device_buffer<std::uint8_t> model_workspace;
        ::cuda::device_buffer<float> path;
        ::cuda::device_buffer<float> target;
        ::cuda::device_buffer<float> times;
        ::cuda::device_buffer<std::uint8_t> labels;
        ::cuda::device_buffer<float> patch_gradient;
        ::cuda::device_buffer<float> loss;
        ::cuda::device_buffer<float> loss_sum;
        ::cuda::device_buffer<std::uint64_t> device_step;
        ::cuda::device_buffer<std::uint64_t> device_processed_samples;
        ::cuda::device_buffer<std::uint64_t> device_seed;
        cudaGraph_t graph{};
        cudaGraphExec_t graph_execution{};

        void training_step();
    };

    struct Sampler final {
        Sampler(const std::filesystem::path& checkpoint, int device_ordinal);

        SamplingResult sample(const SamplingRequest& request);

    private:
        ::cuda::stream stream;
        neural::MatmulRuntime matmul;
        FlowDiT model;
        std::vector<float> checkpoint_ema;
        neural::InferenceParameterBuffer parameters;
        SamplingRuntime runtime;
    };
} // namespace physica::generative::flow_matching
