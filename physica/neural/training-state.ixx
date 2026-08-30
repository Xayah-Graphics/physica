module;

#include <physica/cuda.h>

export module physica.neural.training_state;

import std;

export namespace physica::neural {
    struct ExponentialAverageConfiguration final {
        std::uint64_t half_life_samples{500'000u};
        float ramp_up_ratio{0.05F};
    };

    struct TrainingConfiguration final {
        float learning_rate{1.0e-4F};
        float first_decay{0.9F};
        float second_decay{0.999F};
        float epsilon{1.0e-8F};
        float weight_decay{};
        ExponentialAverageConfiguration exponential_average;
    };

    struct ParameterState final {
        std::vector<float> parameters;
        std::vector<float> first_moments;
        std::vector<float> second_moments;
        std::vector<float> ema;
    };

    struct ParameterBuffer final {
        ::cuda::stream_ref stream;
        ::cuda::device_buffer<float> parameters;
        ::cuda::device_buffer<float> gradients;
        ::cuda::device_buffer<float> first_moments;
        ::cuda::device_buffer<float> second_moments;
        ::cuda::device_buffer<float> ema;
        ::cuda::device_buffer<float> step_scalars;

        ParameterBuffer(::cuda::stream_ref stream, std::size_t count);

        void initialize(std::span<const float> values);
        void clear_gradients();
        void step(const TrainingConfiguration& configuration, std::uint64_t step, std::uint64_t processed_samples, std::uint32_t samples_per_step);
        void step(const TrainingConfiguration& configuration, const std::uint64_t* step, const std::uint64_t* processed_samples, std::uint32_t samples_per_step);
        ParameterState download() const;
        void upload(const ParameterState& state);
    };

    struct InferenceParameterBuffer final {
        ::cuda::stream_ref stream;
        ::cuda::device_buffer<float> parameters;

        InferenceParameterBuffer(::cuda::stream_ref stream, std::span<const float> values);
    };
} // namespace physica::neural
