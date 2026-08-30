module;

#include "training-state-kernels.h"
#include <physica/cuda.h>

module physica.neural.training_state;

import std;

namespace physica::neural {
    ParameterBuffer::ParameterBuffer(const ::cuda::stream_ref source_stream, const std::size_t count)
        : stream{source_stream}, parameters{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, gradients{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, first_moments{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, second_moments{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, ema{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, step_scalars{stream, ::cuda::device_default_memory_pool(stream.device()), 3uz, ::cuda::no_init} {}

    void ParameterBuffer::initialize(const std::span<const float> values) {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{values.data(), values.size()}, parameters);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{values.data(), values.size()}, ema);
        ::cuda::fill_bytes(stream, gradients, 0u);
        ::cuda::fill_bytes(stream, first_moments, 0u);
        ::cuda::fill_bytes(stream, second_moments, 0u);
        stream.sync();
    }

    void ParameterBuffer::clear_gradients() {
        ::cuda::fill_bytes(stream, gradients, 0u);
    }

    void ParameterBuffer::step(const TrainingConfiguration& configuration, const std::uint64_t step, const std::uint64_t processed_samples, const std::uint32_t samples_per_step) {
        const float first_correction  = 1.0F - std::pow(configuration.first_decay, static_cast<float>(step));
        const float second_correction = 1.0F - std::pow(configuration.second_decay, static_cast<float>(step));
        const float half_life = std::min(static_cast<float>(configuration.exponential_average.half_life_samples), static_cast<float>(processed_samples) * configuration.exponential_average.ramp_up_ratio);
        const float exponential_average_decay = processed_samples == 0u ? 0.0F : std::exp2(-static_cast<float>(samples_per_step) / half_life);
        kernels::optimize(stream, parameters.data(), gradients.data(), first_moments.data(), second_moments.data(), ema.data(), parameters.size(), configuration.learning_rate, configuration.first_decay, configuration.second_decay, first_correction, second_correction, configuration.epsilon, configuration.weight_decay, exponential_average_decay);
    }

    void ParameterBuffer::step(const TrainingConfiguration& configuration, const std::uint64_t* const step, const std::uint64_t* const processed_samples, const std::uint32_t samples_per_step) {
        kernels::optimize(stream, parameters.data(), gradients.data(), first_moments.data(), second_moments.data(), ema.data(), parameters.size(), configuration.learning_rate, configuration.first_decay, configuration.second_decay, step, processed_samples, step_scalars.data(), configuration.epsilon, configuration.weight_decay, samples_per_step, configuration.exponential_average.half_life_samples, configuration.exponential_average.ramp_up_ratio);
    }

    ParameterState ParameterBuffer::download() const {
        ParameterState result{
            .parameters     = std::vector<float>(parameters.size()),
            .first_moments  = std::vector<float>(first_moments.size()),
            .second_moments = std::vector<float>(second_moments.size()),
            .ema            = std::vector<float>(ema.size()),
        };
        ::cuda::copy_bytes(stream, parameters, ::cuda::std::span<float>{result.parameters.data(), result.parameters.size()});
        ::cuda::copy_bytes(stream, first_moments, ::cuda::std::span<float>{result.first_moments.data(), result.first_moments.size()});
        ::cuda::copy_bytes(stream, second_moments, ::cuda::std::span<float>{result.second_moments.data(), result.second_moments.size()});
        ::cuda::copy_bytes(stream, ema, ::cuda::std::span<float>{result.ema.data(), result.ema.size()});
        stream.sync();
        return result;
    }

    void ParameterBuffer::upload(const ParameterState& state) {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{state.parameters.data(), state.parameters.size()}, parameters);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{state.first_moments.data(), state.first_moments.size()}, first_moments);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{state.second_moments.data(), state.second_moments.size()}, second_moments);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{state.ema.data(), state.ema.size()}, ema);
        ::cuda::fill_bytes(stream, gradients, 0u);
        stream.sync();
    }

    InferenceParameterBuffer::InferenceParameterBuffer(const ::cuda::stream_ref source_stream, const std::span<const float> values)
        : stream{source_stream}, parameters{stream, ::cuda::device_default_memory_pool(stream.device()), values.size(), ::cuda::no_init} {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{values.data(), values.size()}, parameters);
    }
} // namespace physica::neural
