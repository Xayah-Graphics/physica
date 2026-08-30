module;

#include "../network/kernels.h"
#include "kernels.h"
#include <physica/cuda.h>

module physica.reconstruction.pinfs.field;

import std;
import physica.reconstruction.pinfs.network;

namespace physica::reconstruction::pinfs {
    namespace {
        NetworkConfiguration dynamic_configuration(const float first_frequency, const bool unique_first_frequency) {
            NetworkConfiguration result{.input_width = 4u};
            result.layers.reserve(9uz);
            for (const std::uint32_t layer : std::views::iota(0u, 8u)) {
                const std::uint32_t input_width = layer == 0u ? 4u : layer == 5u ? 260u : 256u;
                const float frequency           = layer == 0u ? first_frequency : 1.0F;
                const float weight_width        = layer == 0u && unique_first_frequency ? 1.0F / static_cast<float>(input_width) : std::sqrt(6.0F / static_cast<float>(input_width)) / frequency;
                result.layers.push_back({
                    .input_width           = input_width,
                    .output_width          = 256u,
                    .activation            = Activation::sine,
                    .frequency             = frequency,
                    .append_network_input  = layer == 5u,
                    .weight_initialization = Initialization::uniform,
                    .weight_scale          = weight_width,
                    .bias_initialization   = Initialization::uniform,
                    .bias_scale            = 1.0F / std::sqrt(static_cast<float>(input_width)),
                });
            }
            result.layers.push_back({
                .input_width                = 256u,
                .output_width               = 4u,
                .activation                 = Activation::rgb_density,
                .weight_initialization      = Initialization::uniform,
                .weight_scale               = 1.0F / 16.0F,
                .bias_initialization        = Initialization::uniform,
                .bias_scale                 = 1.0F / 16.0F,
                .blended_hidden_layer_count = 8u,
            });
            return result;
        }

        NetworkConfiguration velocity_configuration() {
            NetworkConfiguration result{.input_width = 4u};
            result.layers.reserve(7uz);
            for (const std::uint32_t layer : std::views::iota(0u, 6u)) {
                const std::uint32_t input_width = layer == 0u ? 4u : 128u;
                const float frequency           = layer == 0u ? 30.0F : 1.0F;
                result.layers.push_back({
                    .input_width           = input_width,
                    .output_width          = 128u,
                    .activation            = Activation::sine,
                    .frequency             = frequency,
                    .weight_initialization = Initialization::uniform,
                    .weight_scale          = std::sqrt(6.0F / static_cast<float>(input_width)) / frequency,
                    .bias_initialization   = Initialization::uniform,
                    .bias_scale            = 1.0F / std::sqrt(static_cast<float>(input_width)),
                });
            }
            result.layers.push_back({
                .input_width                = 128u,
                .output_width               = 3u,
                .activation                 = Activation::linear,
                .weight_initialization      = Initialization::uniform,
                .weight_scale               = 1.0F / std::sqrt(128.0F),
                .bias_initialization        = Initialization::uniform,
                .bias_scale                 = 1.0F / std::sqrt(128.0F),
                .blended_hidden_layer_count = 6u,
            });
            return result;
        }

        NetworkConfiguration sdf_configuration(const std::uint32_t encoded_width) {
            NetworkConfiguration result{.input_width = encoded_width};
            result.layers.reserve(8uz);
            std::uint32_t previous_width = encoded_width;
            for (const std::uint32_t layer : std::views::iota(0u, 8u)) {
                const std::uint32_t input_width  = layer == 4u ? previous_width + encoded_width : previous_width;
                const std::uint32_t output_width = layer == 3u ? 256u - encoded_width : layer == 7u ? 257u : 256u;
                const bool final_layer           = layer == 7u;
                result.layers.push_back({
                    .input_width               = input_width,
                    .output_width              = output_width,
                    .activation                = final_layer ? Activation::linear : Activation::softplus,
                    .frequency                 = final_layer ? 1.0F : 100.0F,
                    .append_network_input      = layer == 4u,
                    .existing_input_scale      = layer == 4u ? 1.0F / std::numbers::sqrt2_v<float> : 1.0F,
                    .appended_input_scale      = layer == 4u ? 1.0F / std::numbers::sqrt2_v<float> : 1.0F,
                    .weight_normalization      = true,
                    .weight_initialization     = Initialization::normal,
                    .weight_center             = final_layer ? std::sqrt(std::numbers::pi_v<float> / static_cast<float>(input_width)) : 0.0F,
                    .weight_scale              = final_layer ? 0.0001F : std::sqrt(2.0F / static_cast<float>(output_width)),
                    .bias_initialization       = Initialization::constant,
                    .bias_center               = final_layer ? -0.5F : 0.0F,
                    .zero_weight_columns_begin = encoded_width > 3u && layer == 0u ? 3u
                                               : encoded_width > 3u && layer == 4u ? previous_width + 3u
                                                                                   : (std::numeric_limits<std::uint32_t>::max)(),
                });
                previous_width = output_width;
            }
            return result;
        }

        NetworkConfiguration color_configuration() {
            NetworkConfiguration result{.input_width = 265u};
            result.layers.reserve(5uz);
            for (const std::uint32_t layer : std::views::iota(0u, 5u)) {
                const std::uint32_t input_width = layer == 0u ? 265u : 256u;
                result.layers.push_back({
                    .input_width           = input_width,
                    .output_width          = layer == 4u ? 3u : 256u,
                    .activation            = layer == 4u ? Activation::sigmoid : Activation::relu,
                    .weight_normalization  = true,
                    .weight_initialization = Initialization::uniform,
                    .weight_scale          = 1.0F / std::sqrt(static_cast<float>(input_width)),
                    .bias_initialization   = Initialization::uniform,
                    .bias_scale            = 1.0F / std::sqrt(static_cast<float>(input_width)),
                });
            }
            return result;
        }
    } // namespace

    DynamicField::DynamicField(const ::cuda::stream_ref source_stream, const std::uint32_t capacity, const std::uint32_t derivative_capacity, const std::uint32_t maximum_derivative_count, const std::uint32_t seed, const float first_frequency, const bool unique_first_frequency) : stream{source_stream}, network{stream, dynamic_configuration(first_frequency, unique_first_frequency), capacity, derivative_capacity, maximum_derivative_count, seed} {}

    DeviceTensor DynamicField::forward(const ConstDeviceTensor points) {
        return network.forward(points);
    }

    void DynamicField::copy_density(const ConstDeviceTensor field, float* destination) {
        kernels::extract_density(stream, field.values, destination, field.sample_count);
    }

    DeviceTensor DynamicField::backward(const ConstDeviceTensor adjoints) {
        return network.backward(adjoints);
    }

    void DynamicField::clear_gradients() {
        network.clear_gradients();
    }

    void DynamicField::step(const float learning_rate) {
        network.step(learning_rate);
    }

    void DynamicField::set_fading_step(const std::uint32_t step, const std::uint32_t final_step) {
        network.set_fading_step(step, final_step);
    }

    NetworkState DynamicField::download() const {
        return network.download();
    }

    void DynamicField::upload(const NetworkState& state) {
        network.upload(state);
    }

    VelocityField::VelocityField(const ::cuda::stream_ref stream, const std::uint32_t capacity, const std::uint32_t derivative_capacity, const std::uint32_t seed) : network{stream, velocity_configuration(), capacity, derivative_capacity, 4u, seed} {}

    DeviceTensor VelocityField::forward(const ConstDeviceTensor points) {
        return network.forward(points);
    }

    DeviceTensor VelocityField::backward(const ConstDeviceTensor adjoints) {
        return network.backward(adjoints);
    }

    void VelocityField::clear_gradients() {
        network.clear_gradients();
    }

    void VelocityField::step(const float learning_rate) {
        network.step(learning_rate);
    }

    void VelocityField::set_fading_step(const std::uint32_t step, const std::uint32_t final_step) {
        network.set_fading_step(step, final_step);
    }

    NetworkState VelocityField::download() const {
        return network.download();
    }

    void VelocityField::upload(const NetworkState& state) {
        network.upload(state);
    }

    StaticField::StaticField(const ::cuda::stream_ref source_stream, const std::uint32_t source_capacity, const std::uint32_t derivative_capacity, const std::uint32_t seed, const std::uint32_t source_frequency_count)
        : stream{source_stream}, frequency_count{source_frequency_count}, encoded_width{3u * (1u + 2u * frequency_count)}, sdf_network{stream, sdf_configuration(encoded_width), source_capacity, derivative_capacity, 3u, seed}, color_network{stream, color_configuration(), source_capacity, 0u, 0u, seed + 1u}, encoded_positions{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(encoded_width) * source_capacity, ::cuda::no_init}, encoded_position_derivatives{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(encoded_width) * derivative_capacity * 3uz, ::cuda::no_init}, encoding_weights{stream, ::cuda::device_default_memory_pool(stream.device()), frequency_count + 1uz, ::cuda::no_init}, color_inputs{stream, ::cuda::device_default_memory_pool(stream.device()), 265uz * source_capacity, ::cuda::no_init}, sdf_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), 257uz * derivative_capacity, ::cuda::no_init},
          sdf_derivative_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), 257uz * derivative_capacity * 3uz, ::cuda::no_init}, deviation{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, inverse_deviation{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, deviation_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, deviation_first_moment{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, deviation_second_moment{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init} {
        const float initial_deviation = 0.3F;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{&initial_deviation, 1uz}, deviation);
        ::cuda::fill_bytes(stream, deviation_first_moment, 0u);
        ::cuda::fill_bytes(stream, deviation_second_moment, 0u);
        set_fading_step(1u, 1u);
        clear_gradients();
    }

    DeviceTensor StaticField::infer_sdf(const Vector3<float>* positions, const std::uint32_t sample_count) {
        kernels::positional_encoding(stream, positions, encoding_weights.data(), encoded_positions.data(), nullptr, sample_count, frequency_count);
        return sdf_network.forward({.values = encoded_positions.data(), .width = encoded_width, .sample_count = sample_count});
    }

    DeviceTensor StaticField::forward_sdf(const Vector3<float>* positions, const std::uint32_t sample_count) {
        kernels::positional_encoding(stream, positions, encoding_weights.data(), encoded_positions.data(), encoded_position_derivatives.data(), sample_count, frequency_count);
        return sdf_network.forward({.values = encoded_positions.data(), .derivatives = encoded_position_derivatives.data(), .width = encoded_width, .sample_count = sample_count, .derivative_count = 3u});
    }

    const float* StaticField::evaluate_inverse_deviation() {
        kernels::inverse_deviation_forward(stream, deviation.data(), inverse_deviation.data());
        return inverse_deviation.data();
    }

    StaticFieldOutput StaticField::forward(const Vector3<float>* positions, const Vector3<float>* directions, const std::uint32_t sample_count) {
        const DeviceTensor sdf = forward_sdf(positions, sample_count);
        kernels::static_color_input(stream, positions, directions, sdf.values, sdf.derivatives, color_inputs.data(), sample_count);
        const DeviceTensor color = color_network.forward({.values = color_inputs.data(), .width = 265u, .sample_count = sample_count});
        evaluate_inverse_deviation();
        return {.sdf = sdf, .color = color, .inverse_deviation = inverse_deviation.data()};
    }

    void StaticField::backward(const ConstDeviceTensor color_adjoints, const float* source_sdf_adjoints, const float* source_gradient_adjoints, const float* inverse_deviation_adjoint) {
        const DeviceTensor color_input_adjoints = color_network.backward(color_adjoints);
        kernels::static_sdf_adjoint(stream, color_input_adjoints.values, source_sdf_adjoints, source_gradient_adjoints, sdf_adjoints.data(), sdf_derivative_adjoints.data(), color_adjoints.sample_count);
        sdf_network.backward({.values = sdf_adjoints.data(), .derivatives = sdf_derivative_adjoints.data(), .width = 257u, .sample_count = color_adjoints.sample_count, .derivative_count = 3u});
        kernels::inverse_deviation_backward(stream, inverse_deviation.data(), inverse_deviation_adjoint, deviation_gradient.data());
    }

    void StaticField::clear_gradients() {
        sdf_network.clear_gradients();
        color_network.clear_gradients();
        ::cuda::fill_bytes(stream, deviation_gradient, 0u);
    }

    void StaticField::step(const float learning_rate) {
        sdf_network.step(learning_rate);
        color_network.step(learning_rate);
        kernels::adam(stream, deviation.data(), deviation_gradient.data(), deviation_first_moment.data(), deviation_second_moment.data(), 1uz, learning_rate, ++deviation_step);
    }

    void StaticField::set_fading_step(const std::uint32_t step, const std::uint32_t final_step) {
        std::vector<float> weights(frequency_count + 1uz, 1.0F);
        const float alpha = static_cast<float>(step) * static_cast<float>(frequency_count) / static_cast<float>(final_step);
        for (const std::uint32_t frequency : std::views::iota(0u, frequency_count)) weights[frequency + 1u] = 0.5F * (1.0F - std::cos(std::numbers::pi_v<float> * std::clamp(alpha - static_cast<float>(frequency), 0.0F, 1.0F)));
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{weights.data(), weights.size()}, encoding_weights);
    }

    StaticFieldState StaticField::download() const {
        StaticFieldState state{.sdf = sdf_network.download(), .color = color_network.download(), .deviation_step = deviation_step};
        ::cuda::copy_bytes(stream, deviation, ::cuda::std::span<float>{&state.deviation, 1uz});
        ::cuda::copy_bytes(stream, deviation_first_moment, ::cuda::std::span<float>{&state.deviation_first_moment, 1uz});
        ::cuda::copy_bytes(stream, deviation_second_moment, ::cuda::std::span<float>{&state.deviation_second_moment, 1uz});
        stream.sync();
        return state;
    }

    void StaticField::upload(const StaticFieldState& state) {
        sdf_network.upload(state.sdf);
        color_network.upload(state.color);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{&state.deviation, 1uz}, deviation);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{&state.deviation_first_moment, 1uz}, deviation_first_moment);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{&state.deviation_second_moment, 1uz}, deviation_second_moment);
        deviation_step = state.deviation_step;
    }
} // namespace physica::reconstruction::pinfs
