module;

#include "kernels.h"
#include <cublas_v2.h>
#include <physica/cuda.h>

module physica.reconstruction.pinfs.network;

import std;

namespace physica::reconstruction::pinfs {
    namespace {
        void matrix_product(const cublasHandle_t handle, const cublasOperation_t first_operation, const cublasOperation_t second_operation, const std::uint32_t rows, const std::uint32_t columns, const std::uint32_t reduction, const float alpha, const float* first, const std::uint32_t first_leading_dimension, const float* second, const std::uint32_t second_leading_dimension, const float beta, float* output, const std::uint32_t output_leading_dimension) {
            const cublasStatus_t status = cublasSgemm(handle, first_operation, second_operation, static_cast<int>(rows), static_cast<int>(columns), static_cast<int>(reduction), &alpha, first, static_cast<int>(first_leading_dimension), second, static_cast<int>(second_leading_dimension), &beta, output, static_cast<int>(output_leading_dimension));
            if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cublasSgemm failed: {}", cublasGetStatusString(status))};
        }
    } // namespace

    DenseNetwork::DenseNetwork(const ::cuda::stream_ref source_stream, NetworkConfiguration source_configuration, const std::uint32_t source_capacity, const std::uint32_t source_derivative_capacity, const std::uint32_t source_maximum_derivative_count, const std::uint32_t seed)
        : stream{source_stream}, configuration{std::move(source_configuration)}, maximum_width{std::ranges::max(configuration.layers | std::views::transform([](const LayerConfiguration& layer) { return std::max(layer.input_width, layer.output_width); }))}, adjoints_a{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_width) * source_capacity, ::cuda::no_init}, adjoints_b{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_width) * source_capacity, ::cuda::no_init}, derivative_adjoints_a{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_width) * source_derivative_capacity * source_maximum_derivative_count, ::cuda::no_init},
          derivative_adjoints_b{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_width) * source_derivative_capacity * source_maximum_derivative_count, ::cuda::no_init}, network_input_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.input_width) * source_capacity, ::cuda::no_init}, network_input_derivative_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.input_width) * source_derivative_capacity * source_maximum_derivative_count, ::cuda::no_init}, blended_inputs{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_width) * source_capacity, ::cuda::no_init}, blended_input_derivatives{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_width) * source_derivative_capacity * source_maximum_derivative_count, ::cuda::no_init},
          blended_input_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_width) * source_capacity, ::cuda::no_init}, blended_input_derivative_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_width) * source_derivative_capacity * source_maximum_derivative_count, ::cuda::no_init} {
        cublasContext* handle{};
        if (const cublasStatus_t status = cublasCreate(&handle); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cublasCreate failed: {}", cublasGetStatusString(status))};
        cublas.reset(handle);
        if (const cublasStatus_t status = cublasSetStream(cublas.get(), stream.get()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cublasSetStream failed: {}", cublasGetStatusString(status))};

        std::mt19937 generator{seed};
        layers.reserve(configuration.layers.size());
        for (const LayerConfiguration layer_configuration : configuration.layers) {
            const std::size_t weight_count = static_cast<std::size_t>(layer_configuration.input_width) * layer_configuration.output_width;
            std::vector<float> host_weights(weight_count);
            std::vector<float> host_biases(layer_configuration.output_width);
            const auto initialize = [&](std::vector<float>& values, const Initialization initialization, const float center, const float scale) {
                if (initialization == Initialization::constant) {
                    std::ranges::fill(values, center);
                    return;
                }
                if (initialization == Initialization::uniform) {
                    std::uniform_real_distribution<float> distribution{center - scale, center + scale};
                    std::ranges::generate(values, [&] { return distribution(generator); });
                    return;
                }
                std::normal_distribution<float> distribution{center, scale};
                std::ranges::generate(values, [&] { return distribution(generator); });
            };
            initialize(host_weights, layer_configuration.weight_initialization, layer_configuration.weight_center, layer_configuration.weight_scale);
            initialize(host_biases, layer_configuration.bias_initialization, layer_configuration.bias_center, layer_configuration.bias_scale);
            if (layer_configuration.zero_weight_columns_begin < layer_configuration.input_width)
                for (const std::uint32_t column : std::views::iota(layer_configuration.zero_weight_columns_begin, layer_configuration.input_width))
                    for (const std::uint32_t row : std::views::iota(0u, layer_configuration.output_width)) host_weights[row + static_cast<std::size_t>(column) * layer_configuration.output_width] = 0.0F;
            std::vector<float> host_scales(layer_configuration.weight_normalization ? layer_configuration.output_width : 0u);
            if (layer_configuration.weight_normalization) {
                for (const std::uint32_t row : std::views::iota(0u, layer_configuration.output_width)) {
                    double squared_norm{};
                    for (const std::uint32_t column : std::views::iota(0u, layer_configuration.input_width)) squared_norm += static_cast<double>(host_weights[row + column * layer_configuration.output_width]) * host_weights[row + column * layer_configuration.output_width];
                    host_scales[row] = static_cast<float>(std::sqrt(squared_norm));
                }
            }

            Layer layer{stream, layer_configuration, source_capacity, source_derivative_capacity, source_maximum_derivative_count};
            ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{host_weights.data(), host_weights.size()}, layer.weights);
            ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{host_biases.data(), host_biases.size()}, layer.biases);
            if (!host_scales.empty()) ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{host_scales.data(), host_scales.size()}, layer.scales);
            layers.push_back(std::move(layer));
        }
        clear_gradients();
        for (Layer& layer : layers) {
            ::cuda::fill_bytes(stream, layer.weight_first_moments, 0u);
            ::cuda::fill_bytes(stream, layer.bias_first_moments, 0u);
            ::cuda::fill_bytes(stream, layer.scale_first_moments, 0u);
            ::cuda::fill_bytes(stream, layer.weight_second_moments, 0u);
            ::cuda::fill_bytes(stream, layer.bias_second_moments, 0u);
            ::cuda::fill_bytes(stream, layer.scale_second_moments, 0u);
        }
        for (const LayerConfiguration& layer : configuration.layers) {
            if (layer.blended_hidden_layer_count <= fading_weights.size()) continue;
            fading_weights.assign(layer.blended_hidden_layer_count, 0.0F);
            fading_weights.back() = 1.0F;
        }
    }

    DenseNetwork::~DenseNetwork() noexcept = default;

    DeviceTensor DenseNetwork::forward(const ConstDeviceTensor input) {
        current_input           = input;
        ConstDeviceTensor layer_input = input;
        for (const std::size_t layer_index : std::views::iota(0uz, layers.size())) {
            Layer& layer = layers[layer_index];
            if (layer.configuration.blended_hidden_layer_count != 0u) {
                const std::size_t value_count = static_cast<std::size_t>(layer.configuration.input_width) * input.sample_count;
                ::cuda::fill_bytes(stream, ::cuda::std::span<float>{blended_inputs.data(), value_count}, 0u);
                if (input.derivative_count != 0u) ::cuda::fill_bytes(stream, ::cuda::std::span<float>{blended_input_derivatives.data(), value_count * input.derivative_count}, 0u);
                const std::size_t first_hidden_layer = layer_index - layer.configuration.blended_hidden_layer_count;
                for (const std::size_t hidden_offset : std::views::iota(0uz, static_cast<std::size_t>(layer.configuration.blended_hidden_layer_count))) {
                    const Layer& hidden = layers[first_hidden_layer + hidden_offset];
                    kernels::scaled_add(stream, blended_inputs.data(), hidden.outputs.data(), fading_weights[hidden_offset], value_count);
                    if (input.derivative_count != 0u) kernels::scaled_add(stream, blended_input_derivatives.data(), hidden.output_derivatives.data(), fading_weights[hidden_offset], value_count * input.derivative_count);
                }
                layer_input = {.values = blended_inputs.data(), .derivatives = input.derivative_count == 0u ? nullptr : blended_input_derivatives.data(), .width = layer.configuration.input_width, .sample_count = input.sample_count, .derivative_count = input.derivative_count};
            } else if (layer.configuration.append_network_input) {
                const std::uint32_t previous_width = layer.configuration.input_width - configuration.input_width;
                kernels::concatenate_forward(stream, layer_input.values, layer_input.derivatives, previous_width, layer.configuration.existing_input_scale, input.values, input.derivatives, configuration.input_width, layer.configuration.appended_input_scale, layer.inputs.data(), input.derivative_count == 0u ? nullptr : layer.input_derivatives.data(), input.sample_count, input.derivative_count);
                layer_input = {.values = layer.inputs.data(), .derivatives = input.derivative_count == 0u ? nullptr : layer.input_derivatives.data(), .width = layer.configuration.input_width, .sample_count = input.sample_count, .derivative_count = input.derivative_count};
            }
            const float* weights = layer.weights.data();
            if (layer.configuration.weight_normalization) {
                kernels::weight_normalization_forward(stream, layer.weights.data(), layer.scales.data(), layer.normalized_weights.data(), layer.configuration.input_width, layer.configuration.output_width);
                weights = layer.normalized_weights.data();
            }
            matrix_product(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, layer.configuration.output_width, input.sample_count, layer.configuration.input_width, 1.0F, weights, layer.configuration.output_width, layer_input.values, layer.configuration.input_width, 0.0F, layer.linear.data(), layer.configuration.output_width);
            for (const std::uint32_t derivative : std::views::iota(0u, input.derivative_count)) matrix_product(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, layer.configuration.output_width, input.sample_count, layer.configuration.input_width, 1.0F, weights, layer.configuration.output_width, layer_input.derivatives + static_cast<std::size_t>(derivative) * layer.configuration.input_width * input.sample_count, layer.configuration.input_width, 0.0F, layer.linear_derivatives.data() + static_cast<std::size_t>(derivative) * layer.configuration.output_width * input.sample_count, layer.configuration.output_width);
            kernels::activation_forward(stream, static_cast<kernels::Activation>(layer.configuration.activation), layer.configuration.frequency, layer.biases.data(), layer.linear.data(), input.derivative_count == 0u ? nullptr : layer.linear_derivatives.data(), layer.outputs.data(), input.derivative_count == 0u ? nullptr : layer.output_derivatives.data(), layer.configuration.output_width, input.sample_count, input.derivative_count);
            layer_input = {.values = layer.outputs.data(), .derivatives = input.derivative_count == 0u ? nullptr : layer.output_derivatives.data(), .width = layer.configuration.output_width, .sample_count = input.sample_count, .derivative_count = input.derivative_count};
        }
        return {.values = layers.back().outputs.data(), .derivatives = input.derivative_count == 0u ? nullptr : layers.back().output_derivatives.data(), .width = layers.back().configuration.output_width, .sample_count = input.sample_count, .derivative_count = input.derivative_count};
    }

    DeviceTensor DenseNetwork::backward(const ConstDeviceTensor source_adjoints) {
        ::cuda::fill_bytes(stream, network_input_adjoints, 0u);
        if (source_adjoints.derivative_count != 0u) ::cuda::fill_bytes(stream, network_input_derivative_adjoints, 0u);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{source_adjoints.values, static_cast<std::size_t>(source_adjoints.width) * source_adjoints.sample_count}, ::cuda::std::span<float>{adjoints_a.data(), static_cast<std::size_t>(source_adjoints.width) * source_adjoints.sample_count});
        if (source_adjoints.derivative_count != 0u) ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{source_adjoints.derivatives, static_cast<std::size_t>(source_adjoints.width) * source_adjoints.sample_count * source_adjoints.derivative_count}, ::cuda::std::span<float>{derivative_adjoints_a.data(), static_cast<std::size_t>(source_adjoints.width) * source_adjoints.sample_count * source_adjoints.derivative_count});

        float* current_adjoints            = adjoints_a.data();
        float* next_adjoints               = adjoints_b.data();
        float* current_derivative_adjoints = source_adjoints.derivative_count == 0u ? nullptr : derivative_adjoints_a.data();
        float* next_derivative_adjoints    = source_adjoints.derivative_count == 0u ? nullptr : derivative_adjoints_b.data();
        std::size_t blended_head           = layers.size();
        std::size_t first_blended_hidden{};
        for (const std::size_t layer_index : std::views::iota(0uz, layers.size())) {
            if (layers[layer_index].configuration.blended_hidden_layer_count == 0u) continue;
            blended_head         = layer_index;
            first_blended_hidden = layer_index - layers[layer_index].configuration.blended_hidden_layer_count;
        }

        for (std::size_t layer_index = layers.size(); layer_index-- > 0uz;) {
            Layer& layer                  = layers[layer_index];
            const ConstDeviceTensor layer_input = layer.configuration.blended_hidden_layer_count != 0u ? ConstDeviceTensor{blended_inputs.data(), source_adjoints.derivative_count == 0u ? nullptr : blended_input_derivatives.data(), layer.configuration.input_width, source_adjoints.sample_count, source_adjoints.derivative_count} : layer.configuration.append_network_input ? ConstDeviceTensor{layer.inputs.data(), source_adjoints.derivative_count == 0u ? nullptr : layer.input_derivatives.data(), layer.configuration.input_width, source_adjoints.sample_count, source_adjoints.derivative_count} : layer_index == 0uz ? current_input : ConstDeviceTensor{layers[layer_index - 1uz].outputs.data(), source_adjoints.derivative_count == 0u ? nullptr : layers[layer_index - 1uz].output_derivatives.data(), layers[layer_index - 1uz].configuration.output_width, source_adjoints.sample_count, source_adjoints.derivative_count};
            kernels::activation_backward(stream, static_cast<kernels::Activation>(layer.configuration.activation), layer.configuration.frequency, layer.linear.data(), source_adjoints.derivative_count == 0u ? nullptr : layer.linear_derivatives.data(), current_adjoints, current_derivative_adjoints, current_adjoints, current_derivative_adjoints, layer.bias_gradients.data(), layer.configuration.output_width, source_adjoints.sample_count, source_adjoints.derivative_count);
            const float* weights = layer.configuration.weight_normalization ? layer.normalized_weights.data() : layer.weights.data();
            matrix_product(cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N, layer.configuration.input_width, source_adjoints.sample_count, layer.configuration.output_width, 1.0F, weights, layer.configuration.output_width, current_adjoints, layer.configuration.output_width, 0.0F, next_adjoints, layer.configuration.input_width);
            float* weight_gradients = layer.weight_gradients.data();
            float gradient_beta     = 1.0F;
            if (layer.configuration.weight_normalization) {
                ::cuda::fill_bytes(stream, layer.normalized_weight_gradients, 0u);
                weight_gradients = layer.normalized_weight_gradients.data();
                gradient_beta    = 0.0F;
            }
            matrix_product(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_T, layer.configuration.output_width, layer.configuration.input_width, source_adjoints.sample_count, 1.0F, current_adjoints, layer.configuration.output_width, layer_input.values, layer.configuration.input_width, gradient_beta, weight_gradients, layer.configuration.output_width);
            for (const std::uint32_t derivative : std::views::iota(0u, source_adjoints.derivative_count)) {
                matrix_product(cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N, layer.configuration.input_width, source_adjoints.sample_count, layer.configuration.output_width, 1.0F, weights, layer.configuration.output_width, current_derivative_adjoints + static_cast<std::size_t>(derivative) * layer.configuration.output_width * source_adjoints.sample_count, layer.configuration.output_width, 0.0F, next_derivative_adjoints + static_cast<std::size_t>(derivative) * layer.configuration.input_width * source_adjoints.sample_count, layer.configuration.input_width);
                matrix_product(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_T, layer.configuration.output_width, layer.configuration.input_width, source_adjoints.sample_count, 1.0F, current_derivative_adjoints + static_cast<std::size_t>(derivative) * layer.configuration.output_width * source_adjoints.sample_count, layer.configuration.output_width, layer_input.derivatives + static_cast<std::size_t>(derivative) * layer.configuration.input_width * source_adjoints.sample_count, layer.configuration.input_width, 1.0F, weight_gradients, layer.configuration.output_width);
            }
            if (layer.configuration.weight_normalization) kernels::weight_normalization_backward(stream, layer.weights.data(), layer.scales.data(), layer.normalized_weight_gradients.data(), layer.weight_gradients.data(), layer.scale_gradients.data(), layer.configuration.input_width, layer.configuration.output_width);

            if (layer.configuration.blended_hidden_layer_count != 0u) {
                const std::size_t value_count = static_cast<std::size_t>(layer.configuration.input_width) * source_adjoints.sample_count;
                ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{next_adjoints, value_count}, ::cuda::std::span<float>{blended_input_adjoints.data(), value_count});
                ::cuda::fill_bytes(stream, ::cuda::std::span<float>{current_adjoints, value_count}, 0u);
                kernels::scaled_add(stream, current_adjoints, blended_input_adjoints.data(), fading_weights.back(), value_count);
                if (source_adjoints.derivative_count != 0u) {
                    ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{next_derivative_adjoints, value_count * source_adjoints.derivative_count}, ::cuda::std::span<float>{blended_input_derivative_adjoints.data(), value_count * source_adjoints.derivative_count});
                    ::cuda::fill_bytes(stream, ::cuda::std::span<float>{current_derivative_adjoints, value_count * source_adjoints.derivative_count}, 0u);
                    kernels::scaled_add(stream, current_derivative_adjoints, blended_input_derivative_adjoints.data(), fading_weights.back(), value_count * source_adjoints.derivative_count);
                }
            } else if (layer.configuration.append_network_input) {
                const std::uint32_t previous_width = layer.configuration.input_width - configuration.input_width;
                kernels::split_adjoint(stream, next_adjoints, next_derivative_adjoints, previous_width, layer.configuration.existing_input_scale, configuration.input_width, layer.configuration.appended_input_scale, current_adjoints, current_derivative_adjoints, network_input_adjoints.data(), source_adjoints.derivative_count == 0u ? nullptr : network_input_derivative_adjoints.data(), source_adjoints.sample_count, source_adjoints.derivative_count);
            } else {
                std::swap(current_adjoints, next_adjoints);
                std::swap(current_derivative_adjoints, next_derivative_adjoints);
            }
            if (blended_head != layers.size() && layer_index < blended_head && layer_index > first_blended_hidden) {
                const std::size_t value_count = static_cast<std::size_t>(layers[layer_index - 1uz].configuration.output_width) * source_adjoints.sample_count;
                kernels::scaled_add(stream, current_adjoints, blended_input_adjoints.data(), fading_weights[layer_index - 1uz - first_blended_hidden], value_count);
                if (source_adjoints.derivative_count != 0u) kernels::scaled_add(stream, current_derivative_adjoints, blended_input_derivative_adjoints.data(), fading_weights[layer_index - 1uz - first_blended_hidden], value_count * source_adjoints.derivative_count);
            }
        }
        kernels::add(stream, network_input_adjoints.data(), current_adjoints, static_cast<std::size_t>(configuration.input_width) * source_adjoints.sample_count);
        if (source_adjoints.derivative_count != 0u) kernels::add(stream, network_input_derivative_adjoints.data(), current_derivative_adjoints, static_cast<std::size_t>(configuration.input_width) * source_adjoints.sample_count * source_adjoints.derivative_count);
        return {.values = network_input_adjoints.data(), .derivatives = source_adjoints.derivative_count == 0u ? nullptr : network_input_derivative_adjoints.data(), .width = configuration.input_width, .sample_count = source_adjoints.sample_count, .derivative_count = source_adjoints.derivative_count};
    }

    void DenseNetwork::clear_gradients() {
        for (Layer& layer : layers) {
            ::cuda::fill_bytes(stream, layer.weight_gradients, 0u);
            ::cuda::fill_bytes(stream, layer.bias_gradients, 0u);
            ::cuda::fill_bytes(stream, layer.scale_gradients, 0u);
        }
    }

    void DenseNetwork::step(const float learning_rate) {
        std::size_t blended_head = layers.size();
        std::size_t first_blended_hidden{};
        std::size_t last_active_hidden{};
        for (const std::size_t layer_index : std::views::iota(0uz, layers.size())) {
            if (layers[layer_index].configuration.blended_hidden_layer_count == 0u) continue;
            blended_head         = layer_index;
            first_blended_hidden = layer_index - layers[layer_index].configuration.blended_hidden_layer_count;
            for (const std::size_t hidden : std::views::iota(0uz, fading_weights.size()))
                if (fading_weights[hidden] > 1.0e-8F) last_active_hidden = hidden;
        }
        for (const std::size_t layer_index : std::views::iota(0uz, layers.size())) {
            if (blended_head != layers.size() && layer_index >= first_blended_hidden && layer_index < blended_head && layer_index - first_blended_hidden > last_active_hidden) continue;
            Layer& layer = layers[layer_index];
            ++layer.step;
            kernels::adam(stream, layer.weights.data(), layer.weight_gradients.data(), layer.weight_first_moments.data(), layer.weight_second_moments.data(), layer.weights.size(), learning_rate, layer.step);
            kernels::adam(stream, layer.biases.data(), layer.bias_gradients.data(), layer.bias_first_moments.data(), layer.bias_second_moments.data(), layer.biases.size(), learning_rate, layer.step);
            if (layer.configuration.weight_normalization) kernels::adam(stream, layer.scales.data(), layer.scale_gradients.data(), layer.scale_first_moments.data(), layer.scale_second_moments.data(), layer.scales.size(), learning_rate, layer.step);
        }
    }

    void DenseNetwork::set_fading_step(const std::uint32_t step, const std::uint32_t final_step) {
        if (fading_weights.empty()) return;
        const float ratio    = std::min(static_cast<float>(step) / static_cast<float>(final_step), 1.0F);
        const float position = 1.0F + static_cast<float>(fading_weights.size() - 2uz) * ratio;
        for (const std::size_t layer : std::views::iota(0uz, fading_weights.size())) fading_weights[layer] = std::clamp(1.0F + position - static_cast<float>(layer), 0.0F, 1.0F) * std::clamp(1.0F + static_cast<float>(layer) - position, 0.0F, 1.0F);
    }

    NetworkState DenseNetwork::download() const {
        NetworkState result;
        std::size_t parameter_count{};
        for (const Layer& layer : layers) parameter_count += layer.weights.size() + layer.biases.size() + layer.scales.size();
        result.parameters.reserve(parameter_count);
        result.first_moments.reserve(parameter_count);
        result.second_moments.reserve(parameter_count);
        result.layer_steps.reserve(layers.size());
        for (const Layer& layer : layers) {
            const auto append_float = [&](const ::cuda::device_buffer<float>& source, std::vector<float>& destination) {
                const std::size_t offset = destination.size();
                destination.resize(offset + source.size());
                ::cuda::copy_bytes(stream, source, ::cuda::std::span<float>{destination.data() + offset, source.size()});
            };
            append_float(layer.weights, result.parameters);
            append_float(layer.biases, result.parameters);
            append_float(layer.scales, result.parameters);
            append_float(layer.weight_first_moments, result.first_moments);
            append_float(layer.bias_first_moments, result.first_moments);
            append_float(layer.scale_first_moments, result.first_moments);
            append_float(layer.weight_second_moments, result.second_moments);
            append_float(layer.bias_second_moments, result.second_moments);
            append_float(layer.scale_second_moments, result.second_moments);
            result.layer_steps.push_back(layer.step);
        }
        stream.sync();
        return result;
    }

    void DenseNetwork::upload(const NetworkState& state) {
        std::size_t offset{};
        for (Layer& layer : layers) {
            const auto copy_float = [&](::cuda::device_buffer<float>& destination, const std::vector<float>& source) {
                ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{source.data() + offset, destination.size()}, destination);
                offset += destination.size();
            };
            copy_float(layer.weights, state.parameters);
            copy_float(layer.biases, state.parameters);
            copy_float(layer.scales, state.parameters);
        }
        offset = 0uz;
        for (Layer& layer : layers) {
            const auto copy_float = [&](::cuda::device_buffer<float>& destination, const std::vector<float>& source) {
                ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{source.data() + offset, destination.size()}, destination);
                offset += destination.size();
            };
            copy_float(layer.weight_first_moments, state.first_moments);
            copy_float(layer.bias_first_moments, state.first_moments);
            copy_float(layer.scale_first_moments, state.first_moments);
        }
        offset = 0uz;
        for (Layer& layer : layers) {
            const auto copy_float = [&](::cuda::device_buffer<float>& destination, const std::vector<float>& source) {
                ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{source.data() + offset, destination.size()}, destination);
                offset += destination.size();
            };
            copy_float(layer.weight_second_moments, state.second_moments);
            copy_float(layer.bias_second_moments, state.second_moments);
            copy_float(layer.scale_second_moments, state.second_moments);
        }
        for (const std::size_t layer : std::views::iota(0uz, layers.size())) layers[layer].step = state.layer_steps[layer];
    }

    DenseNetwork::Layer::Layer(const ::cuda::stream_ref stream, const LayerConfiguration source_configuration, const std::uint32_t capacity, const std::uint32_t derivative_capacity, const std::uint32_t maximum_derivative_count)
        : configuration{source_configuration}, weights{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.input_width) * configuration.output_width, ::cuda::no_init}, normalized_weights{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.weight_normalization ? weights.size() : 0uz, ::cuda::no_init}, biases{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.output_width, ::cuda::no_init}, scales{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.weight_normalization ? configuration.output_width : 0uz, ::cuda::no_init}, weight_gradients{stream, ::cuda::device_default_memory_pool(stream.device()), weights.size(), ::cuda::no_init}, bias_gradients{stream, ::cuda::device_default_memory_pool(stream.device()), biases.size(), ::cuda::no_init}, scale_gradients{stream, ::cuda::device_default_memory_pool(stream.device()), scales.size(), ::cuda::no_init},
          normalized_weight_gradients{stream, ::cuda::device_default_memory_pool(stream.device()), normalized_weights.size(), ::cuda::no_init}, weight_first_moments{stream, ::cuda::device_default_memory_pool(stream.device()), weights.size(), ::cuda::no_init}, bias_first_moments{stream, ::cuda::device_default_memory_pool(stream.device()), biases.size(), ::cuda::no_init}, scale_first_moments{stream, ::cuda::device_default_memory_pool(stream.device()), scales.size(), ::cuda::no_init}, weight_second_moments{stream, ::cuda::device_default_memory_pool(stream.device()), weights.size(), ::cuda::no_init}, bias_second_moments{stream, ::cuda::device_default_memory_pool(stream.device()), biases.size(), ::cuda::no_init}, scale_second_moments{stream, ::cuda::device_default_memory_pool(stream.device()), scales.size(), ::cuda::no_init}, inputs{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.append_network_input ? static_cast<std::size_t>(configuration.input_width) * capacity : 0uz, ::cuda::no_init}, input_derivatives{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.append_network_input ? static_cast<std::size_t>(configuration.input_width) * derivative_capacity * maximum_derivative_count : 0uz, ::cuda::no_init}, linear{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.output_width) * capacity, ::cuda::no_init}, linear_derivatives{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.output_width) * derivative_capacity * maximum_derivative_count, ::cuda::no_init},
          outputs{stream, ::cuda::device_default_memory_pool(stream.device()), linear.size(), ::cuda::no_init}, output_derivatives{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(configuration.output_width) * derivative_capacity * maximum_derivative_count, ::cuda::no_init} {}

    void DenseNetwork::CublasDeleter::operator()(cublasContext* const handle) const noexcept {
        cublasDestroy(handle);
    }
} // namespace physica::reconstruction::pinfs
