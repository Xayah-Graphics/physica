module;

#include "kernels.h"
#include <physica/cuda.h>

module physica.generative.flow_matching.model;

import std;
import physica.neural.matmul;
import physica.neural.transformer;

namespace physica::generative::flow_matching {
    namespace {
        std::size_t align_workspace(const std::size_t value) {
            return (value + 255uz) & ~255uz;
        }

        template <class Type>
        Type* workspace_pointer(std::uint8_t* const workspace, const std::size_t offset) {
            return reinterpret_cast<Type*>(workspace + offset);
        }

        void initialize_xavier(std::vector<float>& values, const std::size_t offset, const std::size_t count, const std::uint32_t input_width, const std::uint32_t output_width, std::mt19937_64& generator) {
            const float extent = std::sqrt(6.0F / static_cast<float>(input_width + output_width));
            std::uniform_real_distribution<float> distribution{-extent, extent};
            for (std::size_t index = 0uz; index < count; ++index) values[offset + index] = distribution(generator);
        }
    } // namespace

    FlowDiTParameterLayout::FlowDiTParameterLayout() {
        constexpr std::size_t width = flow_transformer_configuration.width;
        std::size_t offset{};
        patch_weight = offset;
        offset += 12uz * width;
        patch_bias = offset;
        offset += width;
        class_embedding = offset;
        offset += 11uz * width;
        time_input_weight = offset;
        offset += width * width;
        time_input_bias = offset;
        offset += width;
        time_output_weight = offset;
        offset += width * width;
        time_output_bias = offset;
        offset += width;
        transformer = offset;
        offset += neural::TransformerParameterLayout{flow_transformer_configuration}.parameter_count;
        final_modulation_weight = offset;
        offset += width * 2uz * width;
        final_modulation_bias = offset;
        offset += 2uz * width;
        velocity_weight = offset;
        offset += width * 12uz;
        velocity_bias   = offset;
        parameter_count = offset + 12uz;
    }

    FlowDiTWorkspaceLayout::FlowDiTWorkspaceLayout(const std::uint32_t source_batch) : batch{source_batch}, transformer_layout{flow_transformer_configuration, batch} {
        constexpr std::size_t sequence = flow_transformer_configuration.sequence;
        constexpr std::size_t width    = flow_transformer_configuration.width;
        const std::size_t token_count  = static_cast<std::size_t>(batch) * sequence;
        std::size_t offset{};
        tokens                            = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        time_embedding                    = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        time_preactivation                = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        time_hidden                       = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        condition                         = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        condition_activated               = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        transformed                       = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        final_modulation                  = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * 2uz * width * sizeof(float));
        final_normalized                  = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        final_means                       = offset;
        offset                            = align_workspace(offset + token_count * sizeof(float));
        final_inverse_standard_deviations = offset;
        offset                            = align_workspace(offset + token_count * sizeof(float));
        velocity                          = offset;
        offset                            = align_workspace(offset + token_count * 12uz * sizeof(float));
        velocity_gradient                 = offset;
        offset                            = align_workspace(offset + token_count * 12uz * sizeof(float));
        normalized_gradient               = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        transformed_gradient              = offset;
        offset                            = align_workspace(offset + token_count * width * sizeof(float));
        final_modulation_gradient         = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * 2uz * width * sizeof(float));
        condition_gradient                = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        time_hidden_gradient              = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        time_preactivation_gradient       = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * width * sizeof(float));
        sample_loss                       = offset;
        offset                            = align_workspace(offset + static_cast<std::size_t>(batch) * sizeof(float));
        transformer_workspace             = offset;
        byte_count                        = align_workspace(offset + transformer_layout.byte_count);
    }

    FlowDiT::FlowDiT(const ::cuda::stream_ref source_stream, neural::MatmulRuntime& source_matmul) : stream{source_stream}, matmul{source_matmul}, transformer{matmul, flow_transformer_configuration}, position{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(flow_transformer_configuration.sequence) * flow_transformer_configuration.width, ::cuda::no_init} {
        std::vector<float> host_position(position.size());
        constexpr std::uint32_t frequency_count = flow_transformer_configuration.width / 4u;
        for (std::uint32_t y = 0u; y < 16u; ++y)
            for (std::uint32_t x = 0u; x < 16u; ++x)
                for (std::uint32_t frequency = 0u; frequency < frequency_count; ++frequency) {
                    const float scale                                         = std::exp(-std::log(10'000.0F) * static_cast<float>(frequency) / static_cast<float>(frequency_count));
                    const std::size_t offset                                  = static_cast<std::size_t>(y * 16u + x) * flow_transformer_configuration.width;
                    host_position[offset + frequency]                         = std::sin(static_cast<float>(y) * scale);
                    host_position[offset + frequency_count + frequency]       = std::cos(static_cast<float>(y) * scale);
                    host_position[offset + 2uz * frequency_count + frequency] = std::sin(static_cast<float>(x) * scale);
                    host_position[offset + 3uz * frequency_count + frequency] = std::cos(static_cast<float>(x) * scale);
                }
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{host_position.data(), host_position.size()}, position);
        stream.sync();
    }

    std::vector<float> FlowDiT::initialize_parameters(const std::uint64_t seed) const {
        std::vector<float> values(parameters.parameter_count);
        std::mt19937_64 generator{seed};
        initialize_xavier(values, parameters.patch_weight, 12uz * flow_transformer_configuration.width, 12u, flow_transformer_configuration.width, generator);
        std::normal_distribution<float> small_normal{0.0F, 0.02F};
        for (std::size_t index = parameters.class_embedding; index < parameters.class_embedding + 11uz * flow_transformer_configuration.width; ++index) values[index] = small_normal(generator);
        for (std::size_t index = parameters.time_input_weight; index < parameters.time_input_weight + static_cast<std::size_t>(flow_transformer_configuration.width) * flow_transformer_configuration.width; ++index) values[index] = small_normal(generator);
        for (std::size_t index = parameters.time_output_weight; index < parameters.time_output_weight + static_cast<std::size_t>(flow_transformer_configuration.width) * flow_transformer_configuration.width; ++index) values[index] = small_normal(generator);
        const std::vector<float> transformer_values = transformer.initialize_parameters(seed + 1u);
        std::ranges::copy(transformer_values, values.begin() + static_cast<std::ptrdiff_t>(parameters.transformer));
        return values;
    }

    void FlowDiT::forward(const float* const parameter_values, const float* const patches, const float* const times, const std::uint8_t* const labels, std::uint8_t* const workspace, const FlowDiTWorkspaceLayout& layout) {
        const std::uint32_t token_count = layout.batch * flow_transformer_configuration.sequence;
        float* tokens                   = workspace_pointer<float>(workspace, layout.tokens);
        float* time_embedding           = workspace_pointer<float>(workspace, layout.time_embedding);
        float* time_preactivation       = workspace_pointer<float>(workspace, layout.time_preactivation);
        float* time_hidden              = workspace_pointer<float>(workspace, layout.time_hidden);
        float* condition                = workspace_pointer<float>(workspace, layout.condition);
        float* condition_activated      = workspace_pointer<float>(workspace, layout.condition_activated);
        float* transformed              = workspace_pointer<float>(workspace, layout.transformed);
        float* final_modulation         = workspace_pointer<float>(workspace, layout.final_modulation);
        float* final_normalized         = workspace_pointer<float>(workspace, layout.final_normalized);
        float* velocity                 = workspace_pointer<float>(workspace, layout.velocity);

        matmul.execute({patches, parameter_values + parameters.patch_weight, tokens, token_count, flow_transformer_configuration.width, 12u, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.patch_bias});
        kernels::add_position(stream, tokens, position.data(), static_cast<std::size_t>(token_count) * flow_transformer_configuration.width);
        kernels::make_time_embedding(stream, times, time_embedding, layout.batch, flow_transformer_configuration.width);
        matmul.execute({time_embedding, parameter_values + parameters.time_input_weight, time_preactivation, layout.batch, flow_transformer_configuration.width, flow_transformer_configuration.width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.time_input_bias});
        kernels::silu_forward(stream, time_preactivation, time_hidden, static_cast<std::size_t>(layout.batch) * flow_transformer_configuration.width);
        matmul.execute({time_hidden, parameter_values + parameters.time_output_weight, condition, layout.batch, flow_transformer_configuration.width, flow_transformer_configuration.width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.time_output_bias});
        kernels::make_condition(stream, condition, parameter_values + parameters.class_embedding, labels, layout.batch, flow_transformer_configuration.width);
        kernels::silu_forward(stream, condition, condition_activated, static_cast<std::size_t>(layout.batch) * flow_transformer_configuration.width);
        transformer.forward(parameter_values + parameters.transformer, tokens, condition_activated, transformed, workspace + layout.transformer_workspace, layout.transformer_layout);
        matmul.execute({condition_activated, parameter_values + parameters.final_modulation_weight, final_modulation, layout.batch, 2u * flow_transformer_configuration.width, flow_transformer_configuration.width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.final_modulation_bias});
        kernels::final_adaln_forward(stream, transformed, final_modulation, final_normalized, workspace_pointer<float>(workspace, layout.final_means), workspace_pointer<float>(workspace, layout.final_inverse_standard_deviations), layout.batch, flow_transformer_configuration.sequence, flow_transformer_configuration.width);
        matmul.execute({final_normalized, parameter_values + parameters.velocity_weight, velocity, token_count, 12u, flow_transformer_configuration.width, false, false, neural::MatmulEpilogue::bias, parameter_values + parameters.velocity_bias});
    }

    void FlowDiT::loss(const float* const target, float* const loss_value, std::uint8_t* const workspace, const FlowDiTWorkspaceLayout& layout) {
        kernels::flow_matching_loss(stream, workspace_pointer<float>(workspace, layout.velocity), target, workspace_pointer<float>(workspace, layout.velocity_gradient), workspace_pointer<float>(workspace, layout.sample_loss), loss_value, layout.batch);
    }

    void FlowDiT::backward(const float* const parameter_values, float* const parameter_gradients, const float* const patches, const float*, const std::uint8_t* const labels, float* const input_patch_gradient, std::uint8_t* const workspace, const FlowDiTWorkspaceLayout& layout) {
        const std::uint32_t token_count    = layout.batch * flow_transformer_configuration.sequence;
        const float* tokens                = workspace_pointer<float>(workspace, layout.tokens);
        const float* time_embedding        = workspace_pointer<float>(workspace, layout.time_embedding);
        const float* time_preactivation    = workspace_pointer<float>(workspace, layout.time_preactivation);
        const float* time_hidden           = workspace_pointer<float>(workspace, layout.time_hidden);
        const float* condition             = workspace_pointer<float>(workspace, layout.condition);
        const float* condition_activated   = workspace_pointer<float>(workspace, layout.condition_activated);
        const float* transformed           = workspace_pointer<float>(workspace, layout.transformed);
        const float* final_modulation      = workspace_pointer<float>(workspace, layout.final_modulation);
        const float* final_normalized      = workspace_pointer<float>(workspace, layout.final_normalized);
        const float* velocity_gradient     = workspace_pointer<float>(workspace, layout.velocity_gradient);
        float* normalized_gradient         = workspace_pointer<float>(workspace, layout.normalized_gradient);
        float* transformed_gradient        = workspace_pointer<float>(workspace, layout.transformed_gradient);
        float* final_modulation_gradient   = workspace_pointer<float>(workspace, layout.final_modulation_gradient);
        float* condition_gradient          = workspace_pointer<float>(workspace, layout.condition_gradient);
        float* time_hidden_gradient        = workspace_pointer<float>(workspace, layout.time_hidden_gradient);
        float* time_preactivation_gradient = workspace_pointer<float>(workspace, layout.time_preactivation_gradient);
        ::cuda::fill_bytes(stream, ::cuda::std::span<float>{condition_gradient, static_cast<std::size_t>(layout.batch) * flow_transformer_configuration.width}, 0u);

        matmul.execute({final_normalized, velocity_gradient, parameter_gradients + parameters.velocity_weight, flow_transformer_configuration.width, 12u, token_count, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.velocity_bias});
        matmul.execute({velocity_gradient, parameter_values + parameters.velocity_weight, normalized_gradient, token_count, flow_transformer_configuration.width, 12u, false, true});
        kernels::final_adaln_backward(stream, transformed, final_modulation, normalized_gradient, workspace_pointer<float>(workspace, layout.final_means), workspace_pointer<float>(workspace, layout.final_inverse_standard_deviations), transformed_gradient, final_modulation_gradient, layout.batch, flow_transformer_configuration.sequence, flow_transformer_configuration.width);
        matmul.execute({condition_activated, final_modulation_gradient, parameter_gradients + parameters.final_modulation_weight, flow_transformer_configuration.width, 2u * flow_transformer_configuration.width, layout.batch, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.final_modulation_bias});
        matmul.execute({final_modulation_gradient, parameter_values + parameters.final_modulation_weight, condition_gradient, layout.batch, flow_transformer_configuration.width, 2u * flow_transformer_configuration.width, false, true});
        transformer.backward(parameter_values + parameters.transformer, parameter_gradients + parameters.transformer, tokens, condition_activated, transformed_gradient, normalized_gradient, condition_gradient, workspace + layout.transformer_workspace, layout.transformer_layout);
        kernels::silu_backward(stream, condition, condition_gradient, time_preactivation_gradient, static_cast<std::size_t>(layout.batch) * flow_transformer_configuration.width);
        kernels::class_embedding_backward(stream, time_preactivation_gradient, labels, parameter_gradients + parameters.class_embedding, layout.batch, flow_transformer_configuration.width);
        matmul.execute({time_hidden, time_preactivation_gradient, parameter_gradients + parameters.time_output_weight, flow_transformer_configuration.width, flow_transformer_configuration.width, layout.batch, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.time_output_bias});
        matmul.execute({time_preactivation_gradient, parameter_values + parameters.time_output_weight, time_hidden_gradient, layout.batch, flow_transformer_configuration.width, flow_transformer_configuration.width, false, true});
        kernels::silu_backward(stream, time_preactivation, time_hidden_gradient, condition_gradient, static_cast<std::size_t>(layout.batch) * flow_transformer_configuration.width);
        matmul.execute({time_embedding, condition_gradient, parameter_gradients + parameters.time_input_weight, flow_transformer_configuration.width, flow_transformer_configuration.width, layout.batch, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.time_input_bias});
        matmul.execute({patches, normalized_gradient, parameter_gradients + parameters.patch_weight, 12u, flow_transformer_configuration.width, token_count, true, false, neural::MatmulEpilogue::bias_gradient, parameter_gradients + parameters.patch_bias});
        matmul.execute({normalized_gradient, parameter_values + parameters.patch_weight, input_patch_gradient, token_count, 12u, flow_transformer_configuration.width, false, true});
    }
} // namespace physica::generative::flow_matching
