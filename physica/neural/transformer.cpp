module;

#include "transformer-kernels.h"
#include <physica/cuda.h>

module physica.neural.transformer;

import std;
import physica.neural.matmul;

namespace physica::neural {
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

    TransformerParameterLayout::TransformerParameterLayout(const TransformerConfiguration& source_configuration) : configuration{source_configuration}, blocks(configuration.block_count) {
        std::size_t offset{};
        for (TransformerBlockParameterLayout& block : blocks) {
            block.modulation_weight = offset;
            offset += static_cast<std::size_t>(configuration.width) * 6uz * configuration.width;
            block.modulation_bias = offset;
            offset += 6uz * configuration.width;
            block.qkv_weight = offset;
            offset += static_cast<std::size_t>(configuration.width) * 3uz * configuration.width;
            block.qkv_bias = offset;
            offset += 3uz * configuration.width;
            block.attention_output_weight = offset;
            offset += static_cast<std::size_t>(configuration.width) * configuration.width;
            block.attention_output_bias = offset;
            offset += configuration.width;
            block.mlp_input_weight = offset;
            offset += static_cast<std::size_t>(configuration.width) * configuration.mlp_width;
            block.mlp_input_bias = offset;
            offset += configuration.mlp_width;
            block.mlp_output_weight = offset;
            offset += static_cast<std::size_t>(configuration.mlp_width) * configuration.width;
            block.mlp_output_bias = offset;
            offset += configuration.width;
        }
        parameter_count = offset;
    }

    TransformerWorkspaceLayout::TransformerWorkspaceLayout(const TransformerConfiguration& source_configuration, const std::uint32_t source_batch) : configuration{source_configuration}, batch{source_batch}, blocks(configuration.block_count) {
        const std::size_t token_count = static_cast<std::size_t>(batch) * configuration.sequence;
        std::size_t offset{};
        for (std::uint32_t index = 0u; index < configuration.block_count; ++index) {
            TransformerBlockWorkspaceLayout& block      = blocks[index];
            block.modulation                            = offset;
            offset                                      = align_workspace(offset + static_cast<std::size_t>(batch) * 6uz * configuration.width * sizeof(float));
            block.attention_normalized                  = offset;
            offset                                      = align_workspace(offset + token_count * configuration.width * sizeof(float));
            block.attention_means                       = offset;
            offset                                      = align_workspace(offset + token_count * sizeof(float));
            block.attention_inverse_standard_deviations = offset;
            offset                                      = align_workspace(offset + token_count * sizeof(float));
            block.qkv                                   = offset;
            offset                                      = align_workspace(offset + token_count * 3uz * configuration.width * sizeof(float));
            block.attention                             = offset;
            offset                                      = align_workspace(offset + token_count * configuration.width * sizeof(float));
            block.attention_log_sum_exp                 = offset;
            offset                                      = align_workspace(offset + static_cast<std::size_t>(batch) * configuration.head_count * configuration.sequence * sizeof(float));
            block.attention_projected                   = offset;
            offset                                      = align_workspace(offset + token_count * configuration.width * sizeof(float));
            block.after_attention                       = offset;
            offset                                      = align_workspace(offset + token_count * configuration.width * sizeof(float));
            block.mlp_normalized                        = offset;
            offset                                      = align_workspace(offset + token_count * configuration.width * sizeof(float));
            block.mlp_means                             = offset;
            offset                                      = align_workspace(offset + token_count * sizeof(float));
            block.mlp_inverse_standard_deviations       = offset;
            offset                                      = align_workspace(offset + token_count * sizeof(float));
            block.mlp_preactivation                     = offset;
            offset                                      = align_workspace(offset + token_count * configuration.mlp_width * sizeof(float));
            block.mlp_hidden                            = offset;
            offset                                      = align_workspace(offset + token_count * configuration.mlp_width * sizeof(float));
            block.mlp_projected                         = offset;
            offset                                      = align_workspace(offset + token_count * configuration.width * sizeof(float));
            block.output                                = offset;
            if (index + 1u < configuration.block_count) offset = align_workspace(offset + token_count * configuration.width * sizeof(float));
        }
        gradient_a                 = offset;
        offset                     = align_workspace(offset + token_count * configuration.width * sizeof(float));
        gradient_b                 = offset;
        offset                     = align_workspace(offset + token_count * configuration.width * sizeof(float));
        branch_gradient            = offset;
        offset                     = align_workspace(offset + token_count * configuration.width * sizeof(float));
        normalized_gradient        = offset;
        offset                     = align_workspace(offset + token_count * configuration.width * sizeof(float));
        modulation_gradient        = offset;
        offset                     = align_workspace(offset + static_cast<std::size_t>(batch) * 6uz * configuration.width * sizeof(float));
        qkv_gradient               = offset;
        offset                     = align_workspace(offset + token_count * 3uz * configuration.width * sizeof(float));
        attention_delta            = offset;
        offset                     = align_workspace(offset + static_cast<std::size_t>(batch) * configuration.head_count * configuration.sequence * sizeof(float));
        mlp_preactivation_gradient = offset;
        byte_count                 = align_workspace(offset + token_count * configuration.mlp_width * sizeof(float));
    }

    Transformer::Transformer(MatmulRuntime& source_matmul, const TransformerConfiguration& configuration) : matmul{source_matmul}, parameters{configuration} {}

    std::vector<float> Transformer::initialize_parameters(const std::uint64_t seed) const {
        std::vector<float> values(parameters.parameter_count);
        std::mt19937_64 generator{seed};
        const std::uint32_t width = parameters.configuration.width;
        for (const TransformerBlockParameterLayout& block : parameters.blocks) {
            initialize_xavier(values, block.qkv_weight, static_cast<std::size_t>(width) * 3uz * width, width, 3u * width, generator);
            initialize_xavier(values, block.attention_output_weight, static_cast<std::size_t>(width) * width, width, width, generator);
            initialize_xavier(values, block.mlp_input_weight, static_cast<std::size_t>(width) * parameters.configuration.mlp_width, width, parameters.configuration.mlp_width, generator);
            initialize_xavier(values, block.mlp_output_weight, static_cast<std::size_t>(parameters.configuration.mlp_width) * width, parameters.configuration.mlp_width, width, generator);
        }
        return values;
    }

    void Transformer::forward(const float* const parameter_values, const float* const tokens, const float* const condition, float* const output, std::uint8_t* const workspace, const TransformerWorkspaceLayout& workspace_layout) {
        const TransformerConfiguration& configuration = parameters.configuration;
        const std::uint32_t token_count               = workspace_layout.batch * configuration.sequence;
        for (std::uint32_t index = 0u; index < configuration.block_count; ++index) {
            const TransformerBlockParameterLayout& parameter = parameters.blocks[index];
            const TransformerBlockWorkspaceLayout& block     = workspace_layout.blocks[index];
            const float* input                               = index == 0u ? tokens : workspace_pointer<float>(workspace, workspace_layout.blocks[index - 1u].output);
            float* modulation                                = workspace_pointer<float>(workspace, block.modulation);
            float* attention_normalized                      = workspace_pointer<float>(workspace, block.attention_normalized);
            float* qkv                                       = workspace_pointer<float>(workspace, block.qkv);
            float* attention                                 = workspace_pointer<float>(workspace, block.attention);
            float* attention_projected                       = workspace_pointer<float>(workspace, block.attention_projected);
            float* after_attention                           = workspace_pointer<float>(workspace, block.after_attention);
            float* mlp_normalized                            = workspace_pointer<float>(workspace, block.mlp_normalized);
            float* mlp_preactivation                         = workspace_pointer<float>(workspace, block.mlp_preactivation);
            float* mlp_hidden                                = workspace_pointer<float>(workspace, block.mlp_hidden);
            float* mlp_projected                             = workspace_pointer<float>(workspace, block.mlp_projected);
            float* block_output                              = index + 1u == configuration.block_count ? output : workspace_pointer<float>(workspace, block.output);

            matmul.execute({condition, parameter_values + parameter.modulation_weight, modulation, workspace_layout.batch, 6u * configuration.width, configuration.width, false, false, MatmulEpilogue::bias, parameter_values + parameter.modulation_bias});
            kernels::adaln_forward(matmul.stream, input, modulation, attention_normalized, workspace_pointer<float>(workspace, block.attention_means), workspace_pointer<float>(workspace, block.attention_inverse_standard_deviations), workspace_layout.batch, configuration.sequence, configuration.width, 0u);
            matmul.execute({attention_normalized, parameter_values + parameter.qkv_weight, qkv, token_count, 3u * configuration.width, configuration.width, false, false, MatmulEpilogue::bias, parameter_values + parameter.qkv_bias});
            kernels::sdpa_forward(matmul.stream, qkv, attention, workspace_pointer<float>(workspace, block.attention_log_sum_exp), workspace_layout.batch, configuration.sequence, configuration.width, configuration.head_count);
            matmul.execute({attention, parameter_values + parameter.attention_output_weight, attention_projected, token_count, configuration.width, configuration.width, false, false, MatmulEpilogue::bias, parameter_values + parameter.attention_output_bias});
            kernels::residual_forward(matmul.stream, input, attention_projected, modulation, after_attention, workspace_layout.batch, configuration.sequence, configuration.width, 0u);
            kernels::adaln_forward(matmul.stream, after_attention, modulation, mlp_normalized, workspace_pointer<float>(workspace, block.mlp_means), workspace_pointer<float>(workspace, block.mlp_inverse_standard_deviations), workspace_layout.batch, configuration.sequence, configuration.width, 3u);
            matmul.execute({mlp_normalized, parameter_values + parameter.mlp_input_weight, mlp_hidden, token_count, configuration.mlp_width, configuration.width, false, false, MatmulEpilogue::gelu_aux_bias, parameter_values + parameter.mlp_input_bias, 0.0F, mlp_preactivation});
            matmul.execute({mlp_hidden, parameter_values + parameter.mlp_output_weight, mlp_projected, token_count, configuration.width, configuration.mlp_width, false, false, MatmulEpilogue::bias, parameter_values + parameter.mlp_output_bias});
            kernels::residual_forward(matmul.stream, after_attention, mlp_projected, modulation, block_output, workspace_layout.batch, configuration.sequence, configuration.width, 3u);
        }
    }

    void Transformer::backward(const float* const parameter_values, float* const parameter_gradients, const float* const tokens, const float* const condition, const float* const output_gradient, float* const token_gradient, float* const condition_gradient, std::uint8_t* const workspace, const TransformerWorkspaceLayout& workspace_layout) {
        const TransformerConfiguration& configuration = parameters.configuration;
        const std::uint32_t token_count               = workspace_layout.batch * configuration.sequence;
        float* gradient_a                             = workspace_pointer<float>(workspace, workspace_layout.gradient_a);
        float* gradient_b                             = workspace_pointer<float>(workspace, workspace_layout.gradient_b);
        float* branch_gradient                        = workspace_pointer<float>(workspace, workspace_layout.branch_gradient);
        float* normalized_gradient                    = workspace_pointer<float>(workspace, workspace_layout.normalized_gradient);
        float* modulation_gradient                    = workspace_pointer<float>(workspace, workspace_layout.modulation_gradient);
        float* qkv_gradient                           = workspace_pointer<float>(workspace, workspace_layout.qkv_gradient);
        float* attention_delta                        = workspace_pointer<float>(workspace, workspace_layout.attention_delta);
        float* mlp_preactivation_gradient             = workspace_pointer<float>(workspace, workspace_layout.mlp_preactivation_gradient);
        const float* current_gradient                 = output_gradient;
        for (std::uint32_t reverse = configuration.block_count; reverse != 0u; --reverse) {
            const std::uint32_t index                        = reverse - 1u;
            const TransformerBlockParameterLayout& parameter = parameters.blocks[index];
            const TransformerBlockWorkspaceLayout& block     = workspace_layout.blocks[index];
            const float* input                               = index == 0u ? tokens : workspace_pointer<float>(workspace, workspace_layout.blocks[index - 1u].output);
            const float* modulation                          = workspace_pointer<float>(workspace, block.modulation);
            const float* attention_normalized                = workspace_pointer<float>(workspace, block.attention_normalized);
            const float* qkv                                 = workspace_pointer<float>(workspace, block.qkv);
            const float* attention                           = workspace_pointer<float>(workspace, block.attention);
            const float* attention_projected                 = workspace_pointer<float>(workspace, block.attention_projected);
            const float* after_attention                     = workspace_pointer<float>(workspace, block.after_attention);
            const float* mlp_normalized                      = workspace_pointer<float>(workspace, block.mlp_normalized);
            const float* mlp_preactivation                   = workspace_pointer<float>(workspace, block.mlp_preactivation);
            const float* mlp_hidden                          = workspace_pointer<float>(workspace, block.mlp_hidden);
            const float* mlp_projected                       = workspace_pointer<float>(workspace, block.mlp_projected);
            float* after_attention_gradient                  = index % 2u == 0u ? gradient_a : gradient_b;
            float* input_gradient                            = index == 0u ? token_gradient : (index % 2u == 0u ? gradient_b : gradient_a);

            kernels::residual_backward(matmul.stream, current_gradient, mlp_projected, modulation, branch_gradient, modulation_gradient, workspace_layout.batch, configuration.sequence, configuration.width, 3u);
            matmul.execute({mlp_hidden, branch_gradient, parameter_gradients + parameter.mlp_output_weight, configuration.mlp_width, configuration.width, token_count, true, false, MatmulEpilogue::bias_gradient, parameter_gradients + parameter.mlp_output_bias});
            matmul.execute({branch_gradient, parameter_values + parameter.mlp_output_weight, mlp_preactivation_gradient, token_count, configuration.mlp_width, configuration.width, false, true, MatmulEpilogue::gelu_gradient, nullptr, 0.0F, mlp_preactivation});
            matmul.execute({mlp_normalized, mlp_preactivation_gradient, parameter_gradients + parameter.mlp_input_weight, configuration.width, configuration.mlp_width, token_count, true, false, MatmulEpilogue::bias_gradient, parameter_gradients + parameter.mlp_input_bias});
            matmul.execute({mlp_preactivation_gradient, parameter_values + parameter.mlp_input_weight, normalized_gradient, token_count, configuration.width, configuration.mlp_width, false, true});
            kernels::adaln_backward(matmul.stream, after_attention, modulation, normalized_gradient, current_gradient, workspace_pointer<float>(workspace, block.mlp_means), workspace_pointer<float>(workspace, block.mlp_inverse_standard_deviations), after_attention_gradient, modulation_gradient, workspace_layout.batch, configuration.sequence, configuration.width, 3u);

            kernels::residual_backward(matmul.stream, after_attention_gradient, attention_projected, modulation, branch_gradient, modulation_gradient, workspace_layout.batch, configuration.sequence, configuration.width, 0u);
            matmul.execute({attention, branch_gradient, parameter_gradients + parameter.attention_output_weight, configuration.width, configuration.width, token_count, true, false, MatmulEpilogue::bias_gradient, parameter_gradients + parameter.attention_output_bias});
            matmul.execute({branch_gradient, parameter_values + parameter.attention_output_weight, normalized_gradient, token_count, configuration.width, configuration.width, false, true});
            kernels::sdpa_backward(matmul.stream, qkv, attention, normalized_gradient, workspace_pointer<float>(workspace, block.attention_log_sum_exp), attention_delta, qkv_gradient, workspace_layout.batch, configuration.sequence, configuration.width, configuration.head_count);
            matmul.execute({attention_normalized, qkv_gradient, parameter_gradients + parameter.qkv_weight, configuration.width, 3u * configuration.width, token_count, true, false, MatmulEpilogue::bias_gradient, parameter_gradients + parameter.qkv_bias});
            ::cuda::fill_bytes(matmul.stream, ::cuda::std::span<float>{parameter_gradients + parameter.qkv_bias + configuration.width, configuration.width}, 0u);
            matmul.execute({qkv_gradient, parameter_values + parameter.qkv_weight, normalized_gradient, token_count, configuration.width, 3u * configuration.width, false, true});
            kernels::adaln_backward(matmul.stream, input, modulation, normalized_gradient, after_attention_gradient, workspace_pointer<float>(workspace, block.attention_means), workspace_pointer<float>(workspace, block.attention_inverse_standard_deviations), input_gradient, modulation_gradient, workspace_layout.batch, configuration.sequence, configuration.width, 0u);

            matmul.execute({condition, modulation_gradient, parameter_gradients + parameter.modulation_weight, configuration.width, 6u * configuration.width, workspace_layout.batch, true, false, MatmulEpilogue::bias_gradient, parameter_gradients + parameter.modulation_bias});
            matmul.execute({modulation_gradient, parameter_values + parameter.modulation_weight, condition_gradient, workspace_layout.batch, configuration.width, 6u * configuration.width, false, true, MatmulEpilogue::none, nullptr, 1.0F});
            current_gradient = input_gradient;
        }
    }
} // namespace physica::neural
