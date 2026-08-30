module;

#include <physica/cuda.h>

export module physica.generative.flow_matching.model;

import std;
import physica.neural.matmul;
import physica.neural.transformer;

export namespace physica::generative::flow_matching {
    inline constexpr neural::MatmulRuntimeConfiguration flow_matmul_runtime_configuration{
        .workspace_byte_count   = 64uz * 1024uz * 1024uz,
        .tuning_byte_count      = 256uz * 1024uz * 1024uz,
        .tuning_bias_byte_count = 64uz * 1024uz,
    };

    inline constexpr neural::TransformerConfiguration flow_transformer_configuration{
        .sequence    = 256u,
        .width       = 256u,
        .block_count = 8u,
        .head_count  = 8u,
        .mlp_width   = 1024u,
    };

    struct FlowDiTParameterLayout final {
        std::size_t patch_weight;
        std::size_t patch_bias;
        std::size_t class_embedding;
        std::size_t time_input_weight;
        std::size_t time_input_bias;
        std::size_t time_output_weight;
        std::size_t time_output_bias;
        std::size_t transformer;
        std::size_t final_modulation_weight;
        std::size_t final_modulation_bias;
        std::size_t velocity_weight;
        std::size_t velocity_bias;
        std::size_t parameter_count;

        FlowDiTParameterLayout();
    };

    struct FlowDiTWorkspaceLayout final {
        std::uint32_t batch;
        std::size_t tokens;
        std::size_t time_embedding;
        std::size_t time_preactivation;
        std::size_t time_hidden;
        std::size_t condition;
        std::size_t condition_activated;
        std::size_t transformed;
        std::size_t final_modulation;
        std::size_t final_normalized;
        std::size_t final_means;
        std::size_t final_inverse_standard_deviations;
        std::size_t velocity;
        std::size_t velocity_gradient;
        std::size_t normalized_gradient;
        std::size_t transformed_gradient;
        std::size_t final_modulation_gradient;
        std::size_t condition_gradient;
        std::size_t time_hidden_gradient;
        std::size_t time_preactivation_gradient;
        std::size_t sample_loss;
        std::size_t transformer_workspace;
        neural::TransformerWorkspaceLayout transformer_layout;
        std::size_t byte_count;

        explicit FlowDiTWorkspaceLayout(std::uint32_t batch);
    };

    struct FlowDiT final {
        ::cuda::stream_ref stream;
        neural::MatmulRuntime& matmul;
        FlowDiTParameterLayout parameters;
        neural::Transformer transformer;
        ::cuda::device_buffer<float> position;

        FlowDiT(::cuda::stream_ref stream, neural::MatmulRuntime& matmul);

        std::vector<float> initialize_parameters(std::uint64_t seed) const;
        void forward(const float* parameter_values, const float* patches, const float* times, const std::uint8_t* labels, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& workspace_layout);
        void loss(const float* target, float* loss, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& workspace_layout);
        void backward(const float* parameter_values, float* parameter_gradients, const float* patches, const float* times, const std::uint8_t* labels, float* patch_gradient, std::uint8_t* workspace, const FlowDiTWorkspaceLayout& workspace_layout);
    };
} // namespace physica::generative::flow_matching
