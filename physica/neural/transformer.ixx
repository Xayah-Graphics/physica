module;

#include <physica/cuda.h>

export module physica.neural.transformer;

import std;
import physica.neural.matmul;

export namespace physica::neural {
    struct TransformerConfiguration final {
        std::uint32_t sequence;
        std::uint32_t width;
        std::uint32_t block_count;
        std::uint32_t head_count;
        std::uint32_t mlp_width;

        bool operator==(const TransformerConfiguration&) const = default;
    };

    struct TransformerBlockParameterLayout final {
        std::size_t modulation_weight;
        std::size_t modulation_bias;
        std::size_t qkv_weight;
        std::size_t qkv_bias;
        std::size_t attention_output_weight;
        std::size_t attention_output_bias;
        std::size_t mlp_input_weight;
        std::size_t mlp_input_bias;
        std::size_t mlp_output_weight;
        std::size_t mlp_output_bias;
    };

    struct TransformerParameterLayout final {
        TransformerConfiguration configuration;
        std::vector<TransformerBlockParameterLayout> blocks;
        std::size_t parameter_count;

        explicit TransformerParameterLayout(const TransformerConfiguration& configuration);
    };

    struct TransformerBlockWorkspaceLayout final {
        std::size_t modulation;
        std::size_t attention_normalized;
        std::size_t attention_means;
        std::size_t attention_inverse_standard_deviations;
        std::size_t qkv;
        std::size_t attention;
        std::size_t attention_log_sum_exp;
        std::size_t attention_projected;
        std::size_t after_attention;
        std::size_t mlp_normalized;
        std::size_t mlp_means;
        std::size_t mlp_inverse_standard_deviations;
        std::size_t mlp_preactivation;
        std::size_t mlp_hidden;
        std::size_t mlp_projected;
        std::size_t output;
    };

    struct TransformerWorkspaceLayout final {
        TransformerConfiguration configuration;
        std::uint32_t batch;
        std::vector<TransformerBlockWorkspaceLayout> blocks;
        std::size_t gradient_a;
        std::size_t gradient_b;
        std::size_t branch_gradient;
        std::size_t normalized_gradient;
        std::size_t modulation_gradient;
        std::size_t qkv_gradient;
        std::size_t attention_delta;
        std::size_t mlp_preactivation_gradient;
        std::size_t byte_count;

        TransformerWorkspaceLayout(const TransformerConfiguration& configuration, std::uint32_t batch);
    };

    template <TransformerConfiguration Configuration, std::uint32_t Batch>
    struct StaticTransformerWorkspace final {
        static consteval std::size_t align(const std::size_t value) {
            return (value + 255uz) & ~255uz;
        }

        inline static constexpr std::size_t token_count        = static_cast<std::size_t>(Batch) * Configuration.sequence;
        inline static constexpr std::size_t output_byte_count  = align(token_count * Configuration.width * sizeof(float));
        inline static constexpr std::size_t block_byte_count   = align(static_cast<std::size_t>(Batch) * 6uz * Configuration.width * sizeof(float)) + align(token_count * Configuration.width * sizeof(float)) + align(token_count * sizeof(float)) * 2uz + align(token_count * 3uz * Configuration.width * sizeof(float)) + align(token_count * Configuration.width * sizeof(float)) + align(static_cast<std::size_t>(Batch) * Configuration.head_count * Configuration.sequence * sizeof(float)) + align(token_count * Configuration.width * sizeof(float)) * 3uz + align(token_count * sizeof(float)) * 2uz + align(token_count * Configuration.mlp_width * sizeof(float)) * 2uz + align(token_count * Configuration.width * sizeof(float)) * 2uz;
        inline static constexpr std::size_t scratch_byte_count = align(token_count * Configuration.width * sizeof(float)) * 4uz + align(static_cast<std::size_t>(Batch) * 6uz * Configuration.width * sizeof(float)) + align(token_count * 3uz * Configuration.width * sizeof(float)) + align(static_cast<std::size_t>(Batch) * Configuration.head_count * Configuration.sequence * sizeof(float)) + align(token_count * Configuration.mlp_width * sizeof(float));
        inline static constexpr std::size_t byte_count         = block_byte_count * Configuration.block_count - output_byte_count + scratch_byte_count;
    };

    struct Transformer final {
        MatmulRuntime& matmul;
        TransformerParameterLayout parameters;

        Transformer(MatmulRuntime& matmul, const TransformerConfiguration& configuration);

        std::vector<float> initialize_parameters(std::uint64_t seed) const;
        void forward(const float* parameter_values, const float* tokens, const float* condition, float* output, std::uint8_t* workspace, const TransformerWorkspaceLayout& workspace_layout);
        void backward(const float* parameter_values, float* parameter_gradients, const float* tokens, const float* condition, const float* output_gradient, float* token_gradient, float* condition_gradient, std::uint8_t* workspace, const TransformerWorkspaceLayout& workspace_layout);
    };
} // namespace physica::neural
