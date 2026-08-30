module;

#include <cublasLt.h>
#include <physica/cuda.h>

export module physica.neural.matmul;

import std;

export namespace physica::neural {
    enum class MatmulEpilogue : std::uint8_t {
        none,
        bias,
        gelu_bias,
        gelu_aux_bias,
        gelu_gradient,
        bias_gradient,
    };

    struct MatmulRequest final {
        const float* a;
        const float* b;
        float* output;
        std::uint32_t rows;
        std::uint32_t columns;
        std::uint32_t reduction;
        bool transpose_a{};
        bool transpose_b{};
        MatmulEpilogue epilogue{MatmulEpilogue::none};
        const float* bias{};
        float beta{};
        const float* auxiliary{};
    };

    struct MatmulRuntimeConfiguration final {
        std::size_t workspace_byte_count;
        std::size_t tuning_byte_count;
        std::size_t tuning_bias_byte_count;
    };

    struct MatmulRuntime final {

        struct PlanKey final {
            std::uint32_t rows;
            std::uint32_t columns;
            std::uint32_t reduction;
            bool transpose_a;
            bool transpose_b;
            MatmulEpilogue epilogue;

            bool operator==(const PlanKey&) const = default;
        };

        struct Plan final {
            PlanKey key;
            cublasLtMatmulDesc_t operation{};
            cublasLtMatrixLayout_t a_layout{};
            cublasLtMatrixLayout_t b_layout{};
            cublasLtMatrixLayout_t output_layout{};
            cublasLtMatmulAlgo_t algorithm{};
            std::vector<cublasLtMatmulAlgo_t> candidates;
            bool tuned{};

            Plan(cublasLtHandle_t handle, const PlanKey& key, const float* bias, const float* auxiliary, std::size_t workspace_byte_count, int multiprocessor_count);
            ~Plan() noexcept;

            Plan(const Plan&)            = delete;
            Plan& operator=(const Plan&) = delete;
            Plan(Plan&&)                 = delete;
            Plan& operator=(Plan&&)      = delete;
        };

        ::cuda::stream_ref stream;
        const MatmulRuntimeConfiguration configuration;
        cublasLtHandle_t handle{};
        ::cuda::device_buffer<std::uint8_t> workspace;
        ::cuda::device_buffer<std::uint8_t> tuning_output;
        ::cuda::device_buffer<std::uint8_t> tuning_auxiliary;
        ::cuda::device_buffer<std::uint8_t> tuning_bias;
        std::list<Plan> plans;

        MatmulRuntime(::cuda::stream_ref stream, MatmulRuntimeConfiguration configuration);
        ~MatmulRuntime() noexcept;

        MatmulRuntime(const MatmulRuntime&)            = delete;
        MatmulRuntime& operator=(const MatmulRuntime&) = delete;
        MatmulRuntime(MatmulRuntime&&)                 = delete;
        MatmulRuntime& operator=(MatmulRuntime&&)      = delete;

        void execute(const MatmulRequest& request);
    };
} // namespace physica::neural
