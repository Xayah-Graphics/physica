module;

#include <cublasLt.h>
#include <cudnn_frontend.h>
#include <physica/cuda.h>

export module physica.neural.inference_runtime;

import std;

export namespace physica::neural {
    enum class Scalar : std::uint8_t { f32, f16, bf16 };

    struct TensorView final {
        void* data{};
        int n{1};
        int h{1};
        int w{1};
        int c{1};
        Scalar scalar{Scalar::f16};

        std::size_t elements() const;
        std::size_t bytes() const;
        TensorView reshape(int n, int h, int w, int c, Scalar scalar = Scalar::f16) const;
    };

    struct Linear final {
        TensorView weight;
        TensorView bias;
    };
    struct Norm final {
        TensorView weight;
        TensorView bias;
        float epsilon{1.0e-5F};
    };
    struct Conv final {
        TensorView weight;
        TensorView bias;
        int stride{1};
        int padding{1};
    };

    struct MatmulPlan final {
        int rows;
        int columns;
        int reduction;
        Scalar scalar;
        bool bias;
        bool residual;
        cublasLtMatmulDesc_t operation{};
        cublasLtMatrixLayout_t a{};
        cublasLtMatrixLayout_t b{};
        cublasLtMatrixLayout_t output{};
        cublasLtMatmulAlgo_t algorithm{};
        std::size_t workspace_bytes{};

        MatmulPlan(int rows, int columns, int reduction, Scalar scalar, bool bias, bool residual);
        ~MatmulPlan();
        MatmulPlan(const MatmulPlan&)            = delete;
        MatmulPlan& operator=(const MatmulPlan&) = delete;
    };

    struct ConvPlan final {
        std::array<int, 10> key;
        cudnn_frontend::graph::Graph graph;

        explicit ConvPlan(const std::array<int, 10>& key);
    };

    struct InferenceRuntime final {
        ::cuda::stream_ref stream;
        std::size_t cache_hits{};
        std::size_t cache_misses{};
        double tuning_seconds{};

    private:
        cublasLtHandle_t blas{};
        cudnnHandle_t dnn{};
        ::cuda::device_buffer<std::byte> workspace;
        std::filesystem::path cache_directory;
        void* workspace_data{};
        std::size_t workspace_bytes{};
        std::size_t required_workspace{};
        std::list<MatmulPlan> matmuls;
        std::vector<std::array<int, 3>> geglus;
        std::list<ConvPlan> convolutions;
        struct AttentionPlan final {
            std::array<int, 10> key;
            cudnn_frontend::graph::Graph graph;
            ::cuda::device_buffer<std::int32_t> query_lengths;
            AttentionPlan(::cuda::stream_ref stream, const std::array<int, 10>& shape);
        };
        std::list<AttentionPlan> attentions;

    public:
        InferenceRuntime(::cuda::stream_ref stream, const std::filesystem::path& cache_directory);
        ~InferenceRuntime();
        InferenceRuntime(const InferenceRuntime&)            = delete;
        InferenceRuntime& operator=(const InferenceRuntime&) = delete;
        void begin_preparation();
        ::cuda::device_buffer<std::byte> finish_preparation();
        void linear(TensorView output, TensorView input, const Linear& layer, TensorView residual = {});
        void geglu(TensorView output, TensorView input, const Linear& layer);
        void convolution(TensorView output, TensorView input, const Conv& layer, TensorView residual = {});
        void layer_norm(TensorView output, TensorView input, const Norm& layer);
        void group_norm(TensorView output, TensorView input, const Norm& layer, float* statistics, bool silu, bool prepared = false, TensorView time = {}, const int* step = nullptr);
        void attention(TensorView output, TensorView query, TensorView key, TensorView value, int heads, int query_stride, int key_stride, bool causal = false, const std::int32_t* positions = nullptr, const std::int32_t* lengths = nullptr);
    };

    void check(cudaError_t status);
    void check(cublasStatus_t status);
    void check(cudnnStatus_t status);
    void check(cudnn_frontend::error_t status);
} // namespace physica::neural
