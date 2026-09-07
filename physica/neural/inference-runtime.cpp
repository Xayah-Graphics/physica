module;

#include "inference-kernels.h"
#include <cublasLt.h>
#include <cudnn_frontend.h>
#include <cutlass/version.h>
#include <physica/cuda.h>

#include <nlohmann/json.hpp>

module physica.neural.inference_runtime;

import std;

namespace physica::neural {
    nlohmann::json read_plan(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary};
        file.exceptions(std::ios::failbit | std::ios::badbit);
        return nlohmann::json::from_ubjson(std::vector<std::uint8_t>{std::istreambuf_iterator<char>{file}, {}});
    }

    void write_plan(const std::filesystem::path& path, const nlohmann::json& value) {
        const auto bytes = nlohmann::json::to_ubjson(value);
        std::ofstream file{path, std::ios::binary};
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

    std::size_t TensorView::elements() const {
        return static_cast<std::size_t>(n) * h * w * c;
    }
    std::size_t TensorView::bytes() const {
        return elements() * (scalar == Scalar::f32 ? 4uz : 2uz);
    }

    TensorView TensorView::reshape(const int n, const int h, const int w, const int c, const Scalar scalar) const {
        return {data, n, h, w, c, scalar};
    }

    MatmulPlan::MatmulPlan(const int m, const int n, const int k, const Scalar type, const bool has_bias, const bool has_residual) : rows{m}, columns{n}, reduction{k}, scalar{type}, bias{has_bias}, residual{has_residual} {
        const cudaDataType_t dtype = type == Scalar::f32 ? CUDA_R_32F : type == Scalar::f16 ? CUDA_R_16F : CUDA_R_16BF;
        check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t transpose = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        const cublasLtEpilogue_t epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        check(cublasLtMatrixLayoutCreate(&a, dtype, k, n, k));
        check(cublasLtMatrixLayoutCreate(&b, dtype, k, m, k));
        check(cublasLtMatrixLayoutCreate(&output, dtype, n, m, n));
    }

    MatmulPlan::~MatmulPlan() {
        cublasLtMatmulDescDestroy(operation);
        cublasLtMatrixLayoutDestroy(a);
        cublasLtMatrixLayoutDestroy(b);
        cublasLtMatrixLayoutDestroy(output);
    }

    ConvPlan::ConvPlan(const std::array<int, 10>& shape) : key{shape} {
        const auto [n, h, w, input_width, output_width, kernel, stride, padding, scalar, residual] = shape;
        const auto dtype                                                                           = scalar == 1 ? cudnn_frontend::DataType_t::HALF : cudnn_frontend::DataType_t::BFLOAT16;
        graph.set_io_data_type(dtype).set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT).set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
        auto input  = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("input").set_uid(1).set_dim({n, input_width, h, w}).set_stride({static_cast<std::int64_t>(h) * w * input_width, 1, static_cast<std::int64_t>(w) * input_width, input_width}));
        auto weight = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("weight").set_uid(2).set_dim({output_width, input_width, kernel, kernel}).set_stride({static_cast<std::int64_t>(kernel) * kernel * input_width, 1, static_cast<std::int64_t>(kernel) * input_width, input_width}));
        auto bias   = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("bias").set_uid(3).set_dim({1, output_width, 1, 1}).set_stride({output_width, 1, output_width, output_width}));
        auto hidden = graph.conv_fprop(input, weight, cudnn_frontend::graph::Conv_fprop_attributes{}.set_padding({padding, padding}).set_stride({stride, stride}).set_dilation({1, 1}));
        auto result = graph.pointwise(hidden, bias, cudnn_frontend::graph::Pointwise_attributes{}.set_mode(cudnn_frontend::PointwiseMode_t::ADD));
        if (residual) {
            auto skip = graph.tensor_like(input, "residual");
            skip->set_uid(5).set_dim({n, output_width, (h + 2 * padding - kernel) / stride + 1, (w + 2 * padding - kernel) / stride + 1}).set_stride({static_cast<std::int64_t>((h + 2 * padding - kernel) / stride + 1) * ((w + 2 * padding - kernel) / stride + 1) * output_width, 1, static_cast<std::int64_t>((w + 2 * padding - kernel) / stride + 1) * output_width, output_width});
            result = graph.pointwise(result, skip, cudnn_frontend::graph::Pointwise_attributes{}.set_mode(cudnn_frontend::PointwiseMode_t::ADD));
        }
        const int output_height  = (h + 2 * padding - kernel) / stride + 1;
        const int output_columns = (w + 2 * padding - kernel) / stride + 1;
        result->set_output(true).set_uid(4).set_dim({n, output_width, output_height, output_columns}).set_stride({static_cast<std::int64_t>(output_height) * output_columns * output_width, 1, static_cast<std::int64_t>(output_columns) * output_width, output_width});
        check(graph.validate());
    }

    InferenceRuntime::AttentionPlan::AttentionPlan(const ::cuda::stream_ref stream, const std::array<int, 10>& shape) : key{shape}, query_lengths{stream, ::cuda::device_default_memory_pool(stream.device()), shape[9] ? std::size_t(shape[0]) : 0uz, ::cuda::no_init} {
        const auto [batch, queries, keys, heads, dimension, query_stride, key_stride, scalar, causal, ragged] = shape;
        if (ragged) {
            const std::vector<std::int32_t> lengths(batch, queries);
            ::cuda::copy_bytes(stream, ::cuda::std::span<const std::int32_t>{lengths.data(), lengths.size()}, query_lengths);
            stream.sync();
        }
        graph.set_io_data_type(scalar == 1 ? cudnn_frontend::DataType_t::HALF : cudnn_frontend::DataType_t::BFLOAT16).set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT).set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
        auto q          = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("q").set_uid(1).set_dim({batch, heads, queries, dimension}).set_stride({static_cast<std::int64_t>(queries) * query_stride, dimension, query_stride, 1}));
        auto k          = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("k").set_uid(2).set_dim({batch, heads, keys, dimension}).set_stride({static_cast<std::int64_t>(keys) * key_stride, dimension, key_stride, 1}));
        auto v          = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_name("v").set_uid(3).set_dim({batch, heads, keys, dimension}).set_stride({static_cast<std::int64_t>(keys) * key_stride, dimension, key_stride, 1}));
        auto attributes = cudnn_frontend::graph::SDPA_attributes{}.set_generate_stats(false).set_attn_scale(1.0F / std::sqrt(static_cast<float>(dimension))).set_causal_mask(causal != 0);
        if (ragged) {
            auto lengths       = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_uid(5).set_data_type(cudnn_frontend::DataType_t::INT32).set_dim({batch, 1, 1, 1}).set_stride({1, 1, 1, 1}));
            auto query_lengths = graph.tensor(cudnn_frontend::graph::Tensor_attributes{}.set_uid(6).set_data_type(cudnn_frontend::DataType_t::INT32).set_dim({batch, 1, 1, 1}).set_stride({1, 1, 1, 1}));
            attributes.set_padding_mask(true).set_seq_len_kv(lengths).set_seq_len_q(query_lengths);
        }
        auto [o, stats] = graph.sdpa(q, k, v, attributes);
        o->set_output(true).set_uid(4).set_dim({batch, heads, queries, dimension}).set_stride({static_cast<std::int64_t>(queries) * heads * dimension, dimension, heads * dimension, 1});
        check(graph.validate());
    }

    InferenceRuntime::InferenceRuntime(const ::cuda::stream_ref execution, const std::filesystem::path& directory) : stream{execution}, workspace{stream, ::cuda::device_default_memory_pool(stream.device())} {
        check(cublasLtCreate(&blas));
        check(cudnnCreate(&dnn));
        check(cudnnSetStream(dnn, stream.get()));
        cudaDeviceProp device;
        check(cudaGetDeviceProperties(&device, stream.device().get()));
        int driver;
        check(cudaDriverGetVersion(&driver));
        cache_directory = directory / std::format("v3-sm120a-{}-{}-{}-{}-{}-{}-{}-{}", CUDART_VERSION, driver, cudnnGetVersion(), CUDNN_FRONTEND_VERSION, cublasLtGetVersion(), CUTLASS_VERSION, _MSC_FULL_VER, device.multiProcessorCount);
        std::filesystem::create_directories(cache_directory);
        begin_preparation();
    }
    InferenceRuntime::~InferenceRuntime() {
        stream.sync();
        convolutions.clear();
        matmuls.clear();
        attentions.clear();
        cudnnDestroy(dnn);
        cublasLtDestroy(blas);
    }

    void InferenceRuntime::begin_preparation() {
        workspace       = ::cuda::device_buffer<std::byte>{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz << 30, ::cuda::no_init};
        workspace_data  = workspace.data();
        workspace_bytes = workspace.size();
    }

    ::cuda::device_buffer<std::byte> InferenceRuntime::finish_preparation() {
        stream.sync();
        workspace       = ::cuda::device_buffer<std::byte>{stream, ::cuda::device_default_memory_pool(stream.device()), required_workspace, ::cuda::no_init};
        workspace_data  = workspace.data();
        workspace_bytes = workspace.size();
        return std::move(workspace);
    }

    void InferenceRuntime::linear(const TensorView result, const TensorView input, const Linear& layer, const TensorView residual) {
        const int rows     = input.n * input.h * input.w;
        const int columns  = layer.weight.w;
        auto plan          = std::ranges::find_if(matmuls, [&](const MatmulPlan& value) { return value.rows == rows && value.columns == columns && value.reduction == input.c && value.scalar == input.scalar && value.bias == (layer.bias.data != nullptr) && value.residual == (residual.data != nullptr); });
        const bool prepare = plan == matmuls.end();
        if (prepare) plan = matmuls.emplace(matmuls.end(), rows, columns, input.c, input.scalar, layer.bias.data != nullptr, residual.data != nullptr);
        if (layer.bias.data) check(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &layer.bias.data, sizeof(layer.bias.data)));
        constexpr float alpha = 1.0F;
        const float beta      = residual.data ? 1.0F : 0.0F;
        const void* previous  = residual.data ? residual.data : result.data;
        if (prepare) {
            const auto file = cache_directory / std::format("gemm-{}-{}-{}-{}-{}-{}.bin", rows, columns, input.c, int(input.scalar), layer.bias.data != nullptr, residual.data != nullptr);
            if (std::filesystem::exists(file)) {
                const auto cached = read_plan(file);
                const auto bytes  = cached.at("algorithm").get<std::array<std::uint8_t, sizeof(cublasLtMatmulAlgo_t)>>();
                std::memcpy(&plan->algorithm, bytes.data(), bytes.size());
                plan->workspace_bytes = cached.at("workspace").get<std::size_t>();
                ++cache_hits;
            } else {
                const auto started = std::chrono::steady_clock::now();
                ::cuda::device_buffer<std::byte> temporary{stream, ::cuda::device_default_memory_pool(stream.device()), result.bytes(), ::cuda::no_init};
                cudaEvent_t start, stop;
                check(cudaEventCreate(&start));
                check(cudaEventCreate(&stop));
                cublasLtMatmulPreference_t preference{};
                check(cublasLtMatmulPreferenceCreate(&preference));
                const std::size_t workspace_bytes    = 1uz << 30;
                const std::uint32_t reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
                check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
                check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_scheme, sizeof(reduction_scheme)));
                std::array<cublasLtMatmulHeuristicResult_t, 32> candidates;
                int count{};
                check(cublasLtMatmulAlgoGetHeuristic(blas, plan->operation, plan->a, plan->b, plan->output, plan->output, preference, static_cast<int>(candidates.size()), candidates.data(), &count));
                cublasLtMatmulPreferenceDestroy(preference);
                if (count == 0) throw std::runtime_error{"Inference GEMM has no supported algorithm"};

                float best = std::numeric_limits<float>::infinity();
                for (int index = 0; index < count; ++index) {
                    if (candidates[index].state != CUBLAS_STATUS_SUCCESS) continue;
                    const auto& candidate = candidates[index].algo;
                    cublasLtMatmulHeuristicResult_t checked{};
                    if (cublasLtMatmulAlgoCheck(blas, plan->operation, plan->a, plan->b, plan->output, plan->output, &candidate, &checked) != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS) continue;
                    check(cublasLtMatmul(blas, plan->operation, &alpha, layer.weight.data, plan->a, input.data, plan->b, &beta, previous, plan->output, temporary.data(), plan->output, &candidate, workspace_data, workspace_bytes, stream.get()));
                    check(cudaEventRecord(start, stream.get()));
                    for (int i = 0; i < 3; ++i) check(cublasLtMatmul(blas, plan->operation, &alpha, layer.weight.data, plan->a, input.data, plan->b, &beta, previous, plan->output, temporary.data(), plan->output, &candidate, workspace_data, workspace_bytes, stream.get()));
                    check(cudaEventRecord(stop, stream.get()));
                    check(cudaEventSynchronize(stop));
                    float elapsed;
                    check(cudaEventElapsedTime(&elapsed, start, stop));
                    if (elapsed < best) {
                        best                  = elapsed;
                        plan->algorithm       = candidate;
                        plan->workspace_bytes = checked.workspaceSize;
                    }
                }
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                if (!std::isfinite(best)) throw std::runtime_error{"Inference GEMM candidate measurement failed"};
                std::array<std::uint8_t, sizeof(cublasLtMatmulAlgo_t)> bytes;
                std::memcpy(bytes.data(), &plan->algorithm, bytes.size());
                write_plan(file, {{"algorithm", bytes}, {"workspace", plan->workspace_bytes}});
                ++cache_misses;
                tuning_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            }
            required_workspace = std::max(required_workspace, plan->workspace_bytes);
        }
        check(cublasLtMatmul(blas, plan->operation, &alpha, layer.weight.data, plan->a, input.data, plan->b, &beta, previous, plan->output, result.data, plan->output, &plan->algorithm, workspace_data, workspace_bytes, stream.get()));
    }

    void InferenceRuntime::geglu(const TensorView output, const TensorView input, const Linear& layer) {
        const int rows  = input.n * input.h * input.w;
        const int width = input.c;
        auto plan       = std::ranges::find_if(geglus, [&](const auto& value) { return value[0] == rows && value[1] == width; });
        if (plan == geglus.end()) {
            int configuration{};
            const auto file = cache_directory / std::format("geglu-{}-{}.bin", rows, width);
            if (std::filesystem::exists(file)) {
                configuration = read_plan(file).get<int>();
                ++cache_hits;
            } else {
                const auto started = std::chrono::steady_clock::now();
                cudaEvent_t start, stop;
                check(cudaEventCreate(&start));
                check(cudaEventCreate(&stop));
                float best = std::numeric_limits<float>::infinity();
                for (int candidate = 0; candidate < 4; ++candidate) {
                    kernels::linear_geglu(stream, output.data, input.data, layer.weight.data, layer.bias.data, rows, width, candidate);
                    check(cudaEventRecord(start, stream.get()));
                    for (int i = 0; i < 3; ++i) kernels::linear_geglu(stream, output.data, input.data, layer.weight.data, layer.bias.data, rows, width, candidate);
                    check(cudaEventRecord(stop, stream.get()));
                    check(cudaEventSynchronize(stop));
                    float elapsed;
                    check(cudaEventElapsedTime(&elapsed, start, stop));
                    if (elapsed < best) {
                        best          = elapsed;
                        configuration = candidate;
                    }
                }
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                write_plan(file, configuration);
                ++cache_misses;
                tuning_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            }
            plan = geglus.emplace(geglus.end(), std::array{rows, width, configuration});
        }
        kernels::linear_geglu(stream, output.data, input.data, layer.weight.data, layer.bias.data, rows, width, (*plan)[2]);
    }

    void InferenceRuntime::convolution(const TensorView output, const TensorView input, const Conv& layer, const TensorView residual) {
        const std::array<int, 10> key{input.n, input.h, input.w, input.c, layer.weight.n, layer.weight.h, layer.stride, layer.padding, int(input.scalar), residual.data != nullptr};
        auto plan = std::ranges::find_if(convolutions, [&](const ConvPlan& value) { return value.key == key; });
        std::unordered_map<std::int64_t, void*> tensors{{1, input.data}, {2, layer.weight.data}, {3, layer.bias.data}, {4, output.data}};
        if (residual.data) tensors.emplace(5, residual.data);
        if (plan == convolutions.end()) {
            plan            = convolutions.emplace(convolutions.end(), key);
            const auto file = cache_directory / ("conv-" + nlohmann::json(key).dump() + ".bin");
            if (std::filesystem::exists(file)) {
                check(plan->graph.deserialize(dnn, read_plan(file).get<std::vector<std::uint8_t>>(), false, false));
                ++cache_hits;
            } else {
                const auto started = std::chrono::steady_clock::now();
                check(plan->graph.build_operation_graph(dnn));
                check(plan->graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}));
                plan->graph.deselect_numeric_notes({cudnn_frontend::NumericalNote_t::NONDETERMINISTIC, cudnn_frontend::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
                plan->graph.deselect_workspace_greater_than(1uz << 30);
                check(plan->graph.check_support(dnn));
                check(plan->graph.build_plans(cudnn_frontend::BuildPlanPolicy_t::ALL));
                ::cuda::device_buffer<std::byte> temporary{stream, ::cuda::device_default_memory_pool(stream.device()), output.bytes(), ::cuda::no_init};
                tensors[4] = temporary.data();
                check(plan->graph.autotune(dnn, tensors, workspace_data));
                tensors[4] = output.data;
                std::vector<std::uint8_t> bytes;
                check(plan->graph.serialize(bytes, false));
                write_plan(file, bytes);
                ++cache_misses;
                tuning_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            }
            required_workspace = std::max(required_workspace, static_cast<std::size_t>(plan->graph.get_workspace_size()));
        }
        check(plan->graph.execute(dnn, tensors, workspace_data));
    }

    void InferenceRuntime::layer_norm(const TensorView output, const TensorView input, const Norm& layer) {
        kernels::layer_norm(stream, output.data, input.data, layer.weight.data, layer.bias.data, input.n * input.h * input.w, input.c, layer.epsilon, static_cast<int>(input.scalar));
    }
    void InferenceRuntime::group_norm(const TensorView output, const TensorView input, const Norm& layer, float* statistics, const bool silu, const bool prepared, const TensorView time, const int* step) {
        kernels::group_norm(stream, output.data, input.data, layer.weight.data, layer.bias.data, statistics, input.n, input.h * input.w, input.c, layer.epsilon, static_cast<int>(input.scalar), silu, prepared, time.data, step);
    }

    void InferenceRuntime::attention(const TensorView output, const TensorView query, const TensorView key, const TensorView value, const int heads, const int query_stride, const int key_stride, const bool causal, const std::int32_t* positions, const std::int32_t* lengths) {
        const int dimension = output.c / heads;
        const int queries   = query.h * query.w;
        const int keys      = key.h * key.w;
        if (dimension == 512 || positions) {
            kernels::attention(stream, output.data, query.data, key.data, value.data, query.n, queries, keys, heads, dimension, query_stride, key_stride, int(query.scalar), causal, positions);
            return;
        }
        const std::array<int, 10> shape{query.n, queries, keys, heads, dimension, query_stride, key_stride, int(query.scalar), causal, lengths != nullptr};
        auto plan = std::ranges::find_if(attentions, [&](const AttentionPlan& item) { return item.key == shape; });
        std::unordered_map<std::int64_t, void*> tensors{{1, query.data}, {2, key.data}, {3, value.data}, {4, output.data}};
        if (lengths) tensors.emplace(5, const_cast<std::int32_t*>(lengths));
        if (plan == attentions.end()) {
            plan = attentions.emplace(attentions.end(), stream, shape);
            if (lengths) tensors.emplace(6, plan->query_lengths.data());
            const auto file = cache_directory / ("attention-" + nlohmann::json(shape).dump() + ".bin");
            if (std::filesystem::exists(file)) {
                check(plan->graph.deserialize(dnn, read_plan(file).get<std::vector<std::uint8_t>>(), false, false));
                ++cache_hits;
            } else {
                const auto started = std::chrono::steady_clock::now();
                check(plan->graph.build_operation_graph(dnn));
                check(plan->graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}));
                plan->graph.deselect_numeric_notes({cudnn_frontend::NumericalNote_t::NONDETERMINISTIC, cudnn_frontend::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
                plan->graph.deselect_workspace_greater_than(1uz << 30);
                check(plan->graph.check_support(dnn));
                check(plan->graph.build_plans(cudnn_frontend::BuildPlanPolicy_t::ALL));
                check(plan->graph.autotune(dnn, tensors, workspace_data));
                std::vector<std::uint8_t> bytes;
                check(plan->graph.serialize(bytes, false));
                write_plan(file, bytes);
                ++cache_misses;
                tuning_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            }
            required_workspace = std::max(required_workspace, static_cast<std::size_t>(plan->graph.get_workspace_size()));
        }
        if (lengths) tensors[6] = plan->query_lengths.data();
        check(plan->graph.execute(dnn, tensors, workspace_data));
    }

    void check(const cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error{cudaGetErrorString(status)};
    }
    void check(const cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt status {}", static_cast<int>(status))};
    }
    void check(const cudnnStatus_t status) {
        if (status != CUDNN_STATUS_SUCCESS) throw std::runtime_error{cudnnGetErrorString(status)};
    }
    void check(cudnn_frontend::error_t status) {
        if (status.is_bad()) throw std::runtime_error{status.get_message()};
    }
} // namespace physica::neural
