module;

#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <physica/cuda.h>

module physica.neural.matmul;

import std;

namespace physica::neural {
    namespace {
        cublasLtEpilogue_t epilogue_type(const MatmulEpilogue epilogue) {
            if (epilogue == MatmulEpilogue::bias) return CUBLASLT_EPILOGUE_BIAS;
            if (epilogue == MatmulEpilogue::gelu_bias) return CUBLASLT_EPILOGUE_GELU_BIAS;
            if (epilogue == MatmulEpilogue::gelu_aux_bias) return CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
            if (epilogue == MatmulEpilogue::gelu_gradient) return CUBLASLT_EPILOGUE_DGELU;
            if (epilogue == MatmulEpilogue::bias_gradient) return CUBLASLT_EPILOGUE_BGRADA;
            return CUBLASLT_EPILOGUE_DEFAULT;
        }

        bool uses_bias(const MatmulEpilogue epilogue) {
            return epilogue == MatmulEpilogue::bias || epilogue == MatmulEpilogue::gelu_bias || epilogue == MatmulEpilogue::gelu_aux_bias || epilogue == MatmulEpilogue::bias_gradient;
        }

        bool uses_auxiliary(const MatmulEpilogue epilogue) {
            return epilogue == MatmulEpilogue::gelu_aux_bias || epilogue == MatmulEpilogue::gelu_gradient;
        }
    } // namespace

    MatmulRuntime::Plan::Plan(const cublasLtHandle_t handle, const PlanKey& source_key, const float* const bias, const float* const auxiliary, const std::size_t workspace_byte_count, const int multiprocessor_count) : key{source_key} {
        const cublasOperation_t transpose_a   = key.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t transpose_b   = key.transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasLtEpilogue_t epilogue     = epilogue_type(key.epilogue);
        const std::uint64_t logical_a_rows    = key.transpose_a ? key.reduction : key.rows;
        const std::uint64_t logical_a_columns = key.transpose_a ? key.rows : key.reduction;
        const std::uint64_t logical_b_rows    = key.transpose_b ? key.columns : key.reduction;
        const std::uint64_t logical_b_columns = key.transpose_b ? key.reduction : key.columns;
        const std::uint64_t a_rows            = logical_b_columns;
        const std::uint64_t a_columns         = logical_b_rows;
        const std::uint64_t b_rows            = logical_a_columns;
        const std::uint64_t b_columns         = logical_a_rows;
        const std::int64_t a_leading          = static_cast<std::int64_t>(a_rows);
        const std::int64_t b_leading          = static_cast<std::int64_t>(b_rows);
        const std::int64_t output_leading     = static_cast<std::int64_t>(key.columns);
        cublasLtMatmulPreference_t preference{};
        std::array<cublasLtMatmulHeuristicResult_t, 32u> heuristics{};
        int returned_algorithm_count{};
        constexpr std::uint32_t reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE;

        try {
            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt operation descriptor: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a, sizeof(transpose_a)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt transpose A: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b, sizeof(transpose_b)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt transpose B: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &multiprocessor_count, sizeof(multiprocessor_count)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt SM count: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt epilogue: {}", cublasGetStatusString(status))};
            if (uses_bias(key.epilogue))
                if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt bias: {}", cublasGetStatusString(status))};
            if (uses_auxiliary(key.epilogue)) {
                const std::int64_t auxiliary_leading = static_cast<std::int64_t>(key.columns);
                if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &auxiliary, sizeof(auxiliary)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt auxiliary: {}", cublasGetStatusString(status))};
                if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &auxiliary_leading, sizeof(auxiliary_leading)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt auxiliary leading dimension: {}", cublasGetStatusString(status))};
            }
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_32F, a_rows, a_columns, a_leading); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt A layout: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_32F, b_rows, b_columns, b_leading); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt B layout: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&output_layout, CUDA_R_32F, key.columns, key.rows, output_leading); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt output layout: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt preference: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_byte_count, sizeof(workspace_byte_count)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt workspace preference: {}", cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_scheme, sizeof(reduction_scheme)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt reduction preference: {}", cublasGetStatusString(status))};
            const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(handle, operation, a_layout, b_layout, output_layout, output_layout, preference, static_cast<int>(heuristics.size()), heuristics.data(), &returned_algorithm_count);
            cublasLtMatmulPreferenceDestroy(preference);
            preference = nullptr;
            if (heuristic_status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt algorithm selection {}x{}x{} TA={} TB={}: {}", key.rows, key.columns, key.reduction, key.transpose_a, key.transpose_b, cublasGetStatusString(heuristic_status))};
            for (int index = 0; index < returned_algorithm_count; ++index) {
                std::int32_t split_k{};
                std::uint32_t algorithm_reduction_scheme{};
                if (heuristics[index].state != CUBLAS_STATUS_SUCCESS) continue;
                if (cublasLtMatmulAlgoConfigGetAttribute(&heuristics[index].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k), nullptr) != CUBLAS_STATUS_SUCCESS) continue;
                if (cublasLtMatmulAlgoConfigGetAttribute(&heuristics[index].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &algorithm_reduction_scheme, sizeof(algorithm_reduction_scheme), nullptr) != CUBLAS_STATUS_SUCCESS) continue;
                if (split_k != 1 || algorithm_reduction_scheme != CUBLASLT_REDUCTION_SCHEME_NONE) continue;
                candidates.push_back(heuristics[index].algo);
            }
            if (candidates.empty()) throw std::runtime_error{std::format("cuBLASLt has no deterministic algorithm for {}x{}x{} TA={} TB={}.", key.rows, key.columns, key.reduction, key.transpose_a, key.transpose_b)};
            algorithm = candidates.front();
        } catch (...) {
            if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
            if (output_layout != nullptr) cublasLtMatrixLayoutDestroy(output_layout);
            if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
            if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
            if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
            throw;
        }
    }

    MatmulRuntime::Plan::~Plan() noexcept {
        if (output_layout != nullptr) cublasLtMatrixLayoutDestroy(output_layout);
        if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
        if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
        if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    }

    MatmulRuntime::MatmulRuntime(const ::cuda::stream_ref source_stream, const MatmulRuntimeConfiguration source_configuration) : stream{source_stream}, configuration{source_configuration}, workspace{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.workspace_byte_count, ::cuda::no_init}, tuning_output{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.tuning_byte_count, ::cuda::no_init}, tuning_auxiliary{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.tuning_byte_count, ::cuda::no_init}, tuning_bias{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.tuning_bias_byte_count, ::cuda::no_init} {
        if (const cublasStatus_t status = cublasLtCreate(&handle); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt handle: {}", cublasGetStatusString(status))};
    }

    MatmulRuntime::~MatmulRuntime() noexcept {
        plans.clear();
        if (handle != nullptr) cublasLtDestroy(handle);
    }

    void MatmulRuntime::execute(const MatmulRequest& request) {
        const PlanKey key{
            .rows        = request.rows,
            .columns     = request.columns,
            .reduction   = request.reduction,
            .transpose_a = request.transpose_a,
            .transpose_b = request.transpose_b,
            .epilogue    = request.epilogue,
        };
        auto plan = std::ranges::find(plans, key, &Plan::key);
        if (plan == plans.end()) plan = plans.emplace(plans.end(), handle, key, request.bias, request.auxiliary, configuration.workspace_byte_count, stream.device().attribute(::cuda::device_attributes::multiprocessor_count));
        if (!plan->tuned) {
            const void* const tuning_bias_pointer      = tuning_bias.data();
            const void* const tuning_auxiliary_pointer = tuning_auxiliary.data();
            if (uses_bias(key.epilogue))
                if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &tuning_bias_pointer, sizeof(tuning_bias_pointer)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt tuning bias pointer: {}", cublasGetStatusString(status))};
            if (uses_auxiliary(key.epilogue))
                if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &tuning_auxiliary_pointer, sizeof(tuning_auxiliary_pointer)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt tuning auxiliary pointer: {}", cublasGetStatusString(status))};
            ::cuda::timed_event begin{stream.device()};
            ::cuda::timed_event end{stream.device()};
            constexpr float alpha = 1.0F;
            constexpr float beta  = 0.0F;
            float shortest        = std::numeric_limits<float>::infinity();
            for (const cublasLtMatmulAlgo_t& candidate : plan->candidates) {
                if (const cublasStatus_t status = cublasLtMatmul(handle, plan->operation, &alpha, request.b, plan->a_layout, request.a, plan->b_layout, &beta, tuning_output.data(), plan->output_layout, tuning_output.data(), plan->output_layout, &candidate, workspace.data(), workspace.size(), stream.get()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt tuning warmup: {}", cublasGetStatusString(status))};
                begin.record(stream);
                for (std::uint32_t repetition = 0u; repetition < 3u; ++repetition)
                    if (const cublasStatus_t status = cublasLtMatmul(handle, plan->operation, &alpha, request.b, plan->a_layout, request.a, plan->b_layout, &beta, tuning_output.data(), plan->output_layout, tuning_output.data(), plan->output_layout, &candidate, workspace.data(), workspace.size(), stream.get()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt tuning candidate: {}", cublasGetStatusString(status))};
                end.record(stream);
                end.sync();
                const float milliseconds = static_cast<float>((end - begin).count()) / 1'000'000.0F;
                if (milliseconds < shortest) {
                    shortest        = milliseconds;
                    plan->algorithm = candidate;
                }
            }
            plan->candidates.clear();
            plan->tuned = true;
        }
        if (uses_bias(key.epilogue))
            if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &request.bias, sizeof(request.bias)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt bias pointer: {}", cublasGetStatusString(status))};
        if (uses_auxiliary(key.epilogue))
            if (const cublasStatus_t status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &request.auxiliary, sizeof(request.auxiliary)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt auxiliary pointer: {}", cublasGetStatusString(status))};

        constexpr float alpha       = 1.0F;
        const cublasStatus_t status = cublasLtMatmul(handle, plan->operation, &alpha, request.b, plan->a_layout, request.a, plan->b_layout, &request.beta, request.output, plan->output_layout, request.output, plan->output_layout, &plan->algorithm, workspace.data(), workspace.size(), stream.get());
        if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt matmul: {}", cublasGetStatusString(status))};
    }
} // namespace physica::neural
