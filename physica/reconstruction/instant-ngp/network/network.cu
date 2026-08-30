#include "kernels.h"
#include <cublasLt.h>
#include <cuda/algorithm>
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/random>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <mma.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace physica::reconstruction::instant_ngp {
    inline constexpr kernels::NetworkKernelShape network_cuda_shape{
        .grid_level_count           = 8u,
        .grid_features_per_level    = 4u,
        .grid_base_resolution       = 16u,
        .grid_resolution_scale      = 2u,
        .grid_log2_hashmap_size     = 19u,
        .direction_degree           = 4u,
        .mlp_width                  = 64u,
        .density_hidden_layer_count = 1u,
        .color_hidden_layer_count   = 2u,
        .density_output_width       = 16u,
        .network_output_width       = 16u,
        .training_batch_size        = 1u << 18u,
    };

    struct NetworkGridOffsets final {
        std::uint32_t values[network_cuda_shape.grid_level_count + 1u]{};
    };

    template <kernels::NetworkKernelShape Shape>
    struct NetworkLayout final {
        inline static constexpr std::uint32_t grid_n_levels           = Shape.grid_level_count;
        inline static constexpr std::uint32_t grid_features_per_level = Shape.grid_features_per_level;
        inline static constexpr std::uint32_t grid_base_resolution    = Shape.grid_base_resolution;
        inline static constexpr std::uint32_t grid_resolution_scale   = Shape.grid_resolution_scale;
        inline static constexpr std::uint32_t grid_log2_hashmap_size  = Shape.grid_log2_hashmap_size;
        inline static constexpr std::uint32_t grid_output_width       = Shape.grid_level_count * Shape.grid_features_per_level;
        inline static constexpr std::uint32_t direction_output_width  = Shape.direction_degree * Shape.direction_degree;
        inline static constexpr std::uint32_t mlp_width               = Shape.mlp_width;
        inline static constexpr std::uint32_t density_hidden_layers   = Shape.density_hidden_layer_count;
        inline static constexpr std::uint32_t rgb_hidden_layers       = Shape.color_hidden_layer_count;
        inline static constexpr std::uint32_t density_output_width    = Shape.density_output_width;
        inline static constexpr std::uint32_t rgb_input_width         = Shape.density_output_width + direction_output_width;
        inline static constexpr std::uint32_t network_output_width    = Shape.network_output_width;
        inline static constexpr std::uint32_t network_batch_size      = Shape.training_batch_size;
        inline static constexpr std::uint32_t mlp_width_blocks        = Shape.mlp_width / 16u;

        struct ParameterLayout final {
            NetworkGridOffsets grid_offsets{};
            std::uint32_t density_param_offset         = 0u;
            std::uint32_t density_input_weight_offset  = 0u;
            std::uint32_t density_output_weight_offset = 0u;
            std::uint32_t rgb_param_offset             = 0u;
            std::uint32_t rgb_input_weight_offset      = 0u;
            std::uint32_t rgb_hidden_weight_offset     = 0u;
            std::uint32_t rgb_output_weight_offset     = 0u;
            std::uint32_t mlp_param_count              = 0u;
            std::uint32_t grid_param_offset            = 0u;
            std::uint32_t grid_param_count             = 0u;
            std::uint32_t total_param_count            = 0u;
        };

        static consteval ParameterLayout make_parameter_layout() {
            ParameterLayout layout{};
            std::uint32_t grid_cursor                  = 0u;
            std::uint32_t resolution                   = Shape.grid_base_resolution;
            constexpr std::uint64_t grid_hashmap_size  = 1ull << Shape.grid_log2_hashmap_size;
            constexpr std::uint64_t grid_max_positions = std::numeric_limits<std::uint32_t>::max() / 2ull;

            for (std::uint32_t level = 0u; level < Shape.grid_level_count; ++level) {
                const std::uint64_t dense         = static_cast<std::uint64_t>(resolution) * resolution * resolution;
                std::uint64_t positions           = ::cuda::std::min(dense, grid_max_positions);
                positions                         = ::cuda::round_up(positions, 8ull);
                positions                         = ::cuda::std::min(positions, grid_hashmap_size);
                layout.grid_offsets.values[level] = grid_cursor;
                grid_cursor += static_cast<std::uint32_t>(positions);
                resolution *= Shape.grid_resolution_scale;
            }
            layout.grid_offsets.values[Shape.grid_level_count] = grid_cursor;

            std::uint32_t cursor               = 0u;
            layout.density_param_offset        = cursor;
            layout.density_input_weight_offset = cursor;
            cursor += Shape.mlp_width * grid_output_width;
            cursor += (Shape.density_hidden_layer_count - 1u) * Shape.mlp_width * Shape.mlp_width;
            layout.density_output_weight_offset = cursor;
            cursor += Shape.density_output_width * Shape.mlp_width;
            layout.rgb_param_offset        = cursor;
            layout.rgb_input_weight_offset = cursor;
            cursor += Shape.mlp_width * rgb_input_width;
            layout.rgb_hidden_weight_offset = cursor;
            cursor += (Shape.color_hidden_layer_count - 1u) * Shape.mlp_width * Shape.mlp_width;
            layout.rgb_output_weight_offset = cursor;
            cursor += Shape.network_output_width * Shape.mlp_width;
            layout.mlp_param_count   = cursor;
            layout.grid_param_offset = cursor;
            layout.grid_param_count  = layout.grid_offsets.values[Shape.grid_level_count] * Shape.grid_features_per_level;
            layout.total_param_count = cursor + layout.grid_param_count;
            return layout;
        }

        inline static constexpr ParameterLayout network_parameter_layout = make_parameter_layout();
    };
} // namespace physica::reconstruction::instant_ngp

namespace physica::reconstruction::instant_ngp::kernels {
    namespace {
        inline constexpr std::uint32_t network_random_domain = 0u;

        enum class NetworkRandomSequence : std::uint32_t {
            grid_parameters,
            mlp_parameters,
        };

        inline __device__ std::uint32_t grid_index(const std::uint32_t hashmap_size, const std::uint32_t resolution, const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) {
            const std::uint64_t dense_size = static_cast<std::uint64_t>(resolution) * resolution * resolution;
            if (dense_size <= hashmap_size) return static_cast<std::uint32_t>(x + static_cast<std::uint64_t>(y) * resolution + static_cast<std::uint64_t>(z) * resolution * resolution);
            return (x ^ (y * 2654435761u) ^ (z * 805459861u)) % hashmap_size;
        }

        inline __device__ void grid_position_fraction(const float input, float& pos, std::uint32_t& pos_grid, const float scale) {
            pos                 = ::cuda::std::fma(scale, input, 0.5f);
            const float floored = ::cuda::std::floor(pos);
            pos_grid            = static_cast<std::uint32_t>(static_cast<int>(floored));
            pos -= floored;
        }

        __global__ void encode_grid_forward_kernel(const std::uint32_t sample_count, const NetworkGridOffsets grid_offsets, const float* __restrict__ sample_coords, const __half* __restrict__ grid, __half* __restrict__ encoded_positions) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= sample_count) return;

            const std::uint32_t level             = blockIdx.y;
            const std::uint32_t level_offset      = grid_offsets.values[level];
            const std::uint32_t next_level_offset = grid_offsets.values[level + 1u];
            grid += level_offset * NetworkLayout<network_cuda_shape>::grid_features_per_level;
            const std::uint32_t hashmap_size = next_level_offset - level_offset;
            std::uint32_t resolution         = NetworkLayout<network_cuda_shape>::grid_base_resolution;
            for (std::uint32_t index = 0u; index < level; ++index) resolution *= NetworkLayout<network_cuda_shape>::grid_resolution_scale;
            const float scale   = static_cast<float>(resolution) - 1.0f;
            const float* sample = sample_coords + static_cast<std::uint64_t>(i) * 7u;

            float pos_x          = 0.0f;
            float pos_y          = 0.0f;
            float pos_z          = 0.0f;
            std::uint32_t grid_x = 0u;
            std::uint32_t grid_y = 0u;
            std::uint32_t grid_z = 0u;
            grid_position_fraction(sample[0], pos_x, grid_x, scale);
            grid_position_fraction(sample[1], pos_y, grid_y, scale);
            grid_position_fraction(sample[2], pos_z, grid_z, scale);

            __half result0 = 0.0f;
            __half result1 = 0.0f;
            __half result2 = 0.0f;
            __half result3 = 0.0f;

            for (std::uint32_t corner = 0u; corner < 8u; ++corner) {
                const bool high_x         = (corner & 1u) != 0u;
                const bool high_y         = (corner & 2u) != 0u;
                const bool high_z         = (corner & 4u) != 0u;
                const float weight        = (high_x ? pos_x : 1.0f - pos_x) * (high_y ? pos_y : 1.0f - pos_y) * (high_z ? pos_z : 1.0f - pos_z);
                const std::uint32_t index = grid_index(hashmap_size, resolution, high_x ? grid_x + 1u : grid_x, high_y ? grid_y + 1u : grid_y, high_z ? grid_z + 1u : grid_z) * NetworkLayout<network_cuda_shape>::grid_features_per_level;
                const __half weight_half  = weight;
                result0                   = __hfma(weight_half, grid[index + 0u], result0);
                result1                   = __hfma(weight_half, grid[index + 1u], result1);
                result2                   = __hfma(weight_half, grid[index + 2u], result2);
                result3                   = __hfma(weight_half, grid[index + 3u], result3);
            }

            encoded_positions[i + (level * NetworkLayout<network_cuda_shape>::grid_features_per_level + 0u) * sample_count] = result0;
            encoded_positions[i + (level * NetworkLayout<network_cuda_shape>::grid_features_per_level + 1u) * sample_count] = result1;
            encoded_positions[i + (level * NetworkLayout<network_cuda_shape>::grid_features_per_level + 2u) * sample_count] = result2;
            encoded_positions[i + (level * NetworkLayout<network_cuda_shape>::grid_features_per_level + 3u) * sample_count] = result3;
        }

        __global__ void encode_grid_backward_kernel(const std::uint32_t sample_count, const NetworkGridOffsets grid_offsets, const float* __restrict__ sample_coords, const __half* __restrict__ encoded_position_gradients, __half* __restrict__ grid_gradients) {
            const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t i      = (thread * 2u) / NetworkLayout<network_cuda_shape>::grid_features_per_level;
            if (i >= sample_count) return;

            const std::uint32_t level             = blockIdx.y;
            const std::uint32_t feature           = thread * 2u - i * NetworkLayout<network_cuda_shape>::grid_features_per_level;
            const std::uint32_t level_offset      = grid_offsets.values[level];
            const std::uint32_t next_level_offset = grid_offsets.values[level + 1u];
            grid_gradients += level_offset * NetworkLayout<network_cuda_shape>::grid_features_per_level;
            const std::uint32_t hashmap_size = next_level_offset - level_offset;
            std::uint32_t resolution         = NetworkLayout<network_cuda_shape>::grid_base_resolution;
            for (std::uint32_t index = 0u; index < level; ++index) resolution *= NetworkLayout<network_cuda_shape>::grid_resolution_scale;
            const float scale   = static_cast<float>(resolution) - 1.0f;
            const float* sample = sample_coords + static_cast<std::uint64_t>(i) * 7u;

            float pos_x          = 0.0f;
            float pos_y          = 0.0f;
            float pos_z          = 0.0f;
            std::uint32_t grid_x = 0u;
            std::uint32_t grid_y = 0u;
            std::uint32_t grid_z = 0u;
            grid_position_fraction(sample[0], pos_x, grid_x, scale);
            grid_position_fraction(sample[1], pos_y, grid_y, scale);
            grid_position_fraction(sample[2], pos_z, grid_z, scale);

            const __half grad0 = encoded_position_gradients[i + (level * NetworkLayout<network_cuda_shape>::grid_features_per_level + feature + 0u) * sample_count];
            const __half grad1 = encoded_position_gradients[i + (level * NetworkLayout<network_cuda_shape>::grid_features_per_level + feature + 1u) * sample_count];

            for (std::uint32_t corner = 0u; corner < 8u; ++corner) {
                const bool high_x         = (corner & 1u) != 0u;
                const bool high_y         = (corner & 2u) != 0u;
                const bool high_z         = (corner & 4u) != 0u;
                const float weight        = (high_x ? pos_x : 1.0f - pos_x) * (high_y ? pos_y : 1.0f - pos_y) * (high_z ? pos_z : 1.0f - pos_z);
                const std::uint32_t index = grid_index(hashmap_size, resolution, high_x ? grid_x + 1u : grid_x, high_y ? grid_y + 1u : grid_y, high_z ? grid_z + 1u : grid_z) * NetworkLayout<network_cuda_shape>::grid_features_per_level + feature;
                const __half weight_half  = weight;
                atomicAdd(reinterpret_cast<__half2*>(grid_gradients + index), __halves2half2(__hmul(weight_half, grad0), __hmul(weight_half, grad1)));
            }
        }

        __global__ void encode_spherical_harmonics_kernel(const std::uint32_t sample_count, const float* __restrict__ sample_coords, __half* __restrict__ output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= sample_count) return;

            const float* coord = sample_coords + static_cast<std::uint64_t>(i) * 7u;
            const float x      = coord[4] * 2.0f - 1.0f;
            const float y      = coord[5] * 2.0f - 1.0f;
            const float z      = coord[6] * 2.0f - 1.0f;
            const float xy     = x * y;
            const float xz     = x * z;
            const float yz     = y * z;
            const float x2     = x * x;
            const float y2     = y * y;
            const float z2     = z * z;

            output[i + 0u * sample_count]  = static_cast<__half>(0.28209479177387814f);
            output[i + 1u * sample_count]  = static_cast<__half>(-0.48860251190291987f * y);
            output[i + 2u * sample_count]  = static_cast<__half>(0.48860251190291987f * z);
            output[i + 3u * sample_count]  = static_cast<__half>(-0.48860251190291987f * x);
            output[i + 4u * sample_count]  = static_cast<__half>(1.0925484305920792f * xy);
            output[i + 5u * sample_count]  = static_cast<__half>(-1.0925484305920792f * yz);
            output[i + 6u * sample_count]  = static_cast<__half>(0.94617469575755997f * z2 - 0.31539156525251999f);
            output[i + 7u * sample_count]  = static_cast<__half>(-1.0925484305920792f * xz);
            output[i + 8u * sample_count]  = static_cast<__half>(0.54627421529603959f * x2 - 0.54627421529603959f * y2);
            output[i + 9u * sample_count]  = static_cast<__half>(0.59004358992664352f * y * (-3.0f * x2 + y2));
            output[i + 10u * sample_count] = static_cast<__half>(2.8906114426405538f * xy * z);
            output[i + 11u * sample_count] = static_cast<__half>(0.45704579946446572f * y * (1.0f - 5.0f * z2));
            output[i + 12u * sample_count] = static_cast<__half>(0.3731763325901154f * z * (5.0f * z2 - 3.0f));
            output[i + 13u * sample_count] = static_cast<__half>(0.45704579946446572f * x * (1.0f - 5.0f * z2));
            output[i + 14u * sample_count] = static_cast<__half>(1.4453057213202769f * z * (x2 - y2));
            output[i + 15u * sample_count] = static_cast<__half>(0.59004358992664352f * x * (-x2 + 3.0f * y2));
        }

        template <std::uint32_t PARAM_COUNT>
        __global__ void cast_params_to_half_kernel(const float* __restrict__ params_full_precision, __half* __restrict__ params) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= PARAM_COUNT) return;
            params[i] = static_cast<__half>(params_full_precision[i]);
        }

        __global__ void initialize_grid_params_kernel(const std::uint32_t seed, float* __restrict__ params_full_precision, __half* __restrict__ params, __half* __restrict__ param_gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= NetworkLayout<network_cuda_shape>::network_parameter_layout.grid_param_count) return;
            ::cuda::std::philox4x32 random{seed};
            random.set_counter({network_random_domain, ::cuda::std::to_underlying(NetworkRandomSequence::grid_parameters), 0u, i});
            ::cuda::std::uniform_real_distribution<float> unit_distribution;
            const float value        = unit_distribution(random) * 2e-4f - 1e-4f;
            params_full_precision[i] = value;
            params[i]                = static_cast<__half>(value);
            param_gradients[i]       = static_cast<__half>(0.0f);
        }

        __global__ void initialize_mlp_params_kernel(const std::uint32_t seed, float* __restrict__ params_full_precision, __half* __restrict__ params, __half* __restrict__ param_gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= NetworkLayout<network_cuda_shape>::network_parameter_layout.mlp_param_count) return;

            float scale = 0.0f;
            if (i < NetworkLayout<network_cuda_shape>::network_parameter_layout.density_output_weight_offset) scale = ::cuda::std::sqrt(6.0f / static_cast<float>(NetworkLayout<network_cuda_shape>::mlp_width + NetworkLayout<network_cuda_shape>::grid_output_width));
            else if (i < NetworkLayout<network_cuda_shape>::network_parameter_layout.rgb_input_weight_offset) scale = ::cuda::std::sqrt(6.0f / static_cast<float>(NetworkLayout<network_cuda_shape>::density_output_width + NetworkLayout<network_cuda_shape>::mlp_width));
            else if (i < NetworkLayout<network_cuda_shape>::network_parameter_layout.rgb_hidden_weight_offset) scale = ::cuda::std::sqrt(6.0f / static_cast<float>(NetworkLayout<network_cuda_shape>::mlp_width + NetworkLayout<network_cuda_shape>::rgb_input_width));
            else if (i < NetworkLayout<network_cuda_shape>::network_parameter_layout.rgb_output_weight_offset) scale = ::cuda::std::sqrt(6.0f / static_cast<float>(NetworkLayout<network_cuda_shape>::mlp_width + NetworkLayout<network_cuda_shape>::mlp_width));
            else scale = ::cuda::std::sqrt(6.0f / static_cast<float>(NetworkLayout<network_cuda_shape>::network_output_width + NetworkLayout<network_cuda_shape>::mlp_width));

            ::cuda::std::philox4x32 random{seed};
            random.set_counter({network_random_domain, ::cuda::std::to_underlying(NetworkRandomSequence::mlp_parameters), 0u, i});
            ::cuda::std::uniform_real_distribution<float> unit_distribution;
            const float value        = unit_distribution(random) * 2.0f * scale - scale;
            params_full_precision[i] = value;
            params[i]                = static_cast<__half>(value);
            param_gradients[i]       = static_cast<__half>(0.0f);
        }

        template <typename Fragment>
        __device__ void relu_fragment(Fragment& fragment) {
            for (int i = 0; i < static_cast<int>(fragment.num_elements); ++i) fragment.x[i] = __hmax(fragment.x[i], static_cast<__half>(0.0f));
        }

        template <typename Fragment, typename ForwardFragment>
        __device__ void relu_backward_fragment(Fragment& fragment, const ForwardFragment& forward_fragment) {
            for (int i = 0; i < static_cast<int>(fragment.num_elements); ++i) fragment.x[i] = fragment.x[i] * static_cast<__half>(forward_fragment.x[i] > static_cast<__half>(0.0f));
        }

        __device__ void mlp_input_layer_forward(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, __half* __restrict__ hidden_threadblock, const std::uint32_t batch_size) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::col_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[8u];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % NetworkLayout<network_cuda_shape>::mlp_width;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / NetworkLayout<network_cuda_shape>::mlp_width;
            const std::uint32_t weights_col = 16u * wi;

            __half* __restrict__ weights_shmem        = act_shmem + 16u * (NetworkLayout<network_cuda_shape>::grid_output_width + 8u);
            constexpr std::uint32_t n_elems_per_load  = NetworkLayout<network_cuda_shape>::mlp_width_blocks * 32u * 8u;
            const std::uint32_t thread_elem_idx       = (li + wi * 32u) * 8u;
            constexpr std::uint32_t n_weight_elements = NetworkLayout<network_cuda_shape>::mlp_width * NetworkLayout<network_cuda_shape>::grid_output_width;

            for (std::uint32_t idx = thread_elem_idx; idx < n_weight_elements; idx += n_elems_per_load) {
                const std::uint32_t idx_skewed                       = idx + idx / NetworkLayout<network_cuda_shape>::grid_output_width * 8u;
                *reinterpret_cast<int4*>(&weights_shmem[idx_skewed]) = *reinterpret_cast<const int4*>(&weights_this_layer[idx]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < 8u; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < NetworkLayout<network_cuda_shape>::grid_output_width / 16u; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, input_threadblock + 16u * i * batch_size + 16u * l, batch_size);
                    nvcuda::wmma::load_matrix_sync(weights_frag, weights_shmem + 16u * i + weights_col * (NetworkLayout<network_cuda_shape>::grid_output_width + 8u), NetworkLayout<network_cuda_shape>::grid_output_width + 8u);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
                }
                relu_fragment(result_frag[l]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < 8u; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u), result_frag[l], NetworkLayout<network_cuda_shape>::mlp_width + 8u, nvcuda::wmma::mem_row_major);

            __syncthreads();

            if (hidden_threadblock != nullptr)
                for (std::uint32_t i = 0u; i < 8u; ++i) *reinterpret_cast<int4*>(&hidden_threadblock[lane_offset + (row + 16u * i) * NetworkLayout<network_cuda_shape>::mlp_width]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u)]);
        }

        __device__ void mlp_hidden_layer_forward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ hidden_threadblock) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[NetworkLayout<network_cuda_shape>::mlp_width_blocks];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[8u];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % NetworkLayout<network_cuda_shape>::mlp_width;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / NetworkLayout<network_cuda_shape>::mlp_width;
            const std::uint32_t weights_col = 16u * wi;

            __syncthreads();

            for (std::uint32_t i = 0u; i < NetworkLayout<network_cuda_shape>::mlp_width_blocks; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i + weights_col * NetworkLayout<network_cuda_shape>::mlp_width, NetworkLayout<network_cuda_shape>::mlp_width);

            for (std::uint32_t l = 0u; l < 8u; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < NetworkLayout<network_cuda_shape>::mlp_width_blocks; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * l) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u), NetworkLayout<network_cuda_shape>::mlp_width + 8u);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
                }
                relu_fragment(result_frag[l]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < 8u; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + l * 16u * (NetworkLayout<network_cuda_shape>::mlp_width + 8u), result_frag[l], NetworkLayout<network_cuda_shape>::mlp_width + 8u, nvcuda::wmma::mem_row_major);

            __syncthreads();

            if (hidden_threadblock != nullptr)
                for (std::uint32_t i = 0u; i < 8u; ++i) *reinterpret_cast<int4*>(&hidden_threadblock[lane_offset + (row + 16u * i) * NetworkLayout<network_cuda_shape>::mlp_width]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u)]);
        }

        __device__ void mlp_last_layer_forward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ out, const std::uint32_t output_stride, const nvcuda::wmma::layout_t output_layout) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[NetworkLayout<network_cuda_shape>::mlp_width_blocks];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag;

            const std::uint32_t li = threadIdx.x;
            const std::uint32_t wi = threadIdx.y;

            __half* __restrict__ weights_shmem = act_shmem + 8u * 16u * (NetworkLayout<network_cuda_shape>::mlp_width + 8u);
            const std::uint32_t weights_row    = (8u * li) % NetworkLayout<network_cuda_shape>::mlp_width;
            const std::uint32_t weights_col    = (8u * li + 8u * 32u * wi) / NetworkLayout<network_cuda_shape>::mlp_width;

            *reinterpret_cast<int4*>(&weights_shmem[weights_row + weights_col * (NetworkLayout<network_cuda_shape>::mlp_width + 8u)]) = *reinterpret_cast<const int4*>(&weights_this_layer[weights_row + weights_col * NetworkLayout<network_cuda_shape>::mlp_width]);
            __syncthreads();

            for (std::uint32_t i = 0u; i < NetworkLayout<network_cuda_shape>::mlp_width_blocks; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16u * i, NetworkLayout<network_cuda_shape>::mlp_width + 8u);

            for (std::uint32_t idx = wi; idx < 8u; idx += NetworkLayout<network_cuda_shape>::mlp_width_blocks) {
                nvcuda::wmma::fill_fragment(result_frag, 0.0f);
                for (std::uint32_t i = 0u; i < NetworkLayout<network_cuda_shape>::mlp_width_blocks; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * idx) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u), NetworkLayout<network_cuda_shape>::mlp_width + 8u);
                    nvcuda::wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
                }

                if (output_layout == nvcuda::wmma::mem_row_major) nvcuda::wmma::store_matrix_sync(out + idx * 16u * output_stride, result_frag, output_stride, output_layout);
                else nvcuda::wmma::store_matrix_sync(out + idx * 16u, result_frag, output_stride, output_layout);
            }
        }

        template <bool OutputRowMajor, std::uint32_t HiddenLayers>
        __global__ void mlp_forward_64_relu_kernel(const std::uint32_t batch_size, const __half* __restrict__ input, const __half* __restrict__ weights, __half* __restrict__ hidden, __half* __restrict__ output) {
            extern __shared__ __half shmem[];
            const std::uint32_t elem_idx                = 16u * blockIdx.x * 8u;
            constexpr std::uint32_t first_layer_params  = NetworkLayout<network_cuda_shape>::mlp_width * NetworkLayout<network_cuda_shape>::grid_output_width;
            constexpr std::uint32_t hidden_layer_params = NetworkLayout<network_cuda_shape>::mlp_width * NetworkLayout<network_cuda_shape>::mlp_width;

            mlp_input_layer_forward(shmem, input + elem_idx, weights, hidden == nullptr ? nullptr : hidden + elem_idx * NetworkLayout<network_cuda_shape>::mlp_width, batch_size);
            if constexpr (HiddenLayers == 2u) mlp_hidden_layer_forward(shmem, weights + first_layer_params, hidden == nullptr ? nullptr : hidden + static_cast<std::uint64_t>(NetworkLayout<network_cuda_shape>::mlp_width) * batch_size + elem_idx * NetworkLayout<network_cuda_shape>::mlp_width);

            const __half* last_weights = weights + first_layer_params + (HiddenLayers - 1u) * hidden_layer_params;
            if constexpr (OutputRowMajor) mlp_last_layer_forward(shmem, last_weights, output + elem_idx * NetworkLayout<network_cuda_shape>::network_output_width, NetworkLayout<network_cuda_shape>::network_output_width, nvcuda::wmma::mem_row_major);
            else mlp_last_layer_forward(shmem, last_weights, output + elem_idx, batch_size, nvcuda::wmma::mem_col_major);
        }

        __device__ void mlp_hidden_layer_backward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, const __half* __restrict__ forward_hidden, __half* __restrict__ backward_hidden) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag[NetworkLayout<network_cuda_shape>::mlp_width_blocks];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[8u];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % NetworkLayout<network_cuda_shape>::mlp_width;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / NetworkLayout<network_cuda_shape>::mlp_width;
            const std::uint32_t weights_col = 16u * wi;

            __syncthreads();

            for (std::uint32_t i = 0u; i < NetworkLayout<network_cuda_shape>::mlp_width_blocks; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i * NetworkLayout<network_cuda_shape>::mlp_width + weights_col, NetworkLayout<network_cuda_shape>::mlp_width);

            for (std::uint32_t l = 0u; l < 8u; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < NetworkLayout<network_cuda_shape>::mlp_width_blocks; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * l) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u), NetworkLayout<network_cuda_shape>::mlp_width + 8u);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
                }

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
                nvcuda::wmma::load_matrix_sync(forward_frag, forward_hidden + weights_col + l * 16u * NetworkLayout<network_cuda_shape>::mlp_width, NetworkLayout<network_cuda_shape>::mlp_width);
                relu_backward_fragment(result_frag[l], forward_frag);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < 8u; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u), result_frag[l], NetworkLayout<network_cuda_shape>::mlp_width + 8u, nvcuda::wmma::mem_row_major);

            __syncthreads();

            for (std::uint32_t i = 0u; i < 8u; ++i) *reinterpret_cast<int4*>(&backward_hidden[lane_offset + (row + i * 16u) * NetworkLayout<network_cuda_shape>::mlp_width]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u)]);
        }

        template <typename OutputLayout, std::uint32_t HiddenLayers>
        __global__ void mlp_backward_hidden_64_relu_kernel(const std::uint32_t batch_size, const __half* __restrict__ dloss_doutput, const __half* __restrict__ weights, const __half* __restrict__ forward_hidden, __half* __restrict__ backward_hidden, const std::uint32_t output_stride) {
            const std::uint32_t wi            = threadIdx.y;
            const std::uint32_t elem_idx_base = 16u * blockIdx.x * 8u;

            extern __shared__ __half shmem[];
            __half* act_shmem = shmem;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, OutputLayout> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[8u];

            const std::uint32_t weights_col             = 16u * wi;
            constexpr std::uint32_t first_layer_params  = NetworkLayout<network_cuda_shape>::mlp_width * NetworkLayout<network_cuda_shape>::grid_output_width;
            constexpr std::uint32_t hidden_layer_params = NetworkLayout<network_cuda_shape>::mlp_width * NetworkLayout<network_cuda_shape>::mlp_width;
            const __half* last_weights                  = weights + first_layer_params + (HiddenLayers - 1u) * hidden_layer_params;
            const __half* forward_last                  = forward_hidden + static_cast<std::uint64_t>(HiddenLayers - 1u) * NetworkLayout<network_cuda_shape>::mlp_width * batch_size;
            nvcuda::wmma::load_matrix_sync(weights_frag, last_weights + weights_col, NetworkLayout<network_cuda_shape>::mlp_width);

            for (std::uint32_t l = 0u; l < 8u; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);

                if constexpr (std::is_same_v<OutputLayout, nvcuda::wmma::row_major>) nvcuda::wmma::load_matrix_sync(act_frag, dloss_doutput + (elem_idx_base + 16u * l) * output_stride, output_stride);
                else nvcuda::wmma::load_matrix_sync(act_frag, dloss_doutput + elem_idx_base + 16u * l, output_stride);

                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
                nvcuda::wmma::load_matrix_sync(forward_frag, forward_last + weights_col + (elem_idx_base + l * 16u) * NetworkLayout<network_cuda_shape>::mlp_width, NetworkLayout<network_cuda_shape>::mlp_width);
                relu_backward_fragment(result_frag[l], forward_frag);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < 8u; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u), result_frag[l], NetworkLayout<network_cuda_shape>::mlp_width + 8u, nvcuda::wmma::mem_row_major);

            __syncthreads();

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t lane_offset = (8u * li) % NetworkLayout<network_cuda_shape>::mlp_width;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / NetworkLayout<network_cuda_shape>::mlp_width;

            for (std::uint32_t i = 0u; i < 8u; ++i) *reinterpret_cast<int4*>(&backward_hidden[lane_offset + (row + elem_idx_base + i * 16u) * NetworkLayout<network_cuda_shape>::mlp_width]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (NetworkLayout<network_cuda_shape>::mlp_width + 8u)]);

            if constexpr (HiddenLayers == 2u) mlp_hidden_layer_backward(act_shmem, weights + first_layer_params, forward_hidden + elem_idx_base * NetworkLayout<network_cuda_shape>::mlp_width, backward_hidden + static_cast<std::uint64_t>(NetworkLayout<network_cuda_shape>::mlp_width) * batch_size + elem_idx_base * NetworkLayout<network_cuda_shape>::mlp_width);
        }

        __global__ void extract_density_kernel(const std::uint32_t batch_size, const __half* __restrict__ density_output, __half* __restrict__ network_output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;
            network_output[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + 3u] = density_output[i];
        }

        __global__ void extract_rgb_gradients_kernel(const std::uint32_t batch_size, const __half* __restrict__ network_output_gradients, __half* __restrict__ rgb_output_gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;

            const __half zero = 0.0f;
            for (std::uint32_t j = 0u; j < NetworkLayout<network_cuda_shape>::network_output_width; ++j) rgb_output_gradients[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + j] = zero;
            rgb_output_gradients[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + 0u] = network_output_gradients[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + 0u];
            rgb_output_gradients[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + 1u] = network_output_gradients[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + 1u];
            rgb_output_gradients[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + 2u] = network_output_gradients[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + 2u];
        }

        __global__ void add_density_gradient_kernel(const std::uint32_t batch_size, const __half* __restrict__ network_output_gradients, __half* __restrict__ density_output_gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;
            density_output_gradients[i] = density_output_gradients[i] + network_output_gradients[static_cast<std::uint64_t>(i) * NetworkLayout<network_cuda_shape>::network_output_width + 3u];
        }

        __global__ void adam_step_kernel(float* __restrict__ params_full_precision, __half* __restrict__ params, const __half* __restrict__ gradients, float* __restrict__ first_moments, float* __restrict__ second_moments, std::uint32_t* __restrict__ param_steps) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= NetworkLayout<network_cuda_shape>::network_parameter_layout.total_param_count) return;

            float gradient = static_cast<float>(gradients[i]) / 128.0F;
            if (i >= NetworkLayout<network_cuda_shape>::network_parameter_layout.mlp_param_count && gradient == 0.0f) return;

            const float param = params_full_precision[i];
            if (i < NetworkLayout<network_cuda_shape>::network_parameter_layout.mlp_param_count) gradient += 1e-6F * param;

            const float gradient_sq  = gradient * gradient;
            const float first_moment = first_moments[i] = 0.9F * first_moments[i] + (1.0f - 0.9F) * gradient;
            const float second_moment = second_moments[i] = 0.99F * second_moments[i] + (1.0f - 0.99F) * gradient_sq;
            const std::uint32_t step                      = ++param_steps[i];
            const float corrected_lr                      = 1e-2F * ::cuda::std::sqrt(1.0f - ::cuda::std::pow(0.99F, static_cast<float>(step))) / (1.0f - ::cuda::std::pow(0.9F, static_cast<float>(step)));
            const float updated_param                     = param - corrected_lr * first_moment / (::cuda::std::sqrt(second_moment) + 1e-15F);

            params_full_precision[i] = updated_param;
            params[i]                = static_cast<__half>(updated_param);
        }

        struct CublasLtMatrixLayout final {
            cublasLtOrder_t order;
            std::uint64_t rows;
            std::uint64_t columns;
            std::int64_t leading_dimension;
        };

        struct CublasLtPreference final {
            cublasLtMatmulPreference_t handle = nullptr;

            CublasLtPreference() = default;
            ~CublasLtPreference() noexcept {
                if (handle != nullptr) cublasLtMatmulPreferenceDestroy(handle);
            }

            CublasLtPreference(const CublasLtPreference&)            = delete;
            CublasLtPreference& operator=(const CublasLtPreference&) = delete;
            CublasLtPreference(CublasLtPreference&&)                 = delete;
            CublasLtPreference& operator=(CublasLtPreference&&)      = delete;
        };

        void initialize_cublaslt_matmul_plan(const cublasLtHandle_t handle, const std::string_view name, cublasLtMatmulDesc_t& operation_descriptor, cublasLtMatrixLayout_t& a_descriptor, cublasLtMatrixLayout_t& b_descriptor, cublasLtMatrixLayout_t& output_descriptor, cublasLtMatmulAlgo_t& algorithm, const CublasLtMatrixLayout a_layout, const CublasLtMatrixLayout b_layout, const CublasLtMatrixLayout output_layout) {
            constexpr std::size_t max_workspace_bytes = (static_cast<std::size_t>(64u) * 1024u * 1024u);
            cublasLtMatmulHeuristicResult_t heuristic = {};
            CublasLtPreference preference;
            int returned_algorithm_count = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&operation_descriptor, CUBLAS_COMPUTE_16F, CUDA_R_16F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} operation descriptor failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&a_descriptor, CUDA_R_16F, a_layout.rows, a_layout.columns, a_layout.leading_dimension); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} A layout failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&b_descriptor, CUDA_R_16F, b_layout.rows, b_layout.columns, b_layout.leading_dimension); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} B layout failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&output_descriptor, CUDA_R_16F, output_layout.rows, output_layout.columns, output_layout.leading_dimension); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} output layout failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(a_descriptor, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_layout.order, sizeof(a_layout.order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} A order failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(b_descriptor, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_layout.order, sizeof(b_layout.order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} B order failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(output_descriptor, CUBLASLT_MATRIX_LAYOUT_ORDER, &output_layout.order, sizeof(output_layout.order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} output order failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference.handle); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} preference failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(preference.handle, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} workspace preference failed: {}", name, cublasGetStatusString(status))};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, operation_descriptor, a_descriptor, b_descriptor, output_descriptor, output_descriptor, preference.handle, 1, &heuristic, &returned_algorithm_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} algorithm selection failed: {}", name, cublasGetStatusString(status))};
            if (returned_algorithm_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} has no supported algorithm.", name)};
            algorithm = heuristic.algo;
        }

        void cublaslt_matmul(const ::cuda::stream_ref stream, const cublasLtHandle_t handle, const std::string_view name, const cublasLtMatmulDesc_t operation_descriptor, const cublasLtMatrixLayout_t a_descriptor, const cublasLtMatrixLayout_t b_descriptor, const cublasLtMatrixLayout_t output_descriptor, const cublasLtMatmulAlgo_t& algorithm, const __half* const a, const __half* const b, __half* const output, std::uint8_t* const workspace) {
            const auto alpha = static_cast<__half>(1.0f);
            const auto beta  = static_cast<__half>(0.0f);
            if (const cublasStatus_t status = cublasLtMatmul(handle, operation_descriptor, &alpha, a, a_descriptor, b, b_descriptor, &beta, output, output_descriptor, output, output_descriptor, &algorithm, workspace, (static_cast<std::size_t>(64u) * 1024u * 1024u), stream.get()); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("{} failed: {}", name, cublasGetStatusString(status))};
        }

    } // namespace
} // namespace physica::reconstruction::instant_ngp::kernels

namespace physica::reconstruction::instant_ngp::kernels {
    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::initialize_cublaslt_matmul_plans(const cublasLtHandle_t cublaslt, cublasLtMatmulDesc_t* const operation_descriptors, cublasLtMatrixLayout_t* const a_descriptors, cublasLtMatrixLayout_t* const b_descriptors, cublasLtMatrixLayout_t* const output_descriptors, cublasLtMatmulAlgo_t* const algorithms) {
        constexpr int batch = static_cast<int>(NetworkLayout<Shape>::network_batch_size);
        initialize_cublaslt_matmul_plan(cublaslt, "rgb last weight gradients", operation_descriptors[0], a_descriptors[0], b_descriptors[0], output_descriptors[0], algorithms[0], {CUBLASLT_ORDER_COL, NetworkLayout<Shape>::network_output_width, batch, NetworkLayout<Shape>::network_output_width}, {CUBLASLT_ORDER_ROW, batch, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::mlp_width}, {CUBLASLT_ORDER_ROW, NetworkLayout<Shape>::network_output_width, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::mlp_width});
        initialize_cublaslt_matmul_plan(cublaslt, "rgb hidden weight gradients", operation_descriptors[1], a_descriptors[1], b_descriptors[1], output_descriptors[1], algorithms[1], {CUBLASLT_ORDER_COL, NetworkLayout<Shape>::mlp_width, batch, NetworkLayout<Shape>::mlp_width}, {CUBLASLT_ORDER_ROW, batch, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::mlp_width}, {CUBLASLT_ORDER_ROW, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::mlp_width});
        initialize_cublaslt_matmul_plan(cublaslt, "rgb first weight gradients", operation_descriptors[2], a_descriptors[2], b_descriptors[2], output_descriptors[2], algorithms[2], {CUBLASLT_ORDER_COL, NetworkLayout<Shape>::mlp_width, batch, NetworkLayout<Shape>::mlp_width}, {CUBLASLT_ORDER_COL, batch, NetworkLayout<Shape>::rgb_input_width, batch}, {CUBLASLT_ORDER_ROW, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::rgb_input_width, NetworkLayout<Shape>::rgb_input_width});
        initialize_cublaslt_matmul_plan(cublaslt, "rgb input gradients", operation_descriptors[3], a_descriptors[3], b_descriptors[3], output_descriptors[3], algorithms[3], {CUBLASLT_ORDER_COL, NetworkLayout<Shape>::rgb_input_width, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::rgb_input_width}, {CUBLASLT_ORDER_COL, NetworkLayout<Shape>::mlp_width, batch, NetworkLayout<Shape>::mlp_width}, {CUBLASLT_ORDER_ROW, NetworkLayout<Shape>::rgb_input_width, batch, batch});
        initialize_cublaslt_matmul_plan(cublaslt, "density last weight gradients", operation_descriptors[4], a_descriptors[4], b_descriptors[4], output_descriptors[4], algorithms[4], {CUBLASLT_ORDER_ROW, NetworkLayout<Shape>::density_output_width, batch, batch}, {CUBLASLT_ORDER_ROW, batch, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::mlp_width}, {CUBLASLT_ORDER_ROW, NetworkLayout<Shape>::density_output_width, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::mlp_width});
        initialize_cublaslt_matmul_plan(cublaslt, "density first weight gradients", operation_descriptors[5], a_descriptors[5], b_descriptors[5], output_descriptors[5], algorithms[5], {CUBLASLT_ORDER_COL, NetworkLayout<Shape>::mlp_width, batch, NetworkLayout<Shape>::mlp_width}, {CUBLASLT_ORDER_COL, batch, NetworkLayout<Shape>::grid_output_width, batch}, {CUBLASLT_ORDER_ROW, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::grid_output_width, NetworkLayout<Shape>::grid_output_width});
        initialize_cublaslt_matmul_plan(cublaslt, "density input gradients", operation_descriptors[6], a_descriptors[6], b_descriptors[6], output_descriptors[6], algorithms[6], {CUBLASLT_ORDER_COL, NetworkLayout<Shape>::grid_output_width, NetworkLayout<Shape>::mlp_width, NetworkLayout<Shape>::grid_output_width}, {CUBLASLT_ORDER_COL, NetworkLayout<Shape>::mlp_width, batch, NetworkLayout<Shape>::mlp_width}, {CUBLASLT_ORDER_ROW, NetworkLayout<Shape>::grid_output_width, batch, batch});
    }
    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::initialize_mlp_parameters(const ::cuda::stream_ref stream, const std::uint32_t seed, float* const params_full_precision, std::uint16_t* const params, std::uint16_t* const param_gradients) {
        constexpr std::uint32_t blocks = ::cuda::ceil_div(NetworkLayout<Shape>::network_parameter_layout.mlp_param_count, 128u);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims<128u>())), initialize_mlp_params_kernel, seed, params_full_precision, reinterpret_cast<__half*>(params), reinterpret_cast<__half*>(param_gradients));
    }

    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::initialize_grid_parameters(const ::cuda::stream_ref stream, const std::uint32_t seed, float* const params_full_precision, std::uint16_t* const params, std::uint16_t* const param_gradients) {

        constexpr std::uint32_t blocks = ::cuda::ceil_div(NetworkLayout<Shape>::network_parameter_layout.grid_param_count, 128u);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims<128u>())), initialize_grid_params_kernel, seed, params_full_precision + NetworkLayout<Shape>::network_parameter_layout.grid_param_offset, reinterpret_cast<__half*>(params + NetworkLayout<Shape>::network_parameter_layout.grid_param_offset), reinterpret_cast<__half*>(param_gradients + NetworkLayout<Shape>::network_parameter_layout.grid_param_offset));
    }

    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::upload_trainable_parameters(const ::cuda::stream_ref stream, const float* const params_full_precision, float* const out_params_full_precision, std::uint16_t* const out_params, std::uint16_t* const out_param_gradients) {

        constexpr std::size_t parameter_count = NetworkLayout<Shape>::network_parameter_layout.total_param_count;
        ::cuda::copy_bytes(stream, ::cuda::std::span{params_full_precision, parameter_count}, ::cuda::std::span{out_params_full_precision, parameter_count});
        ::cuda::fill_bytes(stream, ::cuda::std::span{out_param_gradients, parameter_count}, 0u);

        constexpr std::uint32_t blocks = ::cuda::ceil_div(NetworkLayout<Shape>::network_parameter_layout.total_param_count, 128u);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims<128u>())), cast_params_to_half_kernel<NetworkLayout<Shape>::network_parameter_layout.total_param_count>, out_params_full_precision, reinterpret_cast<__half*>(out_params));
    }

    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::evaluate_network(const ::cuda::stream_ref stream, const std::uint32_t sample_count, const float* const sample_coords, const std::uint16_t* const params, std::uint16_t* const density_input, std::uint16_t* const rgb_input, std::uint16_t* const network_output) {
        if (sample_count == 0u) return;

        constexpr int forward_shmem = sizeof(__half) * (16u + 16u * 8u) * (NetworkLayout<Shape>::mlp_width + 8u);
        constexpr dim3 threads{32u, NetworkLayout<Shape>::mlp_width_blocks, 1u};
        for (std::uint32_t offset = 0u; offset < sample_count; offset += NetworkLayout<Shape>::network_batch_size) {
            const std::uint32_t chunk              = ::cuda::std::min(NetworkLayout<Shape>::network_batch_size, sample_count - offset);
            const float* const chunk_sample_coords = sample_coords + static_cast<std::uint64_t>(offset) * 7u;

            const dim3 grid_blocks{::cuda::ceil_div(chunk, 512u), NetworkLayout<Shape>::grid_n_levels, 1u};
            ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(grid_blocks), ::cuda::block_dims<512u>())), encode_grid_forward_kernel, chunk, NetworkLayout<Shape>::network_parameter_layout.grid_offsets, chunk_sample_coords, reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.grid_param_offset), reinterpret_cast<__half*>(density_input));

            const auto linear_config = ::cuda::distribute<128u>(chunk);
            ::cuda::launch(stream, linear_config, encode_spherical_harmonics_kernel, chunk, chunk_sample_coords, reinterpret_cast<__half*>(rgb_input) + static_cast<std::uint64_t>(NetworkLayout<Shape>::density_output_width) * chunk);

            const dim3 blocks{chunk / (16u * 8u), 1u, 1u};
            const auto mlp_config = ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims(threads)), ::cuda::dynamic_shared_memory<std::uint8_t[]>(forward_shmem));
            ::cuda::launch(stream, mlp_config, mlp_forward_64_relu_kernel<false, NetworkLayout<Shape>::density_hidden_layers>, chunk, reinterpret_cast<const __half*>(density_input), reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.density_param_offset), nullptr, reinterpret_cast<__half*>(rgb_input));

            ::cuda::launch(stream, mlp_config, mlp_forward_64_relu_kernel<true, NetworkLayout<Shape>::rgb_hidden_layers>, chunk, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.rgb_param_offset), nullptr, reinterpret_cast<__half*>(network_output) + static_cast<std::uint64_t>(offset) * NetworkLayout<Shape>::network_output_width);

            ::cuda::launch(stream, linear_config, extract_density_kernel, chunk, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<__half*>(network_output) + static_cast<std::uint64_t>(offset) * NetworkLayout<Shape>::network_output_width);
        }
    }

    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::evaluate_density_network(const ::cuda::stream_ref stream, const std::uint32_t sample_count, const float* const sample_coords, const std::uint16_t* const params, std::uint16_t* const density_input, std::uint16_t* const density_output) {
        if (sample_count == 0u) return;
        constexpr int forward_shmem = sizeof(__half) * (16u + 16u * 8u) * (NetworkLayout<Shape>::mlp_width + 8u);
        constexpr dim3 threads{32u, NetworkLayout<Shape>::mlp_width_blocks, 1u};
        for (std::uint32_t offset = 0u; offset < sample_count; offset += NetworkLayout<Shape>::network_batch_size) {
            const std::uint32_t chunk              = ::cuda::std::min(NetworkLayout<Shape>::network_batch_size, sample_count - offset);
            const float* const chunk_sample_coords = sample_coords + static_cast<std::uint64_t>(offset) * 7u;
            const dim3 grid_blocks{::cuda::ceil_div(chunk, 512u), NetworkLayout<Shape>::grid_n_levels, 1u};
            ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(grid_blocks), ::cuda::block_dims<512u>())), encode_grid_forward_kernel, chunk, NetworkLayout<Shape>::network_parameter_layout.grid_offsets, chunk_sample_coords, reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.grid_param_offset), reinterpret_cast<__half*>(density_input));

            const dim3 blocks{chunk / (16u * 8u), 1u, 1u};
            const auto mlp_config = ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims(threads)), ::cuda::dynamic_shared_memory<std::uint8_t[]>(forward_shmem));
            ::cuda::launch(stream, mlp_config, mlp_forward_64_relu_kernel<false, NetworkLayout<Shape>::density_hidden_layers>, chunk, reinterpret_cast<const __half*>(density_input), reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.density_param_offset), nullptr, reinterpret_cast<__half*>(density_output));
        }
    }

    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::forward_network(const ::cuda::stream_ref stream, const float* const sample_coords, const std::uint16_t* const params, std::uint16_t* const density_input, std::uint16_t* const rgb_input, std::uint16_t* const density_forward_hidden, std::uint16_t* const rgb_forward_hidden, std::uint16_t* const network_output) {

        constexpr dim3 grid_blocks{::cuda::ceil_div(NetworkLayout<Shape>::network_batch_size, 512u), NetworkLayout<Shape>::grid_n_levels, 1u};
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(grid_blocks), ::cuda::block_dims<512u>())), encode_grid_forward_kernel, NetworkLayout<Shape>::network_batch_size, NetworkLayout<Shape>::network_parameter_layout.grid_offsets, sample_coords, reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.grid_param_offset), reinterpret_cast<__half*>(density_input));

        constexpr auto linear_config = ::cuda::distribute<128u>(NetworkLayout<Shape>::network_batch_size);
        ::cuda::launch(stream, linear_config, encode_spherical_harmonics_kernel, NetworkLayout<Shape>::network_batch_size, sample_coords, reinterpret_cast<__half*>(rgb_input) + static_cast<std::uint64_t>(NetworkLayout<Shape>::density_output_width) * NetworkLayout<Shape>::network_batch_size);

        constexpr int forward_shmem = sizeof(__half) * (16u + 16u * 8u) * (NetworkLayout<Shape>::mlp_width + 8u);
        constexpr dim3 threads{32u, NetworkLayout<Shape>::mlp_width_blocks, 1u};
        constexpr dim3 blocks{NetworkLayout<Shape>::network_batch_size / (16u * 8u), 1u, 1u};
        const auto mlp_config = ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims(threads)), ::cuda::dynamic_shared_memory<std::uint8_t[]>(forward_shmem));
        ::cuda::launch(stream, mlp_config, mlp_forward_64_relu_kernel<false, NetworkLayout<Shape>::density_hidden_layers>, NetworkLayout<Shape>::network_batch_size, reinterpret_cast<const __half*>(density_input), reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.density_param_offset), reinterpret_cast<__half*>(density_forward_hidden), reinterpret_cast<__half*>(rgb_input));

        ::cuda::launch(stream, mlp_config, mlp_forward_64_relu_kernel<true, NetworkLayout<Shape>::rgb_hidden_layers>, NetworkLayout<Shape>::network_batch_size, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.rgb_param_offset), reinterpret_cast<__half*>(rgb_forward_hidden), reinterpret_cast<__half*>(network_output));

        ::cuda::launch(stream, linear_config, extract_density_kernel, NetworkLayout<Shape>::network_batch_size, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<__half*>(network_output));
    }

    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::backward_network(const ::cuda::stream_ref stream, const float* const sample_coords, const std::uint16_t* const params, std::uint16_t* const gradients, const std::uint16_t* const density_input, const std::uint16_t* const rgb_input, const std::uint16_t* const density_forward_hidden, const std::uint16_t* const rgb_forward_hidden, const std::uint16_t* const network_output, const std::uint16_t* const network_output_gradients, std::uint16_t* const rgb_output_gradients, std::uint16_t* const rgb_input_gradients, std::uint16_t* const density_input_gradients, std::uint16_t* const density_backward_hidden, std::uint16_t* const rgb_backward_hidden, const cublasLtHandle_t cublaslt, const cublasLtMatmulDesc_t* const operation_descriptors, const cublasLtMatrixLayout_t* const a_descriptors, const cublasLtMatrixLayout_t* const b_descriptors, const cublasLtMatrixLayout_t* const output_descriptors, const cublasLtMatmulAlgo_t* const algorithms, std::uint8_t* const cublaslt_workspace) {

        constexpr auto linear_config                = ::cuda::distribute<128u>(NetworkLayout<Shape>::network_batch_size);
        constexpr std::uint64_t hidden_layer_stride = static_cast<std::uint64_t>(NetworkLayout<Shape>::mlp_width) * NetworkLayout<Shape>::network_batch_size;

        ::cuda::launch(stream, linear_config, extract_rgb_gradients_kernel, NetworkLayout<Shape>::network_batch_size, reinterpret_cast<const __half*>(network_output_gradients), reinterpret_cast<__half*>(rgb_output_gradients));

        constexpr int backward_shmem = sizeof(__half) * (16u * 8u) * (NetworkLayout<Shape>::mlp_width + 8u);
        constexpr dim3 threads{32u, NetworkLayout<Shape>::mlp_width_blocks, 1u};
        constexpr dim3 blocks{NetworkLayout<Shape>::network_batch_size / (16u * 8u), 1u, 1u};
        const auto backward_config = ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims(threads)), ::cuda::dynamic_shared_memory<std::uint8_t[]>(backward_shmem));
        ::cuda::launch(stream, backward_config, mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::row_major, NetworkLayout<Shape>::rgb_hidden_layers>, NetworkLayout<Shape>::network_batch_size, reinterpret_cast<const __half*>(rgb_output_gradients), reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.rgb_param_offset), reinterpret_cast<const __half*>(rgb_forward_hidden), reinterpret_cast<__half*>(rgb_backward_hidden), NetworkLayout<Shape>::network_output_width);

        cublaslt_matmul(stream, cublaslt, "rgb last weight gradients", operation_descriptors[0], a_descriptors[0], b_descriptors[0], output_descriptors[0], algorithms[0], reinterpret_cast<const __half*>(rgb_output_gradients), reinterpret_cast<const __half*>(rgb_forward_hidden) + (NetworkLayout<Shape>::rgb_hidden_layers - 1u) * hidden_layer_stride, reinterpret_cast<__half*>(gradients + NetworkLayout<Shape>::network_parameter_layout.rgb_output_weight_offset), cublaslt_workspace);

        cublaslt_matmul(stream, cublaslt, "rgb hidden weight gradients", operation_descriptors[1], a_descriptors[1], b_descriptors[1], output_descriptors[1], algorithms[1], reinterpret_cast<const __half*>(rgb_backward_hidden), reinterpret_cast<const __half*>(rgb_forward_hidden), reinterpret_cast<__half*>(gradients + NetworkLayout<Shape>::network_parameter_layout.rgb_hidden_weight_offset), cublaslt_workspace);

        cublaslt_matmul(stream, cublaslt, "rgb first weight gradients", operation_descriptors[2], a_descriptors[2], b_descriptors[2], output_descriptors[2], algorithms[2], reinterpret_cast<const __half*>(rgb_backward_hidden) + (NetworkLayout<Shape>::rgb_hidden_layers - 1u) * hidden_layer_stride, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<__half*>(gradients + NetworkLayout<Shape>::network_parameter_layout.rgb_param_offset), cublaslt_workspace);

        cublaslt_matmul(stream, cublaslt, "rgb input gradients", operation_descriptors[3], a_descriptors[3], b_descriptors[3], output_descriptors[3], algorithms[3], reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.rgb_param_offset), reinterpret_cast<const __half*>(rgb_backward_hidden) + (NetworkLayout<Shape>::rgb_hidden_layers - 1u) * hidden_layer_stride, reinterpret_cast<__half*>(rgb_input_gradients), cublaslt_workspace);

        ::cuda::launch(stream, linear_config, add_density_gradient_kernel, NetworkLayout<Shape>::network_batch_size, reinterpret_cast<const __half*>(network_output_gradients), reinterpret_cast<__half*>(rgb_input_gradients));

        ::cuda::launch(stream, backward_config, mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::col_major, NetworkLayout<Shape>::density_hidden_layers>, NetworkLayout<Shape>::network_batch_size, reinterpret_cast<const __half*>(rgb_input_gradients), reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.density_param_offset), reinterpret_cast<const __half*>(density_forward_hidden), reinterpret_cast<__half*>(density_backward_hidden), NetworkLayout<Shape>::network_batch_size);

        cublaslt_matmul(stream, cublaslt, "density last weight gradients", operation_descriptors[4], a_descriptors[4], b_descriptors[4], output_descriptors[4], algorithms[4], reinterpret_cast<const __half*>(rgb_input_gradients), reinterpret_cast<const __half*>(density_forward_hidden), reinterpret_cast<__half*>(gradients + NetworkLayout<Shape>::network_parameter_layout.density_output_weight_offset), cublaslt_workspace);

        cublaslt_matmul(stream, cublaslt, "density first weight gradients", operation_descriptors[5], a_descriptors[5], b_descriptors[5], output_descriptors[5], algorithms[5], reinterpret_cast<const __half*>(density_backward_hidden), reinterpret_cast<const __half*>(density_input), reinterpret_cast<__half*>(gradients + NetworkLayout<Shape>::network_parameter_layout.density_param_offset), cublaslt_workspace);

        cublaslt_matmul(stream, cublaslt, "density input gradients", operation_descriptors[6], a_descriptors[6], b_descriptors[6], output_descriptors[6], algorithms[6], reinterpret_cast<const __half*>(params + NetworkLayout<Shape>::network_parameter_layout.density_param_offset), reinterpret_cast<const __half*>(density_backward_hidden), reinterpret_cast<__half*>(density_input_gradients), cublaslt_workspace);

        ::cuda::fill_bytes(stream, ::cuda::std::span{gradients + NetworkLayout<Shape>::network_parameter_layout.grid_param_offset, static_cast<std::size_t>(NetworkLayout<Shape>::network_parameter_layout.grid_param_count)}, 0u);

        constexpr std::uint32_t grid_threads = ::cuda::ceil_div(NetworkLayout<Shape>::network_batch_size * NetworkLayout<Shape>::grid_features_per_level / 2u, 256u);
        constexpr dim3 grid_blocks{grid_threads, NetworkLayout<Shape>::grid_n_levels, 1u};
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(grid_blocks), ::cuda::block_dims<256u>())), encode_grid_backward_kernel, NetworkLayout<Shape>::network_batch_size, NetworkLayout<Shape>::network_parameter_layout.grid_offsets, sample_coords, reinterpret_cast<const __half*>(density_input_gradients), reinterpret_cast<__half*>(gradients + NetworkLayout<Shape>::network_parameter_layout.grid_param_offset));
    }

    template <NetworkKernelShape Shape>
    void NetworkKernels<Shape>::step_optimizer(const ::cuda::stream_ref stream, float* const params_full_precision, std::uint16_t* const params, const std::uint16_t* const gradients, float* const first_moments, float* const second_moments, std::uint32_t* const param_steps) {

        constexpr std::uint32_t blocks = ::cuda::ceil_div(NetworkLayout<Shape>::network_parameter_layout.total_param_count, 128u);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(blocks), ::cuda::block_dims<128u>())), adam_step_kernel, params_full_precision, reinterpret_cast<__half*>(params), reinterpret_cast<const __half*>(gradients), first_moments, second_moments, param_steps);
    }

    template struct NetworkKernels<network_cuda_shape>;

} // namespace physica::reconstruction::instant_ngp::kernels
