#include "inference-kernels.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <physica/cuda.h>
#include <cuda/launch>
#include <math_constants.h>
#include <mma.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <stdexcept>

namespace physica::neural::kernels {
    // A CTA owns sixteen queries. K and V share storage because their lifetimes
    // do not overlap. In particular, the BF16 d=512 path fits the 99 KiB limit.
    template<class T, int Dimension, int Keys> __global__ void tiled_attention(
        T* output, const T* query, const T* key, const T* value,
        const int queries, const int keys, const int heads, const int query_stride, const int key_stride) {
        extern __shared__ __align__(32) unsigned char memory[];
        T* q = reinterpret_cast<T*>(memory);
        T* kv = q + 16 * Dimension;
        T* probability = kv + Keys * Dimension;
        float* score = reinterpret_cast<float*>(probability + 16 * Keys);
        float* maximum = score + 16 * Keys;
        float* denominator = maximum + 16;
        float* rescale = denominator + 16;
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int head = blockIdx.y % heads;
        const int batch = blockIdx.y / heads;
        const int begin = blockIdx.x * 16;
        float accumulated[Dimension / 32][4]{};
        for (int i = threadIdx.x; i < 16 * Dimension; i += blockDim.x) {
            q[i] = begin + i / Dimension < queries ? query[(batch * queries + begin + i / Dimension) * query_stride + head * Dimension + i % Dimension] : T(0.0F);
        }
        if (threadIdx.x < 16) {
            maximum[threadIdx.x] = -CUDART_INF_F;
            denominator[threadIdx.x] = 0.0F;
        }
        __syncthreads();
        for (int start = 0; start < keys; start += Keys) {
            for (int i = threadIdx.x; i < Keys * Dimension; i += blockDim.x) kv[i] = start + i / Dimension < keys ? key[(batch * keys + start + i / Dimension) * key_stride + head * Dimension + i % Dimension] : T(0.0F);
            __syncthreads();
            for (int column = warp * 16; column < Keys; column += 64) {
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;
                nvcuda::wmma::fill_fragment(c, 0.0F);
                for (int d = 0; d < Dimension; d += 16) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major> a;
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::col_major> b;
                    nvcuda::wmma::load_matrix_sync(a, q + d, Dimension);
                    nvcuda::wmma::load_matrix_sync(b, kv + column * Dimension + d, Dimension);
                    nvcuda::wmma::mma_sync(c, a, b, c);
                }
                nvcuda::wmma::store_matrix_sync(score + column, c, Keys, nvcuda::wmma::mem_row_major);
            }
            __syncthreads();
            for (int row = warp; row < 16; row += 4) {
                float local_max = -CUDART_INF_F;
                for (int k = lane; k < Keys; k += 32) {
                    const float scaled = start + k < keys ? score[row * Keys + k] * rsqrtf(float(Dimension)) : -CUDART_INF_F;
                    score[row * Keys + k] = scaled;
                    local_max = fmaxf(local_max, scaled);
                }
                for (int offset = 16; offset; offset >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
                const float next_max = fmaxf(maximum[row], local_max);
                const float alpha = expf(maximum[row] - next_max);
                float sum = 0.0F;
                for (int k = lane; k < Keys; k += 32) {
                    const float p = expf(score[row * Keys + k] - next_max);
                    probability[row * Keys + k] = T(p);
                    sum += p;
                }
                for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, offset);
                if (lane == 0) {
                    maximum[row] = next_max;
                    denominator[row] = denominator[row] * alpha + sum;
                    rescale[row] = alpha;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int n = 0; n < Dimension / 32; ++n) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) accumulated[n][j] *= rescale[lane / 4 + j / 2 * 8];
            }
            for (int i = threadIdx.x; i < Keys * Dimension; i += blockDim.x) kv[i] = start + i / Dimension < keys ? value[(batch * keys + start + i / Dimension) * key_stride + head * Dimension + i % Dimension] : T(0.0F);
            __syncthreads();
            #pragma unroll
            for (int d = 0; d < Keys; d += 16) {
                std::uint32_t a[4];
                cute::SM75_U32x4_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(probability + (lane % 16) * Keys + d + lane / 16 * 8), a[0], a[1], a[2], a[3]);
                #pragma unroll
                for (int n = 0; n < Dimension / 32; ++n) {
                    std::uint32_t b0, b1;
                    cute::SM75_U16x4_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(kv + (d + lane % 16) * Dimension + warp * 8 + n * 32), b0, b1);
                    cute::SM80_16x8x16_F32BF16BF16F32_TN::fma(accumulated[n][0], accumulated[n][1], accumulated[n][2], accumulated[n][3],
                        a[0], a[1], a[2], a[3], b0, b1, accumulated[n][0], accumulated[n][1], accumulated[n][2], accumulated[n][3]);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int n = 0; n < Dimension / 32; ++n) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = lane / 4 + j / 2 * 8;
                const int column = warp * 8 + n * 32 + lane % 4 * 2 + j % 2;
                if (begin + row < queries) output[((batch * queries + begin + row) * heads + head) * Dimension + column] = T(accumulated[n][j] / denominator[row]);
            }
        }

    }

    template<class T> __global__ void clip_attention(T* output, const T* query, const T* key, const T* value,
        const int queries, const int keys, const int heads, const int query_stride, const int key_stride, const bool causal, const std::int32_t* positions) {
        __shared__ float scores[4][96];
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int row = blockIdx.x * 4 + warp;
        if (row >= queries) return;
        const int head = blockIdx.y % heads;
        const int batch = blockIdx.y / heads;
        const int last = causal ? (positions ? positions[batch] : row) : keys - 1;
        float maximum = -CUDART_INF_F;
        for (int k = lane; k < keys; k += 32) {
            float sum = 0.0F;
            for (int d = 0; d < 64; ++d) sum = fmaf(float(query[(batch * queries + row) * query_stride + head * 64 + d]), float(key[(batch * keys + k) * key_stride + head * 64 + d]), sum);
            const float score = k <= last ? sum * 0.125F : -CUDART_INF_F;
            scores[warp][k] = score;
            maximum = fmaxf(maximum, score);
        }
        for (int offset = 16; offset; offset >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        float denominator = 0.0F;
        for (int k = lane; k < keys; k += 32) {
            const float p = expf(scores[warp][k] - maximum);
            scores[warp][k] = p;
            denominator += p;
        }
        for (int offset = 16; offset; offset >>= 1) denominator += __shfl_xor_sync(0xffffffff, denominator, offset);
        __syncwarp();
        for (int d = lane; d < 64; d += 32) {
            float sum = 0.0F;
            for (int k = 0; k < keys; ++k) sum = fmaf(scores[warp][k] / denominator, float(value[(batch * keys + k) * key_stride + head * 64 + d]), sum);
            output[((batch * queries + row) * heads + head) * 64 + d] = T(sum);
        }
    }


    void attention(const ::cuda::stream_ref stream, void* output, const void* query, const void* key, const void* value, const int batch, const int queries, const int keys, const int heads, const int dimension, const int query_stride, const int key_stride, const int scalar, const bool causal, const std::int32_t* positions) {
        if (dimension == 64) ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(dim3((queries + 3) / 4, batch * heads)), ::cuda::block_dims(128))), clip_attention<__half>, static_cast<__half*>(output), static_cast<const __half*>(query), static_cast<const __half*>(key), static_cast<const __half*>(value), queries, keys, heads, query_stride, key_stride, causal, positions);
        else {
            constexpr int bytes = (16 * 512 + 64 * 512 + 16 * 64) * 2 + (16 * 64 + 48) * 4;
            ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(dim3((queries + 15) / 16, batch * heads)), ::cuda::block_dims(128)), ::cuda::dynamic_shared_memory<unsigned char[]>(bytes, ::cuda::non_portable)), tiled_attention<__nv_bfloat16, 512, 64>, static_cast<__nv_bfloat16*>(output), static_cast<const __nv_bfloat16*>(query), static_cast<const __nv_bfloat16*>(key), static_cast<const __nv_bfloat16*>(value), queries, keys, heads, query_stride, key_stride);
        }
    }
}
