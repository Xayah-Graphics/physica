#include "inference-kernels.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <physica/cuda.h>
#include <cuda/launch>
#include <algorithm>

namespace physica::neural::kernels {

    template<class Function> void dispatch(const int scalar, Function function) {
        if (scalar == 0) function.template operator()<float>();
        else if (scalar == 1) function.template operator()<__half>();
        else function.template operator()<__nv_bfloat16>();
    }

    template<class Output, class Input> __global__ void convert_kernel(Output* output, const Input* input, const std::size_t count) {
        const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) output[i] = Output(float(input[i]));
    }

    template<class O, class I> __global__ void layout_kernel(O* output, const I* input, const int spatial, const int channels, const std::size_t count, const bool to_nhwc) {
        const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const std::size_t nchw = (i / (spatial * channels) * channels + i % channels) * spatial + i / channels % spatial;
        if (to_nhwc) output[i] = O(float(input[nchw]));
        else output[nchw] = O(float(input[i]));
    }

    template<class T> __global__ void activation_kernel(T* output, const T* input, const std::size_t count, const int kind) {
        const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const float x = float(input[i]);
        output[i] = T(kind == 0 ? x / (1.0F + expf(-x)) : kind == 1 ? x / (1.0F + expf(-1.702F * x)) : 0.5F * x * (1.0F + erff(x * 0.7071067811865475F)));
    }

    struct Moments { float mean; float variance; float count; };

    __device__ Moments combine(const Moments a, const Moments b) {
        const float count = a.count + b.count;
        const float delta = b.mean - a.mean;
        const float ratio = count > 0.0F ? b.count / count : 0.0F;
        return {a.mean + delta * ratio, a.variance + b.variance + delta * delta * a.count * ratio, count};
    }

    __device__ Moments reduce(Moments value, Moments* shared) {
        for (int offset = 16; offset; offset >>= 1) value = combine(value, {__shfl_down_sync(0xffffffff, value.mean, offset), __shfl_down_sync(0xffffffff, value.variance, offset), __shfl_down_sync(0xffffffff, value.count, offset)});
        if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = value;
        __syncthreads();
        value = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : Moments{};
        if (threadIdx.x < 32) for (int offset = 16; offset; offset >>= 1) value = combine(value, {__shfl_down_sync(0xffffffff, value.mean, offset), __shfl_down_sync(0xffffffff, value.variance, offset), __shfl_down_sync(0xffffffff, value.count, offset)});
        if (threadIdx.x == 0) shared[0] = value;
        __syncthreads();
        return shared[0];
    }

    template<class T, int Width, bool Residual = false> __global__ void layer_norm_kernel(T* output, const T* input, T* residual, const T* weight, const T* bias, const float epsilon) {
        constexpr int width = Width;
        __shared__ float partial[8];
        float values[(Width + 255) / 256];
        float sum = 0.0F;
        #pragma unroll
        for (int j = 0; j < (width + 255) / 256; ++j) {
            const int c = j * 256 + threadIdx.x;
            const int i = blockIdx.x * width + c;
            values[j] = c < width ? float(input[i]) : 0.0F;
            if constexpr (Residual) if (c < width) values[j] += float(residual[i]);
            sum += values[j];
        }
        for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0F;
        for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) partial[0] = sum / float(width);
        __syncthreads();
        const float mean = partial[0];
        float squared = 0.0F;
        #pragma unroll
        for (int j = 0; j < (width + 255) / 256; ++j) if (j * 256 + threadIdx.x < width) squared = fmaf(values[j] - mean, values[j] - mean, squared);
        __syncthreads();
        for (int offset = 16; offset; offset >>= 1) squared += __shfl_down_sync(0xffffffff, squared, offset);
        if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = squared;
        __syncthreads();
        squared = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0F;
        for (int offset = 16; offset; offset >>= 1) squared += __shfl_down_sync(0xffffffff, squared, offset);
        if (threadIdx.x == 0) partial[0] = rsqrtf(squared / float(width) + epsilon);
        __syncthreads();
        const float inverse = partial[0];
        #pragma unroll
        for (int j = 0; j < (width + 255) / 256; ++j) {
            const int c = j * 256 + threadIdx.x;
            if (c >= width) continue;
            const int i = blockIdx.x * width + c;
            if constexpr (Residual) residual[i] = T(values[j]);
            output[i] = T(fmaf((values[j] - mean) * inverse, float(weight[c]), float(bias[c])));
        }
    }

    template<class T, int GroupWidth> __global__ void group_moments_kernel(float* statistics, const T* input, const T* second, T* joined, const T* time, const int* step, const int spatial, const int first_width, const int chunks) {
        const int lane = threadIdx.x % 8;
        const int group = threadIdx.x / 8;
        constexpr int Columns = (GroupWidth + 7) / 8;
        constexpr int Width = GroupWidth * 32;
        Moments total{};
        for (int row = blockIdx.x * 8; row < spatial; row += chunks * 8) {
            float values[8 * Columns];
            float sum = 0.0F;
            float count = 0.0F;
            #pragma unroll
            for (int j = 0; j < 8 * Columns; ++j) {
                const int r = row + j / Columns;
                const int c = lane + (j % Columns) * 8;
                const int channel = group * GroupWidth + c;
                float value = 0.0F;
                if (r < spatial && c < GroupWidth) {
                    const int pixel = blockIdx.y * spatial + r;
                    value = second ? (channel < first_width ? float(input[pixel * first_width + channel]) : float(second[pixel * (Width - first_width) + channel - first_width])) : float(input[pixel * Width + channel]);
                    if (second) joined[pixel * Width + channel] = T(value);
                    if (time) value += float(time[(*step * gridDim.y + blockIdx.y) * Width + channel]);
                    sum += value;
                    count += 1.0F;
                }
                values[j] = value;
            }
            const float mean = count ? sum / count : 0.0F;
            float variance = 0.0F;
            #pragma unroll
            for (int j = 0; j < 8 * Columns; ++j) if (row + j / Columns < spatial && lane + (j % Columns) * 8 < GroupWidth) variance = fmaf(values[j] - mean, values[j] - mean, variance);
            total = combine(total, {mean, variance, count});
        }
        for (int offset = 4; offset; offset >>= 1) total = combine(total, {__shfl_down_sync(0xffffffff, total.mean, offset, 8), __shfl_down_sync(0xffffffff, total.variance, offset, 8), __shfl_down_sync(0xffffffff, total.count, offset, 8)});
        if (lane == 0) reinterpret_cast<Moments*>(statistics)[(blockIdx.y * 32 + group) * chunks + blockIdx.x] = total;
    }

    __global__ void group_reduce_kernel(float* statistics, const int chunks, const float epsilon) {
        __shared__ Moments shared[8];
        Moments moments{};
        for (int i = threadIdx.x; i < chunks; i += blockDim.x) moments = combine(moments, reinterpret_cast<Moments*>(statistics)[blockIdx.x * chunks + i]);
        moments = reduce(moments, shared);
        if (threadIdx.x == 0) reinterpret_cast<Moments*>(statistics)[blockIdx.x * chunks] = {moments.mean, rsqrtf(moments.variance / moments.count + epsilon), moments.count};
    }

    template<class T> __global__ void group_output_kernel(T* output, const T* input, const T* weight, const T* bias, const float* statistics, const int count, const int spatial, const int width, const int chunks, const bool silu, const T* time, const int* step, const int batch) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int c = i % width;
        const Moments moments = reinterpret_cast<const Moments*>(statistics)[(i / (spatial * width) * 32 + c / (width / 32)) * chunks];
        const float scale = moments.variance * float(weight[c]);
        const float shift = float(bias[c]) - moments.mean * scale;
        const float x = float(input[i]) + (time ? float(time[(*step * batch + i / (spatial * width)) * width + c]) : 0.0F);
        const float value = fmaf(x, scale, shift);
        output[i] = T(silu ? value / (1.0F + expf(-value)) : value);
    }

    template<class T> __global__ void resize_kernel(T* output, const T* input, const int height, const int width, const int channels, const int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        const int pixel = i / channels;
        output[i] = input[((pixel / (height * width * 4) * height + pixel / (width * 2) % (height * 2) / 2) * width + pixel % (width * 2) / 2) * channels + i % channels];
    }

    void convert(const ::cuda::stream_ref stream, void* output, const void* input, const std::size_t count, const int source, const int destination) {
        dispatch(destination, [&]<class O>() { dispatch(source, [&]<class I>() { ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), convert_kernel<O, I>, static_cast<O*>(output), static_cast<const I*>(input), count); }); });
    }
    void layout(const ::cuda::stream_ref stream, void* output, const void* input, const int batch, const int height, const int width, const int channels, const int scalar, const bool to_nhwc) {
        const std::size_t count = std::size_t(batch) * height * width * channels;
        dispatch(scalar, [&]<class T>() { ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), layout_kernel<T, T>, static_cast<T*>(output), static_cast<const T*>(input), height * width, channels, count, to_nhwc); });
    }
    void convert_layout(const ::cuda::stream_ref stream, void* output, const void* input, const int batch, const int spatial, const int channels, const int source, const int destination) {
        const std::size_t count = std::size_t(batch) * spatial * channels;
        dispatch(destination, [&]<class O>() { dispatch(source, [&]<class I>() {
            ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), layout_kernel<O, I>, static_cast<O*>(output), static_cast<const I*>(input), spatial, channels, count, true);
        }); });
    }

    void activation(const ::cuda::stream_ref stream, void* output, const void* input, const std::size_t count, const int scalar, const int kind) {
        dispatch(scalar, [&]<class T>() { ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), activation_kernel<T>, static_cast<T*>(output), static_cast<const T*>(input), count, kind); });
    }
    template<class T, bool Residual>
    void launch_norm(const ::cuda::stream_ref stream, T* output, const T* input, T* residual, const T* weight, const T* bias, const int rows, const int width, const float epsilon) {
        const auto launch = [&]<int Width>() {
            ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(rows), ::cuda::block_dims(256))), layer_norm_kernel<T, Width, Residual>, output, input, residual, weight, bias, epsilon);
        };
        switch (width) {
            case 320: launch.template operator()<320>(); break;
            case 640: launch.template operator()<640>(); break;
            case 768: launch.template operator()<768>(); break;
            case 1280: launch.template operator()<1280>(); break;
        }
    }

    void layer_norm(const ::cuda::stream_ref stream, void* output, const void* input, const void* weight, const void* bias, const int rows, const int width, const float epsilon, const int scalar) {
        dispatch(scalar, [&]<class T>() { launch_norm<T, false>(stream, static_cast<T*>(output), static_cast<const T*>(input), nullptr, static_cast<const T*>(weight), static_cast<const T*>(bias), rows, width, epsilon); });
    }

    void residual_norm(const ::cuda::stream_ref stream, void* output, void* residual, const void* input, const void* weight, const void* bias, const int rows, const int width, const float epsilon) {
        launch_norm<__half, true>(stream, static_cast<__half*>(output), static_cast<const __half*>(input), static_cast<__half*>(residual), static_cast<const __half*>(weight), static_cast<const __half*>(bias), rows, width, epsilon);
    }

    void group_moments(const ::cuda::stream_ref stream, float* statistics, const void* input, const void* second, void* joined, const void* time, const int* step, const int batch, const int spatial, const int width, const int first_width, const int scalar) {
        const int chunks = std::min((spatial + 7) / 8, 1024);
        dispatch(scalar, [&]<class T>() {
            const auto launch = [&]<int GroupWidth>() {
                ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(dim3(chunks, batch)), ::cuda::block_dims(256))), group_moments_kernel<T, GroupWidth>, statistics, static_cast<const T*>(input), static_cast<const T*>(second), static_cast<T*>(joined), static_cast<const T*>(time), step, spatial, first_width, chunks);
            };
            switch (width / 32) {
                case 4: launch.template operator()<4>(); break;
                case 8: launch.template operator()<8>(); break;
                case 10: launch.template operator()<10>(); break;
                case 16: launch.template operator()<16>(); break;
                case 20: launch.template operator()<20>(); break;
                case 30: launch.template operator()<30>(); break;
                case 40: launch.template operator()<40>(); break;
                case 60: launch.template operator()<60>(); break;
                case 80: launch.template operator()<80>(); break;
            }
        });
    }

    void group_norm(const ::cuda::stream_ref stream, void* output, const void* input, const void* weight, const void* bias, float* statistics, const int batch, const int spatial, const int width, const float epsilon, const int scalar, const bool silu, const bool prepared, const void* time, const int* step) {
        const int chunks = std::min((spatial + 7) / 8, 1024);
        if (!prepared) group_moments(stream, statistics, input, nullptr, nullptr, time, step, batch, spatial, width, width, scalar);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(batch * 32), ::cuda::block_dims(256))), group_reduce_kernel, statistics, chunks, epsilon);
        dispatch(scalar, [&]<class T>() {
            ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((batch * spatial * width + 255) / 256), ::cuda::block_dims(256))), group_output_kernel<T>, static_cast<T*>(output), static_cast<const T*>(input), static_cast<const T*>(weight), static_cast<const T*>(bias), statistics, batch * spatial * width, spatial, width, chunks, silu, static_cast<const T*>(time), step, batch);
        });
    }

    void resize(const ::cuda::stream_ref stream, void* output, const void* input, const int batch, const int height, const int width, const int channels, const int scalar) {
        const int count = batch * height * width * channels * 4;
        dispatch(scalar, [&]<class T>() { ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims((count + 255) / 256), ::cuda::block_dims(256))), resize_kernel<T>, static_cast<T*>(output), static_cast<const T*>(input), height, width, channels, count); });
    }

}
