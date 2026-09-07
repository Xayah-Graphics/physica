#ifndef PHYSICA_GENERATIVE_SDXL_KERNELS_H
#define PHYSICA_GENERATIVE_SDXL_KERNELS_H

#include <cstdint>
#include <cuda_runtime_api.h>
#include <physica/cuda_stream.h>

namespace physica::generative::sdxl::kernels {
    struct SamplingStep {
        float sigma;
        float delta;
        float inverse;
    };
    void embedding(const ::cuda::stream_ref stream, void* output, const void* token, const void* position, const std::int32_t* ids, const int rows, const int width);
    void gather_rows(const ::cuda::stream_ref stream, void* output, const void* input, const std::int32_t* rows, const int count, const int width);
    void weighted_text(const ::cuda::stream_ref stream, void* output, const void* input, const void* empty, const float* weights, const int rows, const int width);
    void conditioning(const ::cuda::stream_ref stream, void* output, const void* clip_l, const void* clip_g, const int positive, const int negative, const int length);
    void time_embedding(const ::cuda::stream_ref stream, void* output, const float* times, const int count, const int width, const int scalar);
    void conditions(const ::cuda::stream_ref stream, void* output, const void* pooled, const void* geometry);
    void time_condition(const ::cuda::stream_ref stream, void* output, const void* times, const void* labels, const int steps);
    void training_sigmas(const ::cuda::stream_ref stream, float* output);
    void prepare_schedule(const ::cuda::stream_ref stream, SamplingStep* output, float* times, const float* training, const int steps);
    void initialize(const ::cuda::stream_ref stream, float* state, void* input, int* step, const std::uint64_t* seed, const SamplingStep* schedule, const int count);
    void euler(const ::cuda::stream_ref stream, float* state, void* input, const void* epsilon, const SamplingStep* schedule, const int* step, const float cfg, const int count);
    void advance(const ::cuda::stream_ref stream, int* step, const int count, const cudaGraphConditionalHandle handle);
    void latent_decode(const ::cuda::stream_ref stream, void* output, const float* latent, const int count);
    void pixels(const ::cuda::stream_ref stream, std::uint8_t* output, const void* input, const int count);
} // namespace physica::generative::sdxl::kernels

#endif
