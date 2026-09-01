#include "module_kernels.h"
#include <cuda/launch>

namespace physica::examples::cloth_flag::module_cuda {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void write_vectors_kernel(const std::uint32_t count, const float* x, const float* y, const float* z, spectra::sdk::Float3* output) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            output[index] = {x[index], y[index], z[index]};
        }

        __global__ void write_strain_kernel(const std::uint32_t edge_count, const std::uint32_t* first, const std::uint32_t* second, const float* rest_lengths, const float* position_x, const float* position_y, const float* position_z, float* strain) {
            const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
            if (edge >= edge_count) return;
            const std::uint32_t a = first[edge];
            const std::uint32_t b = second[edge];
            const float x = position_x[b] - position_x[a];
            const float y = position_y[b] - position_y[a];
            const float z = position_z[b] - position_z[a];
            const float value = fabsf(sqrtf(x * x + y * y + z * z) / rest_lengths[edge] - 1.0F);
            atomicMax(reinterpret_cast<unsigned int*>(strain + a), __float_as_uint(value));
            atomicMax(reinterpret_cast<unsigned int*>(strain + b), __float_as_uint(value));
        }
    } // namespace

    void write_vectors(const ::cuda::stream_ref stream, const std::uint32_t count, const float* x, const float* y, const float* z, spectra::sdk::Float3* output) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(count), write_vectors_kernel, count, x, y, z, output);
    }

    void write_strain(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t edge_count, const std::uint32_t* first, const std::uint32_t* second, const float* rest_lengths, const float* position_x, const float* position_y, const float* position_z, float* strain) {
        cudaMemsetAsync(strain, 0, static_cast<std::size_t>(particle_count) * sizeof(float), stream.get());
        ::cuda::launch(stream, ::cuda::distribute<block_size>(edge_count), write_strain_kernel, edge_count, first, second, rest_lengths, position_x, position_y, position_z, strain);
    }
} // namespace physica::examples::cloth_flag::module_cuda
