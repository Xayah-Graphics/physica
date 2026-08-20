#include "device.cuh"
#include "kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::smoke::cuda_detail {
    namespace {
        __global__ void accumulate_kernel(const double* source, double* destination, const std::uint64_t count) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += source[index];
        }
    } // namespace

    void accumulate(const ::cuda::stream_ref stream, const double* source, double* destination, const std::uint64_t count) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(count), accumulate_kernel, source, destination, count);
    }
} // namespace physica::fluids::gas::smoke::cuda_detail
