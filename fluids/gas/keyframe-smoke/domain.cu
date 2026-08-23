#include "device.cuh"
#include "domain_kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    namespace {
        __global__ void accumulate_kernel(const double* source, double* destination, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += source[index];
        }
    } // namespace

    void accumulate(const ::cuda::stream_ref stream, const double* source, double* destination, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(count), accumulate_kernel, source, destination, count);
    }
} // namespace physica::fluids::gas::keyframe_smoke::cuda_detail
