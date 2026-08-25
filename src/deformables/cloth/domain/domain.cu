#include "domain-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::cuda_detail {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void accumulate_kernel(const ConstFieldView<double> source, const FieldView<double> destination, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            destination.x[index] += source.x[index];
            destination.y[index] += source.y[index];
            destination.z[index] += source.z[index];
        }
    } // namespace

    void accumulate(const ::cuda::stream_ref stream, const ConstFieldView<double> source, const FieldView<double> destination, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(count), accumulate_kernel, source, destination, count);
    }
} // namespace physica::deformables::cloth::cuda_detail
