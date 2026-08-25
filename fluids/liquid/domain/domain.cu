#include "domain-kernels.h"
#include <cuda/launch>

namespace physica::fluids::liquid::cuda_detail {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void accumulate_vector_kernel(const ConstVectorView<double> source, const VectorView<double> destination, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            destination.x[index] += source.x[index];
            destination.y[index] += source.y[index];
            destination.z[index] += source.z[index];
        }

        __global__ void accumulate_scalar_kernel(const double* source, double* destination, const std::size_t count) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            destination[index] += source[index];
        }
    } // namespace

    void accumulate(const ::cuda::stream_ref stream, const double* source, double* destination, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(count), accumulate_scalar_kernel, source, destination, count);
    }

    void accumulate(const ::cuda::stream_ref stream, const ConstVectorView<double> source, const VectorView<double> destination, const std::size_t count) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(count), accumulate_vector_kernel, source, destination, count);
    }
} // namespace physica::fluids::liquid::cuda_detail
