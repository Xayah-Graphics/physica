#include "../detail/cuda/device.cuh"
#include "conservation-kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::operators::cuda_backend {
    namespace {
        __global__ void reduce_sum_kernel(const float* values, const std::uint64_t count, double* result) {
            __shared__ double sums[detail::cuda::block_size];
            double sum = 0.0;
            for (std::uint64_t index = threadIdx.x; index < count; index += blockDim.x) sum += values[index];
            sums[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned offset = blockDim.x / 2u; offset > 0u; offset /= 2u) {
                if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = sums[0];
        }

        __global__ void reduce_dot_kernel(const double* first, const float* second, const std::uint64_t count, double* result) {
            __shared__ double sums[detail::cuda::block_size];
            double sum = 0.0;
            for (std::uint64_t index = threadIdx.x; index < count; index += blockDim.x) sum += first[index] * second[index];
            sums[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned offset = blockDim.x / 2u; offset > 0u; offset /= 2u) {
                if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = sums[0];
        }

        __global__ void normalize_mass_kernel(const float retention, const float* advected, const double* input_mass, const double* advected_mass, const std::uint64_t count, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) output[index] = static_cast<float>(retention * advected[index] * input_mass[0] / advected_mass[0]);
        }

        __global__ void normalize_mass_jvp_kernel(const float retention, const float* advected, const float* advected_tangent, const double* input_mass, const double* advected_mass, const double* input_mass_tangent, const double* advected_mass_tangent, const std::uint64_t count, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            const double scale         = input_mass[0] / advected_mass[0];
            const double scale_tangent = input_mass_tangent[0] / advected_mass[0] - input_mass[0] * advected_mass_tangent[0] / (advected_mass[0] * advected_mass[0]);
            output_tangent[index]      = static_cast<float>(retention * (scale * advected_tangent[index] + scale_tangent * advected[index]));
        }

        __global__ void normalize_mass_reverse_kernel(const float retention, const double* output_adjoint, const double* input_mass, const double* advected_mass, const double* density_dot, const std::uint64_t count, double* input_adjoint, double* advected_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            input_adjoint[index] += retention * density_dot[0] / advected_mass[0];
            advected_adjoint[index] += retention * (input_mass[0] * output_adjoint[index] / advected_mass[0] - density_dot[0] * input_mass[0] / (advected_mass[0] * advected_mass[0]));
        }
    } // namespace

    void mass_forward(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const float retention, const detail::cuda::ScalarView<const float> input, const detail::cuda::ScalarView<const float> advected, double* input_mass, double* advected_mass, const detail::cuda::ScalarView<float> output) {
        const std::uint64_t count = detail::cuda::cell_count(grid);
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::block_size), reduce_sum_kernel, input.values, count, input_mass);
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::block_size), reduce_sum_kernel, advected.values, count, advected_mass);
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(count), normalize_mass_kernel, retention, advected.values, input_mass, advected_mass, count, output.values);
    }

    void mass_jvp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const float retention, const detail::cuda::ScalarView<const float>, const detail::cuda::ScalarView<const float> advected, const detail::cuda::ScalarView<const float> input_tangent, const detail::cuda::ScalarView<const float> advected_tangent, const double* input_mass, const double* advected_mass, double* input_mass_tangent, double* advected_mass_tangent, const detail::cuda::ScalarView<float> output_tangent) {
        const std::uint64_t count = detail::cuda::cell_count(grid);
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::block_size), reduce_sum_kernel, input_tangent.values, count, input_mass_tangent);
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::block_size), reduce_sum_kernel, advected_tangent.values, count, advected_mass_tangent);
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(count), normalize_mass_jvp_kernel, retention, advected.values, advected_tangent.values, input_mass, advected_mass, input_mass_tangent, advected_mass_tangent, count, output_tangent.values);
    }

    void mass_vjp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const float retention, const detail::cuda::ScalarView<const float> advected, const double* input_mass, const double* advected_mass, const detail::cuda::ScalarView<const double> output_adjoint, double* density_dot, const detail::cuda::ScalarView<double> input_adjoint, const detail::cuda::ScalarView<double> advected_adjoint) {
        const std::uint64_t count = detail::cuda::cell_count(grid);
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::block_size), reduce_dot_kernel, output_adjoint.values, advected.values, count, density_dot);
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(count), normalize_mass_reverse_kernel, retention, output_adjoint.values, input_mass, advected_mass, density_dot, count, input_adjoint.values, advected_adjoint.values);
    }
} // namespace physica::fluids::gas::operators::cuda_backend
