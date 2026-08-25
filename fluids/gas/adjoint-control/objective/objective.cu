#include "../domain/device.cuh"
#include "kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::adjoint_control::cuda_detail {
    namespace {
        __global__ void copy_float_kernel(const float* source, float* destination, const std::uint64_t count) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] = source[index];
        }

        __global__ void inject_copy_kernel(const float* source, const double scale, double* destination, const std::uint64_t count) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += scale * source[index];
        }

        __global__ void residual_kernel(const float* state, const float* target, const std::uint64_t count, float* residual) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) residual[index] = state[index] - target[index];
        }

        __global__ void blur_axis_kernel(const float* source, const int nx, const int ny, const int nz, const int dimension, const std::uint32_t radius, const float* weights, float* output) {
            const std::uint64_t count = static_cast<std::uint64_t>(nx) * ny * nz;
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            int x, y, z;
            decode(index, nx, ny, x, y, z);
            float sum = 0.0F;
            for (int offset = -static_cast<int>(radius); offset <= static_cast<int>(radius); ++offset) {
                const int sx = x + (dimension == 0 ? offset : 0);
                const int sy = y + (dimension == 1 ? offset : 0);
                const int sz = z + (dimension == 2 ? offset : 0);
                if (sx >= 0 && sx < nx && sy >= 0 && sy < ny && sz >= 0 && sz < nz) sum += weights[offset + radius] * source[index3(sx, sy, sz, nx, ny)];
            }
            output[index] = sum;
        }

        __global__ void blur_axis_inject_kernel(const float* source, const int nx, const int ny, const int nz, const int dimension, const std::uint32_t radius, const float* weights, const double scale, double* output) {
            const std::uint64_t count = static_cast<std::uint64_t>(nx) * ny * nz;
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            int x, y, z;
            decode(index, nx, ny, x, y, z);
            double sum = 0.0;
            for (int offset = -static_cast<int>(radius); offset <= static_cast<int>(radius); ++offset) {
                const int sx = x + (dimension == 0 ? offset : 0);
                const int sy = y + (dimension == 1 ? offset : 0);
                const int sz = z + (dimension == 2 ? offset : 0);
                if (sx >= 0 && sx < nx && sy >= 0 && sy < ny && sz >= 0 && sz < nz) sum += weights[offset + radius] * source[index3(sx, sy, sz, nx, ny)];
            }
            output[index] += scale * sum;
        }

        __global__ void squared_scalar_kernel(const float* field, const std::uint64_t count, const double weight, double* result) {
            __shared__ double sums[block_size];
            double sum = 0.0;
            for (std::uint64_t index = threadIdx.x; index < count; index += blockDim.x) sum += field[index] * field[index];
            sums[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned offset = blockDim.x / 2u; offset > 0u; offset /= 2u) {
                if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = weight * sums[0];
        }

        __global__ void squared_velocity_kernel(const Grid grid, const ConstStaggeredVectorView field, const double weight, double* result) {
            __shared__ double sums[block_size];
            double sum = 0.0;
            for (int axis = 0; axis < grid.dimensions; ++axis) for (std::uint64_t index = threadIdx.x; index < face_count(grid, axis); index += blockDim.x) sum += component(field, axis)[index] * component(field, axis)[index];
            sums[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned offset = blockDim.x / 2u; offset > 0u; offset /= 2u) {
                if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = weight * sums[0];
        }

        __global__ void directional_kernel(const Grid grid, const double density_weight, const double velocity_weight, const float* density_residual, const float* density_tangent, const ConstStaggeredVectorView velocity_residual, const ConstStaggeredVectorView velocity_tangent, double* result) {
            __shared__ double sums[block_size];
            double sum = 0.0;
            for (std::uint64_t index = threadIdx.x; index < cell_count(grid); index += blockDim.x) sum += density_weight * density_residual[index] * density_tangent[index];
            for (int axis = 0; axis < grid.dimensions; ++axis) for (std::uint64_t index = threadIdx.x; index < face_count(grid, axis); index += blockDim.x) sum += velocity_weight * component(velocity_residual, axis)[index] * component(velocity_tangent, axis)[index];
            sums[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned offset = blockDim.x / 2u; offset > 0u; offset /= 2u) {
                if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = 2.0 * sums[0];
        }

        __global__ void control_effort_kernel(const Grid grid, const ConstCenteredVectorView control, const double weight, double* result) {
            __shared__ double sums[block_size];
            double sum = 0.0;
            for (std::uint64_t index = threadIdx.x; index < cell_count(grid); index += blockDim.x) {
                sum += control.x[index] * control.x[index] + control.y[index] * control.y[index];
                if (grid.dimensions == 3u) sum += control.z[index] * control.z[index];
            }
            sums[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned offset = blockDim.x / 2u; offset > 0u; offset /= 2u) {
                if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = weight * grid.time_step * grid.time_step * sums[0];
        }

        __global__ void control_effort_jvp_kernel(const Grid grid, const ConstCenteredVectorView control, const ConstCenteredVectorView tangent, const double weight, double* result) {
            __shared__ double sums[block_size];
            double sum = 0.0;
            for (std::uint64_t index = threadIdx.x; index < cell_count(grid); index += blockDim.x) {
                sum += control.x[index] * tangent.x[index] + control.y[index] * tangent.y[index];
                if (grid.dimensions == 3u) sum += control.z[index] * tangent.z[index];
            }
            sums[threadIdx.x] = sum;
            __syncthreads();
            for (unsigned offset = blockDim.x / 2u; offset > 0u; offset /= 2u) {
                if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = 2.0 * weight * grid.time_step * grid.time_step * sums[0];
        }

        __global__ void control_effort_vjp_kernel(const Grid grid, const ConstCenteredVectorView control, const double scale, const CenteredVectorAdjointView control_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            control_adjoint.x[index] += scale * control.x[index];
            control_adjoint.y[index] += scale * control.y[index];
            if (grid.dimensions == 3u) control_adjoint.z[index] += scale * control.z[index];
        }

        void blur_field(const ::cuda::stream_ref stream, const int nx, const int ny, const int nz, const std::uint32_t radius, const float* weights, const float* source, float* first, float* second, float* output) {
            const std::uint64_t count = static_cast<std::uint64_t>(nx) * ny * nz;
            ::cuda::launch(stream, ::cuda::distribute<block_size>(count), blur_axis_kernel, source, nx, ny, nz, 0, radius, weights, first);
            ::cuda::launch(stream, ::cuda::distribute<block_size>(count), blur_axis_kernel, first, nx, ny, nz, 1, radius, weights, second);
            if (nz == 1) ::cuda::launch(stream, ::cuda::distribute<block_size>(count), copy_float_kernel, second, output, count);
            else ::cuda::launch(stream, ::cuda::distribute<block_size>(count), blur_axis_kernel, second, nx, ny, nz, 2, radius, weights, output);
        }

        void inject_field(const ::cuda::stream_ref stream, const int nx, const int ny, const int nz, const std::uint32_t radius, const float* weights, const double scale, const float* source, float* first, float* second, double* output) {
            const std::uint64_t count = static_cast<std::uint64_t>(nx) * ny * nz;
            ::cuda::launch(stream, ::cuda::distribute<block_size>(count), blur_axis_kernel, source, nx, ny, nz, 0, radius, weights, first);
            ::cuda::launch(stream, ::cuda::distribute<block_size>(count), blur_axis_kernel, first, nx, ny, nz, 1, radius, weights, second);
            if (nz == 1) ::cuda::launch(stream, ::cuda::distribute<block_size>(count), inject_copy_kernel, second, scale, output, count);
            else ::cuda::launch(stream, ::cuda::distribute<block_size>(count), blur_axis_inject_kernel, second, nx, ny, nz, 2, radius, weights, scale, output);
        }
    } // namespace

    void residual_forward(const ::cuda::stream_ref stream, const Grid grid, const ConstScalarView state, const ConstScalarView target, const ScalarView residual) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), residual_kernel, state.values, target.values, cell_count(grid), residual.values);
    }

    void residual_forward(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorView state, const ConstStaggeredVectorView target, const StaggeredVectorView residual) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), residual_kernel, component(state, axis), component(target, axis), face_count(grid, axis), component(residual, axis));
    }

    void blur_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t radius, const float* weights, const ConstScalarView source, const ScalarView first, const ScalarView second, const ScalarView output) {
        blur_field(stream, grid.nx, grid.ny, grid.nz, radius, weights, source.values, first.values, second.values, output.values);
    }

    void blur_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t radius, const float* weights, const ConstStaggeredVectorView source, const StaggeredVectorView first, const StaggeredVectorView second, const StaggeredVectorView output) {
        for (int axis = 0; axis < grid.dimensions; ++axis) blur_field(stream, extent(grid, axis, 0), extent(grid, axis, 1), extent(grid, axis, 2), radius, weights, component(source, axis), component(first, axis), component(second, axis), component(output, axis));
    }

    void squared_loss(const ::cuda::stream_ref stream, const Grid grid, const double weight, const ConstScalarView field, double* output) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), squared_scalar_kernel, field.values, cell_count(grid), weight, output);
    }

    void squared_loss(const ::cuda::stream_ref stream, const Grid grid, const double weight, const ConstStaggeredVectorView field, double* output) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), squared_velocity_kernel, grid, field, weight, output);
    }

    void directional_loss(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t radius, const float* weights, const double density_weight, const double velocity_weight, const ConstScalarView density_residual, const ConstStaggeredVectorView velocity_residual, const ConstScalarView density_tangent, const ConstStaggeredVectorView velocity_tangent, const ScalarView scalar_first, const ScalarView scalar_second, const ScalarView scalar_output, const StaggeredVectorView velocity_first, const StaggeredVectorView velocity_second, const StaggeredVectorView velocity_output, double* result) {
        blur_forward(stream, grid, radius, weights, density_tangent, scalar_first, scalar_second, scalar_output);
        blur_forward(stream, grid, radius, weights, velocity_tangent, velocity_first, velocity_second, velocity_output);
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), directional_kernel, grid, density_weight, velocity_weight, density_residual.values, scalar_output.values, velocity_residual, ConstStaggeredVectorView{velocity_output.x, velocity_output.y, velocity_output.z}, result);
    }

    void inject_keyframe_adjoint(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t radius, const float* weights, const double density_weight, const double velocity_weight, const ConstScalarView blurred_density_residual, const ConstStaggeredVectorView blurred_velocity_residual, const ScalarView scalar_first, const ScalarView scalar_second, const StaggeredVectorView velocity_first, const StaggeredVectorView velocity_second, const ScalarAdjointView density_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
        inject_field(stream, grid.nx, grid.ny, grid.nz, radius, weights, 2.0 * density_weight, blurred_density_residual.values, scalar_first.values, scalar_second.values, density_adjoint.values);
        for (int axis = 0; axis < grid.dimensions; ++axis) inject_field(stream, extent(grid, axis, 0), extent(grid, axis, 1), extent(grid, axis, 2), radius, weights, 2.0 * velocity_weight, component(blurred_velocity_residual, axis), component(velocity_first, axis), component(velocity_second, axis), component(velocity_adjoint, axis));
    }

    void control_effort(const ::cuda::stream_ref stream, const Grid grid, const double weight, const ConstCenteredVectorView control, double* result) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), control_effort_kernel, grid, control, weight, result);
    }

    void control_effort_jvp(const ::cuda::stream_ref stream, const Grid grid, const double weight, const ConstCenteredVectorView control, const ConstCenteredVectorView control_tangent, double* result) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), control_effort_jvp_kernel, grid, control, control_tangent, weight, result);
    }

    void control_effort_vjp(const ::cuda::stream_ref stream, const Grid grid, const double weight, const ConstCenteredVectorView control, const CenteredVectorAdjointView control_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), control_effort_vjp_kernel, grid, control, 2.0 * weight * grid.time_step * grid.time_step, control_adjoint);
    }
} // namespace physica::fluids::gas::adjoint_control::cuda_detail
