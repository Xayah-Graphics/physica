#include "sampling.cuh"
#include "solver_kernels.h"
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    namespace {
        __global__ void copy_float_kernel(const float* source, float* destination, const std::uint64_t count) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] = source[index];
        }

        __global__ void copy_double_kernel(const double* source, double* destination, const std::uint64_t count) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] = source[index];
        }

        __global__ void accumulate_double_kernel(const double* source, double* destination, const std::uint64_t count) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += source[index];
        }

        __global__ void diffusion_iteration_kernel(const Grid grid, const int axis, const float alpha, const float* source, const float* previous, const VelocityBoundaryData boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            float neighbors = load_face(previous, axis, x - 1, y, z, grid, boundary) + load_face(previous, axis, x + 1, y, z, grid, boundary) + load_face(previous, axis, x, y - 1, z, grid, boundary) + load_face(previous, axis, x, y + 1, z, grid, boundary);
            if (grid.dimensions == 3u) neighbors += load_face(previous, axis, x, y, z - 1, grid, boundary) + load_face(previous, axis, x, y, z + 1, grid, boundary);
            output[index] = (source[index] + alpha * neighbors) / (1.0F + 2.0F * grid.dimensions * alpha);
        }

        __device__ void scatter_diffusion_neighbor(double* previous_adjoint, const int axis, const int x, const int y, const int z, const Grid grid, const VelocityBoundaryData boundary, const double value) {
            const int coordinates[3]{x, y, z};
            for (int dimension = 0; dimension < 3; ++dimension) {
                const int size = extent(grid, axis, dimension);
                if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
                const int face = 2 * dimension + (coordinates[dimension] >= size);
                if (boundary.modes[face] == 0u || (boundary.modes[face] == 2u && axis == dimension)) return;
            }
            atomicAdd(previous_adjoint + mapped_face_index(x, y, z, grid, axis, boundary), value);
        }

        __global__ void diffusion_iteration_reverse_kernel(const Grid grid, const int axis, const float alpha, const double* output_adjoint, const VelocityBoundaryData boundary, double* previous_adjoint, double* source_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            const double value = output_adjoint[index] / (1.0 + 2.0 * grid.dimensions * alpha);
            source_adjoint[index] += value;
            const double neighbor_value = alpha * value;
            scatter_diffusion_neighbor(previous_adjoint, axis, x - 1, y, z, grid, boundary, neighbor_value);
            scatter_diffusion_neighbor(previous_adjoint, axis, x + 1, y, z, grid, boundary, neighbor_value);
            scatter_diffusion_neighbor(previous_adjoint, axis, x, y - 1, z, grid, boundary, neighbor_value);
            scatter_diffusion_neighbor(previous_adjoint, axis, x, y + 1, z, grid, boundary, neighbor_value);
            if (grid.dimensions == 3u) {
                scatter_diffusion_neighbor(previous_adjoint, axis, x, y, z - 1, grid, boundary, neighbor_value);
                scatter_diffusion_neighbor(previous_adjoint, axis, x, y, z + 1, grid, boundary, neighbor_value);
            }
        }
    } // namespace

    void diffusion_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t iterations, const float viscosity, const VelocityBoundaryData boundary, const ConstStaggeredVectorView source, const StaggeredVectorView first, const StaggeredVectorView second, const StaggeredVectorView output) {
        const float alpha = viscosity * grid.time_step / (grid.cell_size * grid.cell_size);
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint64_t count = face_count(grid, axis);
            ::cuda::launch(stream, ::cuda::distribute<block_size>(count), copy_float_kernel, component(source, axis), component(first, axis), count);
            for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
                const float* previous = iteration % 2u == 0u ? component(first, axis) : component(second, axis);
                float* next = iteration % 2u == 0u ? component(second, axis) : component(first, axis);
                ::cuda::launch(stream, ::cuda::distribute<block_size>(count), diffusion_iteration_kernel, grid, axis, alpha, component(source, axis), previous, boundary, next);
            }
            const float* result = iterations % 2u == 0u ? component(first, axis) : component(second, axis);
            ::cuda::launch(stream, ::cuda::distribute<block_size>(count), copy_float_kernel, result, component(output, axis), count);
        }
    }

    void diffusion_vjp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t iterations, const float viscosity, const VelocityBoundaryData boundary, const ConstStaggeredVectorAdjointView output_adjoint, const StaggeredVectorAdjointView first, const StaggeredVectorAdjointView second, const StaggeredVectorAdjointView source_adjoint) {
        const float alpha = viscosity * grid.time_step / (grid.cell_size * grid.cell_size);
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint64_t count = face_count(grid, axis);
            ::cuda::fill_bytes(stream, ::cuda::std::span{component(first, axis), static_cast<std::size_t>(count)}, 0u);
            ::cuda::fill_bytes(stream, ::cuda::std::span{component(second, axis), static_cast<std::size_t>(count)}, 0u);
            double* result_adjoint = iterations % 2u == 0u ? component(first, axis) : component(second, axis);
            ::cuda::launch(stream, ::cuda::distribute<block_size>(count), copy_double_kernel, component(output_adjoint, axis), result_adjoint, count);
            for (std::uint32_t reverse = 0u; reverse < iterations; ++reverse) {
                const std::uint32_t iteration = iterations - 1u - reverse;
                double* current = iteration % 2u == 0u ? component(second, axis) : component(first, axis);
                double* previous = iteration % 2u == 0u ? component(first, axis) : component(second, axis);
                ::cuda::fill_bytes(stream, ::cuda::std::span{previous, static_cast<std::size_t>(count)}, 0u);
                ::cuda::launch(stream, ::cuda::distribute<block_size>(count), diffusion_iteration_reverse_kernel, grid, axis, alpha, current, boundary, previous, component(source_adjoint, axis));
            }
            ::cuda::launch(stream, ::cuda::distribute<block_size>(count), accumulate_double_kernel, component(first, axis), component(source_adjoint, axis), count);
        }
    }
} // namespace physica::fluids::gas::keyframe_smoke::cuda_detail
