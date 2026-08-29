#include <physica/fluids/gas/device.cuh>
#include "diffusion-kernels.h"
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>

namespace physica::fluids::gas::operators::kernels {
    namespace {
        __global__ void accumulate_double_kernel(const double* source, double* destination, const std::uint64_t count) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += source[index];
        }

        __device__ bool collider_face(const device::Discretization grid, const int axis, const int x, const int y, const int z, const std::uint32_t* collider_ids) {
            const int first[3]{x - (axis == 0), y - (axis == 1), z - (axis == 2)};
            const int second[3]{x, y, z};
            const bool first_inside  = first[0] >= 0 && first[0] < grid.grid.nx && first[1] >= 0 && first[1] < grid.grid.ny && first[2] >= 0 && first[2] < grid.grid.nz;
            const bool second_inside = second[0] >= 0 && second[0] < grid.grid.nx && second[1] >= 0 && second[1] < grid.grid.ny && second[2] >= 0 && second[2] < grid.grid.nz;
            return first_inside && collider_ids[fluids::grid::device::index3(first[0], first[1], first[2], grid.grid.nx, grid.grid.ny)] != 0u || second_inside && collider_ids[fluids::grid::device::index3(second[0], second[1], second[2], grid.grid.nx, grid.grid.ny)] != 0u;
        }

        __global__ void diffusion_iteration_kernel(const device::Discretization grid, const int axis, const float alpha, const std::uint32_t* collider_ids, const float* source, const float* previous, const device::VelocityBoundary boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            if (collider_face(grid, axis, x, y, z, collider_ids)) {
                output[index] = source[index];
                return;
            }
            float neighbors = device::load_face(previous, axis, x - 1, y, z, grid, boundary) + device::load_face(previous, axis, x + 1, y, z, grid, boundary) + device::load_face(previous, axis, x, y - 1, z, grid, boundary) + device::load_face(previous, axis, x, y + 1, z, grid, boundary);
            if (grid.dimensions == 3u) neighbors += device::load_face(previous, axis, x, y, z - 1, grid, boundary) + device::load_face(previous, axis, x, y, z + 1, grid, boundary);
            output[index] = (source[index] + alpha * neighbors) / (1.0F + 2.0F * grid.dimensions * alpha);
        }

        __device__ void scatter_diffusion_neighbor(double* previous_adjoint, const int axis, const int x, const int y, const int z, const device::Discretization grid, const device::VelocityBoundary boundary, const double value) {
            const int coordinates[3]{x, y, z};
            for (int dimension = 0; dimension < 3; ++dimension) {
                const int size = fluids::grid::device::extent(grid.grid, axis, dimension);
                if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
                const int face = 2 * dimension + (coordinates[dimension] >= size);
                if (boundary.faces[face].mode == 0u || (boundary.faces[face].mode == 2u && axis == dimension)) return;
            }
            atomicAdd(previous_adjoint + device::mapped_face_index(x, y, z, grid, axis, boundary), value);
        }

        __global__ void diffusion_iteration_reverse_kernel(const device::Discretization grid, const int axis, const float alpha, const std::uint32_t* collider_ids, const double* output_adjoint, const device::VelocityBoundary boundary, double* previous_adjoint, double* source_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            if (collider_face(grid, axis, x, y, z, collider_ids)) {
                source_adjoint[index] += output_adjoint[index];
                return;
            }
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

    void identity_velocity_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const field::VectorView<const double> output_adjoint, const field::VectorView<double> source_adjoint) {
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint64_t count = fluids::grid::device::face_count(grid.grid, axis);
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(count), accumulate_double_kernel, fluids::grid::device::component(output_adjoint, axis), fluids::grid::device::component(source_adjoint, axis), count);
        }
    }

    void diffusion_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t iterations, const float viscosity, const std::uint32_t* collider_ids, const device::VelocityBoundary boundary, const field::VectorView<const float> source, const field::VectorView<float> first, const field::VectorView<float> second, const field::VectorView<float> output) {
        const float alpha = viscosity * grid.time_step / (grid.grid.cell_size * grid.grid.cell_size);
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint64_t count = fluids::grid::device::face_count(grid.grid, axis);
            ::cuda::copy_bytes(stream, ::cuda::std::span{fluids::grid::device::component(source, axis), static_cast<std::size_t>(count)}, ::cuda::std::span{fluids::grid::device::component(first, axis), static_cast<std::size_t>(count)});
            for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
                const float* previous = iteration % 2u == 0u ? fluids::grid::device::component(first, axis) : fluids::grid::device::component(second, axis);
                float* next           = iteration % 2u == 0u ? fluids::grid::device::component(second, axis) : fluids::grid::device::component(first, axis);
                ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(count), diffusion_iteration_kernel, grid, axis, alpha, collider_ids, fluids::grid::device::component(source, axis), previous, boundary, next);
            }
            const float* result = iterations % 2u == 0u ? fluids::grid::device::component(first, axis) : fluids::grid::device::component(second, axis);
            ::cuda::copy_bytes(stream, ::cuda::std::span{result, static_cast<std::size_t>(count)}, ::cuda::std::span{fluids::grid::device::component(output, axis), static_cast<std::size_t>(count)});
        }
    }

    void diffusion_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t iterations, const float viscosity, const std::uint32_t* collider_ids, const device::VelocityBoundary boundary, const field::VectorView<const double> output_adjoint, const field::VectorView<double> first, const field::VectorView<double> second, const field::VectorView<double> source_adjoint) {
        const float alpha = viscosity * grid.time_step / (grid.grid.cell_size * grid.grid.cell_size);
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint64_t count = fluids::grid::device::face_count(grid.grid, axis);
            ::cuda::fill_bytes(stream, ::cuda::std::span{fluids::grid::device::component(first, axis), static_cast<std::size_t>(count)}, 0u);
            ::cuda::fill_bytes(stream, ::cuda::std::span{fluids::grid::device::component(second, axis), static_cast<std::size_t>(count)}, 0u);
            double* result_adjoint = iterations % 2u == 0u ? fluids::grid::device::component(first, axis) : fluids::grid::device::component(second, axis);
            ::cuda::copy_bytes(stream, ::cuda::std::span{fluids::grid::device::component(output_adjoint, axis), static_cast<std::size_t>(count)}, ::cuda::std::span{result_adjoint, static_cast<std::size_t>(count)});
            for (std::uint32_t reverse = 0u; reverse < iterations; ++reverse) {
                const std::uint32_t iteration = iterations - 1u - reverse;
                double* current               = iteration % 2u == 0u ? fluids::grid::device::component(second, axis) : fluids::grid::device::component(first, axis);
                double* previous              = iteration % 2u == 0u ? fluids::grid::device::component(first, axis) : fluids::grid::device::component(second, axis);
                ::cuda::fill_bytes(stream, ::cuda::std::span{previous, static_cast<std::size_t>(count)}, 0u);
                ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(count), diffusion_iteration_reverse_kernel, grid, axis, alpha, collider_ids, current, boundary, previous, fluids::grid::device::component(source_adjoint, axis));
            }
            ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(count), accumulate_double_kernel, fluids::grid::device::component(first, axis), fluids::grid::device::component(source_adjoint, axis), count);
        }
    }
} // namespace physica::fluids::gas::operators::kernels
