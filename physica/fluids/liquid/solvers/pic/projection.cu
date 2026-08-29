#include "projection-kernels.h"
#include <cub/device/device_reduce.cuh>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>

namespace physica::fluids::liquid::pic::kernels::projection {
    namespace {
        constexpr std::uint32_t air   = 0u;
        constexpr std::uint32_t fluid = 1u;
        constexpr std::uint32_t solid = 2u;

        __device__ std::uint32_t cell_type(const grid::device::Grid grid, const std::uint32_t* cell_types, const int x, const int y, const int z) {
            if (!grid::device::valid_cell(grid, x, y, z)) return solid;
            return cell_types[grid::device::cell_index(grid, x, y, z)];
        }

        __device__ float interface_fraction(const float fluid_level_set, const float air_level_set) {
            return ::cuda::std::max(0.01F, ::cuda::std::min(1.0F, fluid_level_set / (fluid_level_set - air_level_set)));
        }

        __device__ float neighbor_coefficient(const grid::device::Grid grid, const std::uint32_t* cell_types, const float* level_set, const int x, const int y, const int z, const int nx, const int ny, const int nz) {
            const std::uint32_t type = cell_type(grid, cell_types, nx, ny, nz);
            if (type == solid) return 0.0F;
            const float inverse_h2 = 1.0F / (grid.cell_size * grid.cell_size);
            if (type == fluid) return inverse_h2;
            const float center_phi   = level_set[grid::device::cell_index(grid, x, y, z)];
            const float neighbor_phi = grid::device::valid_cell(grid, nx, ny, nz) ? level_set[grid::device::cell_index(grid, nx, ny, nz)] : grid.cell_size;
            return inverse_h2 / interface_fraction(center_phi, neighbor_phi);
        }

        __global__ void build_system_kernel(const grid::device::Grid grid, const float time_step, const float density, const std::uint32_t* cell_types, const float* level_set, const field::VectorView<const float> velocity, float* rhs, float* diagonal, float* pressure, float* residual, float* preconditioned_residual, float* direction, float* products) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::cell_count(grid)) return;
            pressure[index] = 0.0F;
            if (cell_types[index] != fluid) {
                rhs[index]                     = 0.0F;
                diagonal[index]                = 1.0F;
                residual[index]                = 0.0F;
                preconditioned_residual[index] = 0.0F;
                direction[index]               = 0.0F;
                products[index]                = 0.0F;
                return;
            }
            int x, y, z;
            grid::device::decode(index, static_cast<int>(grid.nx), static_cast<int>(grid.ny), x, y, z);
            const float divergence = (
                velocity.x[grid::device::face_index(grid, 0, x + 1, y, z)] - velocity.x[grid::device::face_index(grid, 0, x, y, z)] +
                velocity.y[grid::device::face_index(grid, 1, x, y + 1, z)] - velocity.y[grid::device::face_index(grid, 1, x, y, z)] +
                velocity.z[grid::device::face_index(grid, 2, x, y, z + 1)] - velocity.z[grid::device::face_index(grid, 2, x, y, z)]) /
                grid.cell_size;
            const float value_diagonal =
                neighbor_coefficient(grid, cell_types, level_set, x, y, z, x - 1, y, z) +
                neighbor_coefficient(grid, cell_types, level_set, x, y, z, x + 1, y, z) +
                neighbor_coefficient(grid, cell_types, level_set, x, y, z, x, y - 1, z) +
                neighbor_coefficient(grid, cell_types, level_set, x, y, z, x, y + 1, z) +
                neighbor_coefficient(grid, cell_types, level_set, x, y, z, x, y, z - 1) +
                neighbor_coefficient(grid, cell_types, level_set, x, y, z, x, y, z + 1);
            rhs[index]                     = -density * divergence / time_step;
            diagonal[index]                = value_diagonal;
            residual[index]                = rhs[index];
            preconditioned_residual[index] = rhs[index] / value_diagonal;
            direction[index]               = preconditioned_residual[index];
            products[index]                = residual[index] * preconditioned_residual[index];
        }

        __device__ float apply_matrix_cell(const grid::device::Grid grid, const std::uint32_t* cell_types, const float* level_set, const float* values, const int x, const int y, const int z) {
            const std::size_t center = grid::device::cell_index(grid, x, y, z);
            float result{};
            constexpr int offsets[6][3]{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
            for (int neighbor = 0; neighbor < 6; ++neighbor) {
                const int nx = x + offsets[neighbor][0];
                const int ny = y + offsets[neighbor][1];
                const int nz = z + offsets[neighbor][2];
                const float coefficient = neighbor_coefficient(grid, cell_types, level_set, x, y, z, nx, ny, nz);
                if (coefficient == 0.0F) continue;
                result += coefficient * values[center];
                if (cell_type(grid, cell_types, nx, ny, nz) == fluid) result -= coefficient * values[grid::device::cell_index(grid, nx, ny, nz)];
            }
            return result;
        }

        __global__ void apply_matrix_kernel(const grid::device::Grid grid, const std::uint32_t* cell_types, const float* level_set, const float* direction, const std::uint32_t* state, float* matrix_direction, float* products) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::cell_count(grid)) return;
            if (state[0] == 0u || cell_types[index] != fluid) {
                matrix_direction[index] = 0.0F;
                products[index]         = 0.0F;
                return;
            }
            int x, y, z;
            grid::device::decode(index, static_cast<int>(grid.nx), static_cast<int>(grid.ny), x, y, z);
            matrix_direction[index] = apply_matrix_cell(grid, cell_types, level_set, direction, x, y, z);
            products[index]         = direction[index] * matrix_direction[index];
        }

        __global__ void initialize_solver_kernel(float* scalars, std::uint32_t* state) {
            if (threadIdx.x != 0u || blockIdx.x != 0u) return;
            scalars[1] = scalars[0];
            scalars[6] = scalars[0] == 0.0F ? 0.0F : 1.0F;
            state[0]   = scalars[0] > 0.0F ? 1u : 0u;
            state[1]   = 0u;
        }

        __global__ void prepare_iteration_kernel(float* scalars, const std::uint32_t* state) {
            if (threadIdx.x != 0u || blockIdx.x != 0u || state[0] == 0u) return;
            scalars[3] = scalars[0] / scalars[2];
        }

        __global__ void update_solution_kernel(const std::size_t count, const std::uint32_t* cell_types, const float* diagonal, const float* direction, const float* matrix_direction, const float* scalars, const std::uint32_t* state, float* pressure, float* residual, float* preconditioned_residual, float* products) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) return;
            if (state[0] == 0u || cell_types[index] != fluid) {
                products[index] = 0.0F;
                return;
            }
            pressure[index] += scalars[3] * direction[index];
            residual[index] -= scalars[3] * matrix_direction[index];
            preconditioned_residual[index] = residual[index] / diagonal[index];
            products[index]                = residual[index] * preconditioned_residual[index];
        }

        __global__ void finish_iteration_kernel(float* scalars, std::uint32_t* state, const std::uint32_t iteration, const float tolerance) {
            if (threadIdx.x != 0u || blockIdx.x != 0u || state[0] == 0u) return;
            scalars[6] = ::cuda::std::sqrt(::cuda::std::max(0.0F, scalars[4] / scalars[1]));
            state[1]   = iteration + 1u;
            if (scalars[6] <= tolerance) {
                state[0] = 0u;
                return;
            }
            scalars[5] = scalars[4] / scalars[0];
            scalars[0] = scalars[4];
        }

        __global__ void update_direction_kernel(const std::size_t count, const std::uint32_t* cell_types, const float* preconditioned_residual, const float* scalars, const std::uint32_t* state, float* direction) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count || state[0] == 0u || cell_types[index] != fluid) return;
            direction[index] = preconditioned_residual[index] + scalars[5] * direction[index];
        }

        __device__ float boundary_velocity(const grid::device::Grid grid, const int axis) {
            if (axis == 0) return grid.velocity_x;
            if (axis == 1) return grid.velocity_y;
            return grid.velocity_z;
        }

        __global__ void project_velocity_kernel(const grid::device::Grid grid, const float time_step, const float density, const int axis, const std::uint32_t* cell_types, const float* level_set, const float* input, const float* pressure, float* output) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::face_count(grid, axis)) return;
            int x, y, z;
            grid::device::decode(index, grid::device::extent(grid, axis, 0), grid::device::extent(grid, axis, 1), x, y, z);
            const int negative_x = x - (axis == 0 ? 1 : 0);
            const int negative_y = y - (axis == 1 ? 1 : 0);
            const int negative_z = z - (axis == 2 ? 1 : 0);
            const int positive_x = x;
            const int positive_y = y;
            const int positive_z = z;
            const std::uint32_t negative_type = cell_type(grid, cell_types, negative_x, negative_y, negative_z);
            const std::uint32_t positive_type = cell_type(grid, cell_types, positive_x, positive_y, positive_z);
            if (negative_type == solid || positive_type == solid) {
                output[index] = boundary_velocity(grid, axis);
                return;
            }
            if (negative_type == air && positive_type == air) {
                output[index] = input[index];
                return;
            }
            float negative_pressure{};
            float positive_pressure{};
            float distance = grid.cell_size;
            if (negative_type == fluid) negative_pressure = pressure[grid::device::cell_index(grid, negative_x, negative_y, negative_z)];
            if (positive_type == fluid) positive_pressure = pressure[grid::device::cell_index(grid, positive_x, positive_y, positive_z)];
            if (negative_type != positive_type) {
                const int fluid_x = negative_type == fluid ? negative_x : positive_x;
                const int fluid_y = negative_type == fluid ? negative_y : positive_y;
                const int fluid_z = negative_type == fluid ? negative_z : positive_z;
                const int air_x   = negative_type == air ? negative_x : positive_x;
                const int air_y   = negative_type == air ? negative_y : positive_y;
                const int air_z   = negative_type == air ? negative_z : positive_z;
                const float fluid_phi = level_set[grid::device::cell_index(grid, fluid_x, fluid_y, fluid_z)];
                const float air_phi   = level_set[grid::device::cell_index(grid, air_x, air_y, air_z)];
                distance *= interface_fraction(fluid_phi, air_phi);
            }
            output[index] = input[index] - time_step * (positive_pressure - negative_pressure) / (density * distance);
        }
    } // namespace

    std::size_t reduction_storage_size(const std::size_t count) {
        std::size_t bytes{};
        cub::DeviceReduce::Sum(nullptr, bytes, static_cast<const float*>(nullptr), static_cast<float*>(nullptr), count);
        return bytes;
    }

    void project(const ::cuda::stream_ref stream, const grid::device::Grid grid, const float time_step, const float density, const std::uint32_t maximum_iterations, const float tolerance, const std::uint32_t* cell_types, const float* level_set, const field::VectorView<const float> input_velocity, const field::VectorView<float> output_velocity, float* rhs, float* pressure, float* diagonal, float* residual, float* preconditioned_residual, float* direction, float* matrix_direction, float* products, float* scalars, std::uint32_t* state, void* reduction_scratch, std::size_t reduction_scratch_bytes) {
        const std::size_t count = grid::device::cell_count(grid);
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(count), build_system_kernel, grid, time_step, density, cell_types, level_set, input_velocity, rhs, diagonal, pressure, residual, preconditioned_residual, direction, products);
        cub::DeviceReduce::Sum(reduction_scratch, reduction_scratch_bytes, products, scalars, count, stream.get());
        ::cuda::launch(stream, ::cuda::distribute<1u>(1u), initialize_solver_kernel, scalars, state);
        for (std::uint32_t iteration = 0u; iteration < maximum_iterations; ++iteration) {
            ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(count), apply_matrix_kernel, grid, cell_types, level_set, direction, state, matrix_direction, products);
            cub::DeviceReduce::Sum(reduction_scratch, reduction_scratch_bytes, products, scalars + 2u, count, stream.get());
            ::cuda::launch(stream, ::cuda::distribute<1u>(1u), prepare_iteration_kernel, scalars, state);
            ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(count), update_solution_kernel, count, cell_types, diagonal, direction, matrix_direction, scalars, state, pressure, residual, preconditioned_residual, products);
            cub::DeviceReduce::Sum(reduction_scratch, reduction_scratch_bytes, products, scalars + 4u, count, stream.get());
            ::cuda::launch(stream, ::cuda::distribute<1u>(1u), finish_iteration_kernel, scalars, state, iteration, tolerance);
            ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(count), update_direction_kernel, count, cell_types, preconditioned_residual, scalars, state, direction);
        }
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::face_count(grid, axis)), project_velocity_kernel, grid, time_step, density, axis, cell_types, level_set, grid::device::component(input_velocity, axis), pressure, grid::device::component(output_velocity, axis));
    }
} // namespace physica::fluids::liquid::pic::kernels::projection

