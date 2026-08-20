#include "../domain/device.cuh"
#include "kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::smoke::cuda_detail {
    namespace {
        __global__ void pressure_rhs_kernel(const Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, float* rhs) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (cell_mask[index] != 0u || index == pressure_anchor) {
                rhs[index] = 0.0F;
                return;
            }
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const float divergence = (velocity.x[index3(x + 1, y, z, grid.nx + 1, grid.ny)] - velocity.x[index3(x, y, z, grid.nx + 1, grid.ny)] + velocity.y[index3(x, y + 1, z, grid.nx, grid.ny + 1)] - velocity.y[index3(x, y, z, grid.nx, grid.ny + 1)] + velocity.z[index3(x, y, z + 1, grid.nx, grid.ny)] - velocity.z[index3(x, y, z, grid.nx, grid.ny)]) / grid.cell_size;
            rhs[index]             = -grid.cell_size * grid.cell_size * divergence / grid.time_step;
        }

        __device__ bool projectable_face(const Grid grid, const int axis, const int x, const int y, const int z, const std::uint32_t* cell_mask, int& first_x, int& first_y, int& first_z, int& second_x, int& second_y, int& second_z) {
            first_x  = x - (axis == 0);
            first_y  = y - (axis == 1);
            first_z  = z - (axis == 2);
            second_x = x;
            second_y = y;
            second_z = z;
            if (first_x < 0 || first_x >= grid.nx || first_y < 0 || first_y >= grid.ny || first_z < 0 || first_z >= grid.nz) return false;
            if (second_x < 0 || second_x >= grid.nx || second_y < 0 || second_y >= grid.ny || second_z < 0 || second_z >= grid.nz) return false;
            return cell_mask[index3(first_x, first_y, first_z, grid.nx, grid.ny)] == 0u && cell_mask[index3(second_x, second_y, second_z, grid.nx, grid.ny)] == 0u;
        }

        __global__ void project_velocity_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const float* velocity, const float* pressure, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            int fx, fy, fz, sx, sy, sz;
            if (!projectable_face(grid, axis, x, y, z, cell_mask, fx, fy, fz, sx, sy, sz)) {
                output[index] = velocity[index];
                return;
            }
            output[index] = velocity[index] - grid.time_step * (pressure[index3(sx, sy, sz, grid.nx, grid.ny)] - pressure[index3(fx, fy, fz, grid.nx, grid.ny)]) / grid.cell_size;
        }

        __global__ void project_velocity_reverse_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const double* output_adjoint, double* velocity_adjoint, double* pressure_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            velocity_adjoint[index] += output_adjoint[index];
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            int fx, fy, fz, sx, sy, sz;
            if (!projectable_face(grid, axis, x, y, z, cell_mask, fx, fy, fz, sx, sy, sz)) return;
            const double scale = static_cast<double>(grid.time_step) * output_adjoint[index] / grid.cell_size;
            atomicAdd(pressure_adjoint + index3(fx, fy, fz, grid.nx, grid.ny), scale);
            atomicAdd(pressure_adjoint + index3(sx, sy, sz, grid.nx, grid.ny), -scale);
        }

        __global__ void pressure_rhs_reverse_kernel(const Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const double* rhs_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u || index == pressure_anchor) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double scale = -static_cast<double>(grid.cell_size) * rhs_adjoint[index] / grid.time_step;
            atomicAdd(velocity_adjoint.x + index3(x + 1, y, z, grid.nx + 1, grid.ny), scale);
            atomicAdd(velocity_adjoint.x + index3(x, y, z, grid.nx + 1, grid.ny), -scale);
            atomicAdd(velocity_adjoint.y + index3(x, y + 1, z, grid.nx, grid.ny + 1), scale);
            atomicAdd(velocity_adjoint.y + index3(x, y, z, grid.nx, grid.ny + 1), -scale);
            atomicAdd(velocity_adjoint.z + index3(x, y, z + 1, grid.nx, grid.ny), scale);
            atomicAdd(velocity_adjoint.z + index3(x, y, z, grid.nx, grid.ny), -scale);
        }
    } // namespace

    void pressure_rhs_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const ScalarView rhs) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), pressure_rhs_kernel, grid, pressure_anchor, cell_mask, velocity, rhs.values);
    }

    void pressure_rhs_vjp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ConstScalarAdjointView rhs_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), pressure_rhs_reverse_kernel, grid, pressure_anchor, cell_mask, rhs_adjoint.values, velocity_adjoint);
    }

    void project_velocity_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const ConstScalarView pressure, const StaggeredVectorView output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), project_velocity_kernel, grid, axis, cell_mask, component(velocity, axis), pressure.values, component(output, axis));
    }

    void project_velocity_vjp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorAdjointView output_adjoint, const StaggeredVectorAdjointView velocity_adjoint, const ScalarAdjointView pressure_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), project_velocity_reverse_kernel, grid, axis, cell_mask, component(output_adjoint, axis), component(velocity_adjoint, axis), pressure_adjoint.values);
    }

} // namespace physica::fluids::gas::smoke::cuda_detail
