#include "../detail/cuda/device.cuh"
#include "projection-kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::operators::cuda_backend {
    namespace {
        __global__ void pressure_rhs_kernel(const detail::cuda::Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, float* rhs) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::cell_count(grid)) return;
            if (collider_ids[index] != 0u || index == pressure_anchor) {
                rhs[index] = 0.0F;
                return;
            }
            int x, y, z;
            detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
            float divergence = velocity.x[detail::cuda::index3(x + 1, y, z, grid.nx + 1, grid.ny)] - velocity.x[detail::cuda::index3(x, y, z, grid.nx + 1, grid.ny)] + velocity.y[detail::cuda::index3(x, y + 1, z, grid.nx, grid.ny + 1)] - velocity.y[detail::cuda::index3(x, y, z, grid.nx, grid.ny + 1)];
            if (grid.dimensions == 3u) divergence += velocity.z[detail::cuda::index3(x, y, z + 1, grid.nx, grid.ny)] - velocity.z[detail::cuda::index3(x, y, z, grid.nx, grid.ny)];
            divergence /= grid.cell_size;
            rhs[index] = -grid.cell_size * grid.cell_size * divergence / grid.time_step;
        }

        __device__ bool projectable_face(const detail::cuda::Grid grid, const int axis, const int x, const int y, const int z, const std::uint32_t* collider_ids, int& first_x, int& first_y, int& first_z, int& second_x, int& second_y, int& second_z) {
            first_x  = x - (axis == 0);
            first_y  = y - (axis == 1);
            first_z  = z - (axis == 2);
            second_x = x;
            second_y = y;
            second_z = z;
            if (first_x < 0 || first_x >= grid.nx || first_y < 0 || first_y >= grid.ny || first_z < 0 || first_z >= grid.nz) return false;
            if (second_x < 0 || second_x >= grid.nx || second_y < 0 || second_y >= grid.ny || second_z < 0 || second_z >= grid.nz) return false;
            return collider_ids[detail::cuda::index3(first_x, first_y, first_z, grid.nx, grid.ny)] == 0u && collider_ids[detail::cuda::index3(second_x, second_y, second_z, grid.nx, grid.ny)] == 0u;
        }

        __global__ void project_velocity_kernel(const detail::cuda::Grid grid, const int axis, const std::uint32_t* collider_ids, const float* velocity, const float* pressure, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::face_count(grid, axis)) return;
            int x, y, z;
            detail::cuda::decode(index, detail::cuda::extent(grid, axis, 0), detail::cuda::extent(grid, axis, 1), x, y, z);
            int fx, fy, fz, sx, sy, sz;
            if (!projectable_face(grid, axis, x, y, z, collider_ids, fx, fy, fz, sx, sy, sz)) {
                output[index] = velocity[index];
                return;
            }
            output[index] = velocity[index] - grid.time_step * (pressure[detail::cuda::index3(sx, sy, sz, grid.nx, grid.ny)] - pressure[detail::cuda::index3(fx, fy, fz, grid.nx, grid.ny)]) / grid.cell_size;
        }

        __global__ void project_velocity_reverse_kernel(const detail::cuda::Grid grid, const int axis, const std::uint32_t* collider_ids, const double* output_adjoint, double* velocity_adjoint, double* pressure_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::face_count(grid, axis)) return;
            velocity_adjoint[index] += output_adjoint[index];
            int x, y, z;
            detail::cuda::decode(index, detail::cuda::extent(grid, axis, 0), detail::cuda::extent(grid, axis, 1), x, y, z);
            int fx, fy, fz, sx, sy, sz;
            if (!projectable_face(grid, axis, x, y, z, collider_ids, fx, fy, fz, sx, sy, sz)) return;
            const double scale = static_cast<double>(grid.time_step) * output_adjoint[index] / grid.cell_size;
            atomicAdd(pressure_adjoint + detail::cuda::index3(fx, fy, fz, grid.nx, grid.ny), scale);
            atomicAdd(pressure_adjoint + detail::cuda::index3(sx, sy, sz, grid.nx, grid.ny), -scale);
        }

        __global__ void pressure_rhs_reverse_kernel(const detail::cuda::Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const double* rhs_adjoint, const detail::cuda::StaggeredVectorView<double> velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::cell_count(grid) || collider_ids[index] != 0u || index == pressure_anchor) return;
            int x, y, z;
            detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
            const double scale = -static_cast<double>(grid.cell_size) * rhs_adjoint[index] / grid.time_step;
            atomicAdd(velocity_adjoint.x + detail::cuda::index3(x + 1, y, z, grid.nx + 1, grid.ny), scale);
            atomicAdd(velocity_adjoint.x + detail::cuda::index3(x, y, z, grid.nx + 1, grid.ny), -scale);
            atomicAdd(velocity_adjoint.y + detail::cuda::index3(x, y + 1, z, grid.nx, grid.ny + 1), scale);
            atomicAdd(velocity_adjoint.y + detail::cuda::index3(x, y, z, grid.nx, grid.ny + 1), -scale);
            if (grid.dimensions == 3u) {
                atomicAdd(velocity_adjoint.z + detail::cuda::index3(x, y, z + 1, grid.nx, grid.ny), scale);
                atomicAdd(velocity_adjoint.z + detail::cuda::index3(x, y, z, grid.nx, grid.ny), -scale);
            }
        }
    } // namespace

    void pressure_rhs_forward(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::ScalarView<float> rhs) {
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::cell_count(grid)), pressure_rhs_kernel, grid, pressure_anchor, collider_ids, velocity, rhs.values);
    }

    void pressure_rhs_vjp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const detail::cuda::ScalarView<const double> rhs_adjoint, const detail::cuda::StaggeredVectorView<double> velocity_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::cell_count(grid)), pressure_rhs_reverse_kernel, grid, pressure_anchor, collider_ids, rhs_adjoint.values, velocity_adjoint);
    }

    void project_velocity_forward(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::ScalarView<const float> pressure, const detail::cuda::StaggeredVectorView<float> output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::face_count(grid, axis)), project_velocity_kernel, grid, axis, collider_ids, detail::cuda::component(velocity, axis), pressure.values, detail::cuda::component(output, axis));
    }

    void project_velocity_vjp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const double> output_adjoint, const detail::cuda::StaggeredVectorView<double> velocity_adjoint, const detail::cuda::ScalarView<double> pressure_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::face_count(grid, axis)), project_velocity_reverse_kernel, grid, axis, collider_ids, detail::cuda::component(output_adjoint, axis), detail::cuda::component(velocity_adjoint, axis), pressure_adjoint.values);
    }

} // namespace physica::fluids::gas::operators::cuda_backend
