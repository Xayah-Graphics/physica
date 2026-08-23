#include "device.cuh"
#include "solver_kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    namespace {
        __global__ void pressure_rhs_kernel(const Grid grid, const std::uint32_t pressure_anchor, const ConstStaggeredVectorView velocity, float* rhs) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (index == pressure_anchor) {
                rhs[index] = 0.0F;
                return;
            }
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            float divergence = velocity.x[index3(x + 1, y, z, grid.nx + 1, grid.ny)] - velocity.x[index3(x, y, z, grid.nx + 1, grid.ny)] + velocity.y[index3(x, y + 1, z, grid.nx, grid.ny + 1)] - velocity.y[index3(x, y, z, grid.nx, grid.ny + 1)];
            if (grid.dimensions == 3u) divergence += velocity.z[index3(x, y, z + 1, grid.nx, grid.ny)] - velocity.z[index3(x, y, z, grid.nx, grid.ny)];
            divergence /= grid.cell_size;
            rhs[index] = -grid.cell_size * grid.cell_size * divergence / grid.time_step;
        }

        __global__ void pressure_rhs_reverse_kernel(const Grid grid, const std::uint32_t pressure_anchor, const double* rhs_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || index == pressure_anchor) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double scale = -static_cast<double>(grid.cell_size) * rhs_adjoint[index] / grid.time_step;
            atomicAdd(velocity_adjoint.x + index3(x + 1, y, z, grid.nx + 1, grid.ny), scale);
            atomicAdd(velocity_adjoint.x + index3(x, y, z, grid.nx + 1, grid.ny), -scale);
            atomicAdd(velocity_adjoint.y + index3(x, y + 1, z, grid.nx, grid.ny + 1), scale);
            atomicAdd(velocity_adjoint.y + index3(x, y, z, grid.nx, grid.ny + 1), -scale);
            if (grid.dimensions == 3u) {
                atomicAdd(velocity_adjoint.z + index3(x, y, z + 1, grid.nx, grid.ny), scale);
                atomicAdd(velocity_adjoint.z + index3(x, y, z, grid.nx, grid.ny), -scale);
            }
        }

        __device__ bool projectable_face(const Grid grid, const int axis, const int x, const int y, const int z, int& first_x, int& first_y, int& first_z, int& second_x, int& second_y, int& second_z) {
            first_x = x - (axis == 0);
            first_y = y - (axis == 1);
            first_z = z - (axis == 2);
            second_x = x;
            second_y = y;
            second_z = z;
            if (first_x < 0 || first_x >= grid.nx || first_y < 0 || first_y >= grid.ny || first_z < 0 || first_z >= grid.nz) return false;
            return second_x >= 0 && second_x < grid.nx && second_y >= 0 && second_y < grid.ny && second_z >= 0 && second_z < grid.nz;
        }

        __global__ void project_velocity_kernel(const Grid grid, const int axis, const float* velocity, const float* pressure, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            int fx, fy, fz, sx, sy, sz;
            if (!projectable_face(grid, axis, x, y, z, fx, fy, fz, sx, sy, sz)) {
                output[index] = velocity[index];
                return;
            }
            output[index] = velocity[index] - grid.time_step * (pressure[index3(sx, sy, sz, grid.nx, grid.ny)] - pressure[index3(fx, fy, fz, grid.nx, grid.ny)]) / grid.cell_size;
        }

        __global__ void project_velocity_reverse_kernel(const Grid grid, const int axis, const double* output_adjoint, double* velocity_adjoint, double* pressure_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            velocity_adjoint[index] += output_adjoint[index];
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            int fx, fy, fz, sx, sy, sz;
            if (!projectable_face(grid, axis, x, y, z, fx, fy, fz, sx, sy, sz)) return;
            const double scale = static_cast<double>(grid.time_step) * output_adjoint[index] / grid.cell_size;
            atomicAdd(pressure_adjoint + index3(fx, fy, fz, grid.nx, grid.ny), scale);
            atomicAdd(pressure_adjoint + index3(sx, sy, sz, grid.nx, grid.ny), -scale);
        }

        __device__ bool pressure_periodic(const ScalarBoundaryData boundary, const int dimension) {
            return boundary.modes[dimension * 2] == 2u && boundary.modes[dimension * 2 + 1] == 2u;
        }

        __device__ void pressure_neighbor(const Grid grid, const ScalarBoundaryData boundary, const int x, const int y, const int z, const int dimension, const int direction, int& neighbor_x, int& neighbor_y, int& neighbor_z, bool& connected, bool& fixed, float& fixed_value) {
            neighbor_x = x + (dimension == 0 ? direction : 0);
            neighbor_y = y + (dimension == 1 ? direction : 0);
            neighbor_z = z + (dimension == 2 ? direction : 0);
            const int coordinate = dimension == 0 ? neighbor_x : dimension == 1 ? neighbor_y : neighbor_z;
            const int size = dimension == 0 ? grid.nx : dimension == 1 ? grid.ny : grid.nz;
            if (coordinate < 0 || coordinate >= size) {
                if (pressure_periodic(boundary, dimension)) {
                    if (dimension == 0) neighbor_x = wrap(neighbor_x, grid.nx);
                    if (dimension == 1) neighbor_y = wrap(neighbor_y, grid.ny);
                    if (dimension == 2) neighbor_z = wrap(neighbor_z, grid.nz);
                } else {
                    const int face = dimension * 2 + (direction > 0);
                    fixed = boundary.modes[face] == 0u;
                    fixed_value = boundary.values[face];
                    connected = false;
                    return;
                }
            }
            connected = true;
            fixed = false;
        }

        __global__ void rbgs_kernel(const Grid grid, const int parity, const std::uint32_t pressure_anchor, const ScalarBoundaryData boundary, const float* rhs, float* pressure) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (((x + y + z) & 1) != parity || index == pressure_anchor) return;
            float sum = rhs[index];
            float diagonal = 0.0F;
            for (int dimension = 0; dimension < grid.dimensions; ++dimension) {
                for (int direction = -1; direction <= 1; direction += 2) {
                    int nx, ny, nz;
                    bool connected, fixed;
                    float fixed_value{};
                    pressure_neighbor(grid, boundary, x, y, z, dimension, direction, nx, ny, nz, connected, fixed, fixed_value);
                    if (connected) {
                        sum += pressure[index3(nx, ny, nz, grid.nx, grid.ny)];
                        diagonal += 1.0F;
                    } else if (fixed) {
                        sum += fixed_value;
                        diagonal += 1.0F;
                    }
                }
            }
            pressure[index] = diagonal == 0.0F ? 0.0F : sum / diagonal;
        }

        __global__ void rbgs_reverse_kernel(const Grid grid, const int parity, const std::uint32_t pressure_anchor, const ScalarBoundaryData boundary, double* pressure_adjoint, double* rhs_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (((x + y + z) & 1) != parity || index == pressure_anchor) return;
            float diagonal = 0.0F;
            int neighbors[6];
            int neighbor_count = 0;
            for (int dimension = 0; dimension < grid.dimensions; ++dimension) {
                for (int direction = -1; direction <= 1; direction += 2) {
                    int nx, ny, nz;
                    bool connected, fixed;
                    float fixed_value{};
                    pressure_neighbor(grid, boundary, x, y, z, dimension, direction, nx, ny, nz, connected, fixed, fixed_value);
                    if (connected) {
                        neighbors[neighbor_count++] = static_cast<int>(index3(nx, ny, nz, grid.nx, grid.ny));
                        diagonal += 1.0F;
                    } else if (fixed) diagonal += 1.0F;
                }
            }
            const double value = diagonal == 0.0F ? 0.0 : pressure_adjoint[index] / diagonal;
            pressure_adjoint[index] = 0.0;
            rhs_adjoint[index] += value;
            for (int neighbor = 0; neighbor < neighbor_count; ++neighbor) atomicAdd(pressure_adjoint + neighbors[neighbor], value);
        }
    } // namespace

    void pressure_rhs_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t pressure_anchor, const ConstStaggeredVectorView velocity, const ScalarView rhs) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), pressure_rhs_kernel, grid, pressure_anchor, velocity, rhs.values);
    }

    void pressure_rhs_vjp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t pressure_anchor, const ConstScalarAdjointView rhs_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), pressure_rhs_reverse_kernel, grid, pressure_anchor, rhs_adjoint.values, velocity_adjoint);
    }

    void pressure_forward(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const ScalarBoundaryData boundary, const ConstScalarView rhs, const ScalarView pressure) {
        for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
            ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), rbgs_kernel, grid, 0, pressure_anchor, boundary, rhs.values, pressure.values);
            ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), rbgs_kernel, grid, 1, pressure_anchor, boundary, rhs.values, pressure.values);
        }
    }

    void pressure_vjp(const ::cuda::stream_ref stream, const Grid grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const ScalarBoundaryData boundary, const ScalarAdjointView pressure_adjoint, const ScalarAdjointView rhs_adjoint) {
        for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
            ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), rbgs_reverse_kernel, grid, 1, pressure_anchor, boundary, pressure_adjoint.values, rhs_adjoint.values);
            ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), rbgs_reverse_kernel, grid, 0, pressure_anchor, boundary, pressure_adjoint.values, rhs_adjoint.values);
        }
    }

    void project_velocity_forward(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorView velocity, const ConstScalarView pressure, const StaggeredVectorView output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), project_velocity_kernel, grid, axis, component(velocity, axis), pressure.values, component(output, axis));
    }

    void project_velocity_vjp(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorAdjointView output_adjoint, const StaggeredVectorAdjointView velocity_adjoint, const ScalarAdjointView pressure_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), project_velocity_reverse_kernel, grid, axis, component(output_adjoint, axis), component(velocity_adjoint, axis), pressure_adjoint.values);
    }
} // namespace physica::fluids::gas::keyframe_smoke::cuda_detail
