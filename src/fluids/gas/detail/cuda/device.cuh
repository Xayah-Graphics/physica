#ifndef PHYSICA_FLUIDS_GAS_DETAIL_CUDA_DEVICE_CUH
#define PHYSICA_FLUIDS_GAS_DETAIL_CUDA_DEVICE_CUH

#include "types.h"
#include <cuda_runtime.h>

namespace physica::fluids::gas::detail::cuda {
    namespace {
        constexpr unsigned block_size = 256u;

        __host__ __device__ std::uint64_t cell_count(const Grid grid) {
            return static_cast<std::uint64_t>(grid.nx) * grid.ny * grid.nz;
        }

        __host__ __device__ int extent(const Grid grid, const int component_axis, const int dimension) {
            const int base = dimension == 0 ? grid.nx : dimension == 1 ? grid.ny : grid.nz;
            return base + (component_axis == dimension ? 1 : 0);
        }

        __host__ __device__ std::uint64_t face_count(const Grid grid, const int axis) {
            return static_cast<std::uint64_t>(extent(grid, axis, 0)) * extent(grid, axis, 1) * extent(grid, axis, 2);
        }

        __device__ std::uint64_t index3(const int x, const int y, const int z, const int nx, const int ny) {
            return static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(nx) * (static_cast<std::uint64_t>(y) + static_cast<std::uint64_t>(ny) * z);
        }

        __device__ void decode(const std::uint64_t index, const int nx, const int ny, int& x, int& y, int& z) {
            x                      = static_cast<int>(index % nx);
            const std::uint64_t yz = index / nx;
            y                      = static_cast<int>(yz % ny);
            z                      = static_cast<int>(yz / ny);
        }

        __host__ __device__ float* component(const StaggeredVectorView<float> field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const float* component(const StaggeredVectorView<const float> field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const float* component(const CenteredVectorView<const float> field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ double* component(const StaggeredVectorView<double> field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const double* component(const StaggeredVectorView<const double> field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ double* component(const CenteredVectorView<double> field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __device__ int wrap(const int value, const int period) {
            const int remainder = value % period;
            return remainder < 0 ? remainder + period : remainder;
        }

        __device__ bool periodic(const VelocityBoundaryData boundary, const int dimension) {
            return boundary.modes[dimension * 2] == 3u && boundary.modes[dimension * 2 + 1] == 3u;
        }

        __device__ bool periodic(const ScalarBoundaryData boundary, const int dimension) {
            return boundary.modes[dimension * 2] == 2u && boundary.modes[dimension * 2 + 1] == 2u;
        }

        __device__ int map_coordinate(const int value, const int size, const bool is_periodic, const int period) {
            if (is_periodic) return wrap(value, period);
            return max(0, min(size - 1, value));
        }

        __device__ std::uint64_t mapped_face_index(int x, int y, int z, const Grid grid, const int axis, const VelocityBoundaryData boundary) {
            const int ex = extent(grid, axis, 0);
            const int ey = extent(grid, axis, 1);
            const int ez = extent(grid, axis, 2);
            x            = map_coordinate(x, ex, periodic(boundary, 0), grid.nx);
            y            = map_coordinate(y, ey, periodic(boundary, 1), grid.ny);
            z            = map_coordinate(z, ez, periodic(boundary, 2), grid.nz);
            return index3(x, y, z, ex, ey);
        }

        __device__ float load_face(const float* values, const int axis, const int x, const int y, const int z, const Grid grid, const VelocityBoundaryData boundary) {
            const int coordinates[3]{x, y, z};
            for (int dimension = 0; dimension < 3; ++dimension) {
                const int size = extent(grid, axis, dimension);
                if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
                const int face = 2 * dimension + (coordinates[dimension] >= size);
                if (boundary.modes[face] == 0u || (boundary.modes[face] == 2u && axis == dimension)) return boundary.values[3 * face + axis];
            }
            return values[mapped_face_index(x, y, z, grid, axis, boundary)];
        }

        __device__ Vector face_position(const int axis, const int x, const int y, const int z, const Grid grid) {
            return {(x + (axis == 0 ? 0.0F : 0.5F)) * grid.cell_size, (y + (axis == 1 ? 0.0F : 0.5F)) * grid.cell_size, (z + (axis == 2 ? 0.0F : 0.5F)) * grid.cell_size};
        }

        __device__ Vector cell_position(const int x, const int y, const int z, const Grid grid) {
            return {(x + 0.5F) * grid.cell_size, (y + 0.5F) * grid.cell_size, (z + 0.5F) * grid.cell_size};
        }
    } // namespace
} // namespace physica::fluids::gas::detail::cuda

#endif
