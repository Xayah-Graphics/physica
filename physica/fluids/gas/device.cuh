#ifndef PHYSICA_FLUIDS_GAS_DEVICE_CUH
#define PHYSICA_FLUIDS_GAS_DEVICE_CUH

#include <fluids/grid/device.cuh>
#include <cstdint>

namespace physica::fluids::gas::device {
    struct Discretization final {
        grid::device::Grid grid;
        std::uint32_t dimensions;
        float time_step;
    };

    struct ScalarBoundaryFace final {
        std::uint32_t mode;
        float value;
    };

    struct ScalarBoundary final {
        ScalarBoundaryFace faces[6];
    };

    struct VelocityBoundaryFace final {
        std::uint32_t mode;
        Vector3<float> value;
    };

    struct VelocityBoundary final {
        VelocityBoundaryFace faces[6];
    };

    __device__ inline int wrap(const int value, const int period) {
        const int remainder = value % period;
        return remainder < 0 ? remainder + period : remainder;
    }

    __device__ inline bool periodic(const VelocityBoundary boundary, const int dimension) {
        return boundary.faces[dimension * 2].mode == 3u && boundary.faces[dimension * 2 + 1].mode == 3u;
    }

    __device__ inline bool periodic(const ScalarBoundary boundary, const int dimension) {
        return boundary.faces[dimension * 2].mode == 2u && boundary.faces[dimension * 2 + 1].mode == 2u;
    }

    __device__ inline int map_coordinate(const int value, const int size, const bool is_periodic, const int period) {
        if (is_periodic) return wrap(value, period);
        return value < 0 ? 0 : value >= size ? size - 1 : value;
    }

    __device__ inline std::uint64_t mapped_face_index(int x, int y, int z, const Discretization discretization, const int axis, const VelocityBoundary boundary) {
        const int ex = grid::device::extent(discretization.grid, axis, 0);
        const int ey = grid::device::extent(discretization.grid, axis, 1);
        const int ez = grid::device::extent(discretization.grid, axis, 2);
        x            = map_coordinate(x, ex, periodic(boundary, 0), discretization.grid.nx);
        y            = map_coordinate(y, ey, periodic(boundary, 1), discretization.grid.ny);
        z            = map_coordinate(z, ez, periodic(boundary, 2), discretization.grid.nz);
        return grid::device::index3(x, y, z, ex, ey);
    }

    __device__ inline float load_face(const float* values, const int axis, const int x, const int y, const int z, const Discretization discretization, const VelocityBoundary boundary) {
        const int coordinates[3]{x, y, z};
        for (int dimension = 0; dimension < 3; ++dimension) {
            const int size = grid::device::extent(discretization.grid, axis, dimension);
            if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
            const int face = 2 * dimension + (coordinates[dimension] >= size);
            if (boundary.faces[face].mode == 0u || (boundary.faces[face].mode == 2u && axis == dimension)) return boundary.faces[face].value[axis];
        }
        return values[mapped_face_index(x, y, z, discretization, axis, boundary)];
    }
}

#endif
