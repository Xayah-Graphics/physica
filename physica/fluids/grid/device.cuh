#ifndef PHYSICA_FLUIDS_GRID_DEVICE_CUH
#define PHYSICA_FLUIDS_GRID_DEVICE_CUH

#include <cstdint>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <simulation/field/device.cuh>

namespace physica::fluids::grid::device {
    struct Grid final {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        float cell_size;
        float origin_x;
        float origin_y;
        float origin_z;
        float velocity_x;
        float velocity_y;
        float velocity_z;
    };

    inline constexpr std::uint32_t block_size = 256u;

    __host__ __device__ inline std::uint64_t cell_count(const Grid grid) {
        return static_cast<std::uint64_t>(grid.nx) * grid.ny * grid.nz;
    }

    __host__ __device__ inline std::uint64_t index3(const int x, const int y, const int z, const std::uint32_t extent_x, const std::uint32_t extent_y) {
        return (static_cast<std::uint64_t>(z) * extent_y + static_cast<std::uint32_t>(y)) * extent_x + static_cast<std::uint32_t>(x);
    }

    __host__ __device__ inline std::uint32_t extent(const Grid grid, const int axis, const int dimension) {
        const std::uint32_t base = dimension == 0 ? grid.nx : dimension == 1 ? grid.ny : grid.nz;
        return base + (axis == dimension ? 1 : 0);
    }

    __host__ __device__ inline std::uint64_t face_count(const Grid grid, const int axis) {
        return static_cast<std::uint64_t>(extent(grid, axis, 0)) * extent(grid, axis, 1) * extent(grid, axis, 2);
    }

    __host__ __device__ inline std::uint64_t cell_index(const Grid grid, const int x, const int y, const int z) {
        return (static_cast<std::uint64_t>(z) * grid.ny + static_cast<std::uint32_t>(y)) * grid.nx + static_cast<std::uint32_t>(x);
    }

    __host__ __device__ inline std::uint64_t face_index(const Grid grid, const int axis, const int x, const int y, const int z) {
        const std::uint32_t ex = extent(grid, axis, 0);
        const std::uint32_t ey = extent(grid, axis, 1);
        return (static_cast<std::uint64_t>(z) * ey + static_cast<std::uint32_t>(y)) * ex + static_cast<std::uint32_t>(x);
    }

    template <class Value>
    __host__ __device__ inline Value* component(const simulation::VectorView<Value> field, const int axis) {
        if (axis == 0) return field.x;
        if (axis == 1) return field.y;
        return field.z;
    }

    __host__ __device__ inline void decode(const std::uint64_t index, const std::uint32_t ex, const std::uint32_t ey, int& x, int& y, int& z) {
        x = static_cast<int>(index % ex);
        y = static_cast<int>((index / ex) % ey);
        z = static_cast<int>(index / (static_cast<std::uint64_t>(ex) * ey));
    }

    __host__ __device__ inline Vector3<float> cell_position(const Grid grid, const int x, const int y, const int z) {
        return {
            grid.origin_x + (static_cast<float>(x) + 0.5F) * grid.cell_size,
            grid.origin_y + (static_cast<float>(y) + 0.5F) * grid.cell_size,
            grid.origin_z + (static_cast<float>(z) + 0.5F) * grid.cell_size,
        };
    }

    __host__ __device__ inline Vector3<float> face_position(const Grid grid, const int axis, const int x, const int y, const int z) {
        return {
            grid.origin_x + (static_cast<float>(x) + (axis == 0 ? 0.0F : 0.5F)) * grid.cell_size,
            grid.origin_y + (static_cast<float>(y) + (axis == 1 ? 0.0F : 0.5F)) * grid.cell_size,
            grid.origin_z + (static_cast<float>(z) + (axis == 2 ? 0.0F : 0.5F)) * grid.cell_size,
        };
    }

    __host__ __device__ inline bool valid_cell(const Grid grid, const int x, const int y, const int z) {
        return x >= 0 && x < static_cast<int>(grid.nx) && y >= 0 && y < static_cast<int>(grid.ny) && z >= 0 && z < static_cast<int>(grid.nz);
    }

    __host__ __device__ inline bool valid_face(const Grid grid, const int axis, const int x, const int y, const int z) {
        return x >= 0 && x < static_cast<int>(extent(grid, axis, 0)) && y >= 0 && y < static_cast<int>(extent(grid, axis, 1)) && z >= 0 && z < static_cast<int>(extent(grid, axis, 2));
    }

    __host__ __device__ inline void interior_cell(const Grid grid, const Vector3<float> position, int& x, int& y, int& z) {
        x = static_cast<int>(::cuda::std::floor((position.x - grid.origin_x) / grid.cell_size));
        y = static_cast<int>(::cuda::std::floor((position.y - grid.origin_y) / grid.cell_size));
        z = static_cast<int>(::cuda::std::floor((position.z - grid.origin_z) / grid.cell_size));
        x = ::cuda::std::clamp(x, 1, static_cast<int>(grid.nx) - 2);
        y = ::cuda::std::clamp(y, 1, static_cast<int>(grid.ny) - 2);
        z = ::cuda::std::clamp(z, 1, static_cast<int>(grid.nz) - 2);
    }

    __host__ __device__ inline void quadratic_weights(const float coordinate, int& base, float weights[3]) {
        base              = static_cast<int>(::cuda::std::floor(coordinate - 0.5F));
        const float local = coordinate - static_cast<float>(base);
        weights[0]        = 0.5F * (1.5F - local) * (1.5F - local);
        weights[1]        = 0.75F - (local - 1.0F) * (local - 1.0F);
        weights[2]        = 0.5F * (local - 0.5F) * (local - 0.5F);
    }

    __host__ __device__ inline void face_stencil(const Grid grid, const Vector3<float> position, const int axis, int& base_x, int& base_y, int& base_z, float weights_x[3], float weights_y[3], float weights_z[3]) {
        quadratic_weights((position.x - grid.origin_x) / grid.cell_size - (axis == 0 ? 0.0F : 0.5F), base_x, weights_x);
        quadratic_weights((position.y - grid.origin_y) / grid.cell_size - (axis == 1 ? 0.0F : 0.5F), base_y, weights_y);
        quadratic_weights((position.z - grid.origin_z) / grid.cell_size - (axis == 2 ? 0.0F : 0.5F), base_z, weights_z);
    }

    __device__ inline float sample_face(const Grid grid, const int axis, const Vector3<float> position, const float* values) {
        const float gx = (position.x - grid.origin_x) / grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
        const float gy = (position.y - grid.origin_y) / grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
        const float gz = (position.z - grid.origin_z) / grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
        const int x0   = static_cast<int>(::cuda::std::floor(gx));
        const int y0   = static_cast<int>(::cuda::std::floor(gy));
        const int z0   = static_cast<int>(::cuda::std::floor(gz));
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        float result   = 0.0F;
        for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const int x = x0 + dx;
                    const int y = y0 + dy;
                    const int z = z0 + dz;
                    if (!valid_face(grid, axis, x, y, z)) continue;
                    const float weight = (dx == 0 ? 1.0F - tx : tx) * (dy == 0 ? 1.0F - ty : ty) * (dz == 0 ? 1.0F - tz : tz);
                    result += weight * values[face_index(grid, axis, x, y, z)];
                }
        return result;
    }

    __device__ inline Vector3<float> sample_velocity(const Grid grid, const Vector3<float> position, const simulation::VectorView<const float> velocity) {
        return {sample_face(grid, 0, position, velocity.x), sample_face(grid, 1, position, velocity.y), sample_face(grid, 2, position, velocity.z)};
    }
} // namespace physica::fluids::grid::device

#endif
