#ifndef PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_SAMPLING_CUH
#define PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_SAMPLING_CUH

#include "../domain/device.cuh"

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    struct Sample final {
        float value;
        Vector gradient;
    };

    struct Trace final {
        Vector position;
        Vector derivative;
    };

    __device__ inline bool periodic(const VelocityBoundaryData boundary, const int dimension) {
        return boundary.modes[dimension * 2] == 3u && boundary.modes[dimension * 2 + 1] == 3u;
    }

    __device__ inline bool periodic(const ScalarBoundaryData boundary, const int dimension) {
        return boundary.modes[dimension * 2] == 2u && boundary.modes[dimension * 2 + 1] == 2u;
    }

    __device__ inline int map_coordinate(const int value, const int size, const bool is_periodic, const int period) {
        if (is_periodic) return wrap(value, period);
        return max(0, min(size - 1, value));
    }

    __device__ inline std::uint64_t mapped_cell_index(int x, int y, int z, const Grid grid, const bool periodic_x, const bool periodic_y, const bool periodic_z) {
        x = map_coordinate(x, grid.nx, periodic_x, grid.nx);
        y = map_coordinate(y, grid.ny, periodic_y, grid.ny);
        z = map_coordinate(z, grid.nz, periodic_z, grid.nz);
        return index3(x, y, z, grid.nx, grid.ny);
    }

    __device__ inline std::uint64_t mapped_face_index(int x, int y, int z, const Grid grid, const int axis, const VelocityBoundaryData boundary) {
        const int ex = extent(grid, axis, 0);
        const int ey = extent(grid, axis, 1);
        const int ez = extent(grid, axis, 2);
        x = map_coordinate(x, ex, periodic(boundary, 0), grid.nx);
        y = map_coordinate(y, ey, periodic(boundary, 1), grid.ny);
        z = map_coordinate(z, ez, periodic(boundary, 2), grid.nz);
        return index3(x, y, z, ex, ey);
    }

    __device__ inline float load_scalar(const float* values, const int x, const int y, const int z, const Grid grid, const ScalarBoundaryData boundary) {
        if (x < 0 && boundary.modes[0] == 0u) return boundary.values[0];
        if (x >= grid.nx && boundary.modes[1] == 0u) return boundary.values[1];
        if (y < 0 && boundary.modes[2] == 0u) return boundary.values[2];
        if (y >= grid.ny && boundary.modes[3] == 0u) return boundary.values[3];
        if (z < 0 && boundary.modes[4] == 0u) return boundary.values[4];
        if (z >= grid.nz && boundary.modes[5] == 0u) return boundary.values[5];
        return values[mapped_cell_index(x, y, z, grid, periodic(boundary, 0), periodic(boundary, 1), periodic(boundary, 2))];
    }

    __device__ inline float load_face(const float* values, const int axis, const int x, const int y, const int z, const Grid grid, const VelocityBoundaryData boundary) {
        const int coordinates[3]{x, y, z};
        for (int dimension = 0; dimension < 3; ++dimension) {
            const int size = extent(grid, axis, dimension);
            if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
            const int face = 2 * dimension + (coordinates[dimension] >= size);
            if (boundary.modes[face] == 0u || (boundary.modes[face] == 2u && axis == dimension)) return boundary.values[3 * face + axis];
        }
        return values[mapped_face_index(x, y, z, grid, axis, boundary)];
    }

    __device__ inline Sample sample_scalar(const float* values, const Vector position, const Grid grid, const ScalarBoundaryData boundary) {
        const float gx = position.x / grid.cell_size - 0.5F;
        const float gy = position.y / grid.cell_size - 0.5F;
        const float gz = position.z / grid.cell_size - 0.5F;
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const float tx = gx - x0;
        const float ty = gy - y0;
        const float tz = gz - z0;
        float values8[2][2][2];
        for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) values8[dz][dy][dx] = load_scalar(values, x0 + dx, y0 + dy, z0 + dz, grid, boundary);
        const float wx[2]{1.0F - tx, tx};
        const float wy[2]{1.0F - ty, ty};
        const float wz[2]{1.0F - tz, tz};
        Sample result{};
        for (int dz = 0; dz < 2; ++dz) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) result.value += wx[dx] * wy[dy] * wz[dz] * values8[dz][dy][dx];
                result.gradient.x += wy[dy] * wz[dz] * (values8[dz][dy][1] - values8[dz][dy][0]) / grid.cell_size;
            }
            for (int dx = 0; dx < 2; ++dx) result.gradient.y += wx[dx] * wz[dz] * (values8[dz][1][dx] - values8[dz][0][dx]) / grid.cell_size;
        }
        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) result.gradient.z += wx[dx] * wy[dy] * (values8[1][dy][dx] - values8[0][dy][dx]) / grid.cell_size;
        return result;
    }

    __device__ inline Sample sample_face(const float* values, const int axis, const Vector position, const Grid grid, const VelocityBoundaryData boundary) {
        const float gx = position.x / grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
        const float gy = position.y / grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
        const float gz = position.z / grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const float tx = gx - x0;
        const float ty = gy - y0;
        const float tz = gz - z0;
        float values8[2][2][2];
        for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) values8[dz][dy][dx] = load_face(values, axis, x0 + dx, y0 + dy, z0 + dz, grid, boundary);
        const float wx[2]{1.0F - tx, tx};
        const float wy[2]{1.0F - ty, ty};
        const float wz[2]{1.0F - tz, tz};
        Sample result{};
        for (int dz = 0; dz < 2; ++dz) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) result.value += wx[dx] * wy[dy] * wz[dz] * values8[dz][dy][dx];
                result.gradient.x += wy[dy] * wz[dz] * (values8[dz][dy][1] - values8[dz][dy][0]) / grid.cell_size;
            }
            for (int dx = 0; dx < 2; ++dx) result.gradient.y += wx[dx] * wz[dz] * (values8[dz][1][dx] - values8[dz][0][dx]) / grid.cell_size;
        }
        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) result.gradient.z += wx[dx] * wy[dy] * (values8[1][dy][dx] - values8[0][dy][dx]) / grid.cell_size;
        return result;
    }

    __device__ inline void scatter_scalar(double* values, const Vector position, const double adjoint, const Grid grid, const ScalarBoundaryData boundary) {
        const float gx = position.x / grid.cell_size - 0.5F;
        const float gy = position.y / grid.cell_size - 0.5F;
        const float gz = position.z / grid.cell_size - 0.5F;
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const double wx[2]{1.0 - (gx - x0), gx - x0};
        const double wy[2]{1.0 - (gy - y0), gy - y0};
        const double wz[2]{1.0 - (gz - z0), gz - z0};
        for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            const int x = x0 + dx;
            const int y = y0 + dy;
            const int z = z0 + dz;
            if ((x < 0 && boundary.modes[0] == 0u) || (x >= grid.nx && boundary.modes[1] == 0u) || (y < 0 && boundary.modes[2] == 0u) || (y >= grid.ny && boundary.modes[3] == 0u) || (z < 0 && boundary.modes[4] == 0u) || (z >= grid.nz && boundary.modes[5] == 0u)) continue;
            atomicAdd(values + mapped_cell_index(x, y, z, grid, periodic(boundary, 0), periodic(boundary, 1), periodic(boundary, 2)), adjoint * wx[dx] * wy[dy] * wz[dz]);
        }
    }

    __device__ inline void scatter_face(double* values, const int axis, const Vector position, const double adjoint, const Grid grid, const VelocityBoundaryData boundary) {
        const float gx = position.x / grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
        const float gy = position.y / grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
        const float gz = position.z / grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const double wx[2]{1.0 - (gx - x0), gx - x0};
        const double wy[2]{1.0 - (gy - y0), gy - y0};
        const double wz[2]{1.0 - (gz - z0), gz - z0};
        for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            const int coordinates[3]{x0 + dx, y0 + dy, z0 + dz};
            bool constant = false;
            for (int dimension = 0; dimension < 3; ++dimension) {
                const int size = extent(grid, axis, dimension);
                if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
                const int face = 2 * dimension + (coordinates[dimension] >= size);
                if (boundary.modes[face] == 0u || (boundary.modes[face] == 2u && axis == dimension)) constant = true;
            }
            if (!constant) atomicAdd(values + mapped_face_index(coordinates[0], coordinates[1], coordinates[2], grid, axis, boundary), adjoint * wx[dx] * wy[dy] * wz[dz]);
        }
    }

    __device__ inline Vector sample_velocity_value(const ConstStaggeredVectorView velocity, const Vector position, const Grid grid, const VelocityBoundaryData boundary) {
        return {sample_face(velocity.x, 0, position, grid, boundary).value, sample_face(velocity.y, 1, position, grid, boundary).value, sample_face(velocity.z, 2, position, grid, boundary).value};
    }

    __device__ inline Trace trace_rk2(const Vector start, const ConstStaggeredVectorView velocity, const Grid grid, const VelocityBoundaryData boundary) {
        const Vector value0 = sample_velocity_value(velocity, start, grid, boundary);
        const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
        const Vector value1 = sample_velocity_value(velocity, midpoint, grid, boundary);
        Vector raw{start.x - grid.time_step * value1.x, start.y - grid.time_step * value1.y, start.z - grid.time_step * value1.z};
        Vector derivative{1.0F, 1.0F, 1.0F};
        const Vector maximum{grid.nx * grid.cell_size, grid.ny * grid.cell_size, grid.nz * grid.cell_size};
        if (!periodic(boundary, 0) && (raw.x < 0.0F || raw.x > maximum.x)) {
            raw.x = fminf(fmaxf(raw.x, 0.0F), maximum.x);
            derivative.x = 0.0F;
        }
        if (!periodic(boundary, 1) && (raw.y < 0.0F || raw.y > maximum.y)) {
            raw.y = fminf(fmaxf(raw.y, 0.0F), maximum.y);
            derivative.y = 0.0F;
        }
        if (!periodic(boundary, 2) && (raw.z < 0.0F || raw.z > maximum.z)) {
            raw.z = fminf(fmaxf(raw.z, 0.0F), maximum.z);
            derivative.z = 0.0F;
        }
        return {raw, derivative};
    }

    __device__ inline Vector face_position(const int axis, const int x, const int y, const int z, const Grid grid) {
        return {(x + (axis == 0 ? 0.0F : 0.5F)) * grid.cell_size, (y + (axis == 1 ? 0.0F : 0.5F)) * grid.cell_size, (z + (axis == 2 ? 0.0F : 0.5F)) * grid.cell_size};
    }

    __device__ inline Vector cell_position(const int x, const int y, const int z, const Grid grid) {
        return {(x + 0.5F) * grid.cell_size, (y + 0.5F) * grid.cell_size, (z + 0.5F) * grid.cell_size};
    }
} // namespace physica::fluids::gas::keyframe_smoke::cuda_detail

#endif
