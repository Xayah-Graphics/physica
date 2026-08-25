#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_SAMPLING_CUH
#define PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_SAMPLING_CUH

#include "../detail/cuda/device.cuh"

namespace physica::fluids::gas::operators::cuda_backend {
    namespace {
        struct Sample final {
            float value;
            detail::cuda::Vector gradient;
        };

        struct Trace final {
            detail::cuda::Vector position;
            detail::cuda::Vector derivative;
        };

        __device__ std::uint64_t mapped_cell_index(int x, int y, int z, const detail::cuda::Grid grid, const bool periodic_x, const bool periodic_y, const bool periodic_z) {
            x = detail::cuda::map_coordinate(x, grid.nx, periodic_x, grid.nx);
            y = detail::cuda::map_coordinate(y, grid.ny, periodic_y, grid.ny);
            z = detail::cuda::map_coordinate(z, grid.nz, periodic_z, grid.nz);
            return detail::cuda::index3(x, y, z, grid.nx, grid.ny);
        }

        __device__ float load_scalar(const float* values, const int x, const int y, const int z, const detail::cuda::Grid grid, const detail::cuda::ScalarBoundaryData boundary) {
            if (x < 0 && boundary.modes[0] == 0u) return boundary.values[0];
            if (x >= grid.nx && boundary.modes[1] == 0u) return boundary.values[1];
            if (y < 0 && boundary.modes[2] == 0u) return boundary.values[2];
            if (y >= grid.ny && boundary.modes[3] == 0u) return boundary.values[3];
            if (z < 0 && boundary.modes[4] == 0u) return boundary.values[4];
            if (z >= grid.nz && boundary.modes[5] == 0u) return boundary.values[5];
            return values[mapped_cell_index(x, y, z, grid, detail::cuda::periodic(boundary, 0), detail::cuda::periodic(boundary, 1), detail::cuda::periodic(boundary, 2))];
        }

        __device__ Sample sample_scalar(const float* values, const detail::cuda::Vector position, const detail::cuda::Grid grid, const detail::cuda::ScalarBoundaryData boundary) {
            const float gx = position.x / grid.cell_size - 0.5F;
            const float gy = position.y / grid.cell_size - 0.5F;
            const float gz = position.z / grid.cell_size - 0.5F;
            const int x0   = static_cast<int>(floorf(gx));
            const int y0   = static_cast<int>(floorf(gy));
            const int z0   = static_cast<int>(floorf(gz));
            const float tx = gx - x0;
            const float ty = gy - y0;
            const float tz = gz - z0;
            float values8[2][2][2];
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) values8[dz][dy][dx] = load_scalar(values, x0 + dx, y0 + dy, z0 + dz, grid, boundary);
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
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) result.gradient.z += wx[dx] * wy[dy] * (values8[1][dy][dx] - values8[0][dy][dx]) / grid.cell_size;
            return result;
        }

        __device__ Sample sample_face(const float* values, const int axis, const detail::cuda::Vector position, const detail::cuda::Grid grid, const detail::cuda::VelocityBoundaryData boundary) {
            const float gx = position.x / grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
            const float gy = position.y / grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
            const float gz = position.z / grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
            const int x0   = static_cast<int>(floorf(gx));
            const int y0   = static_cast<int>(floorf(gy));
            const int z0   = static_cast<int>(floorf(gz));
            const float tx = gx - x0;
            const float ty = gy - y0;
            const float tz = gz - z0;
            float values8[2][2][2];
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) values8[dz][dy][dx] = detail::cuda::load_face(values, axis, x0 + dx, y0 + dy, z0 + dz, grid, boundary);
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
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) result.gradient.z += wx[dx] * wy[dy] * (values8[1][dy][dx] - values8[0][dy][dx]) / grid.cell_size;
            return result;
        }

        __device__ void scatter_scalar(double* values, const detail::cuda::Vector position, const double adjoint, const detail::cuda::Grid grid, const detail::cuda::ScalarBoundaryData boundary) {
            const float gx = position.x / grid.cell_size - 0.5F;
            const float gy = position.y / grid.cell_size - 0.5F;
            const float gz = position.z / grid.cell_size - 0.5F;
            const int x0   = static_cast<int>(floorf(gx));
            const int y0   = static_cast<int>(floorf(gy));
            const int z0   = static_cast<int>(floorf(gz));
            const double wx[2]{1.0 - (gx - x0), gx - x0};
            const double wy[2]{1.0 - (gy - y0), gy - y0};
            const double wz[2]{1.0 - (gz - z0), gz - z0};
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int x = x0 + dx;
                        const int y = y0 + dy;
                        const int z = z0 + dz;
                        if ((x < 0 && boundary.modes[0] == 0u) || (x >= grid.nx && boundary.modes[1] == 0u) || (y < 0 && boundary.modes[2] == 0u) || (y >= grid.ny && boundary.modes[3] == 0u) || (z < 0 && boundary.modes[4] == 0u) || (z >= grid.nz && boundary.modes[5] == 0u)) continue;
                        atomicAdd(values + mapped_cell_index(x, y, z, grid, detail::cuda::periodic(boundary, 0), detail::cuda::periodic(boundary, 1), detail::cuda::periodic(boundary, 2)), adjoint * wx[dx] * wy[dy] * wz[dz]);
                    }
        }

        __device__ void scatter_face(double* values, const int axis, const detail::cuda::Vector position, const double adjoint, const detail::cuda::Grid grid, const detail::cuda::VelocityBoundaryData boundary) {
            const float gx = position.x / grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
            const float gy = position.y / grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
            const float gz = position.z / grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
            const int x0   = static_cast<int>(floorf(gx));
            const int y0   = static_cast<int>(floorf(gy));
            const int z0   = static_cast<int>(floorf(gz));
            const double wx[2]{1.0 - (gx - x0), gx - x0};
            const double wy[2]{1.0 - (gy - y0), gy - y0};
            const double wz[2]{1.0 - (gz - z0), gz - z0};
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int coordinates[3]{x0 + dx, y0 + dy, z0 + dz};
                        bool constant = false;
                        for (int dimension = 0; dimension < 3; ++dimension) {
                            const int size = detail::cuda::extent(grid, axis, dimension);
                            if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
                            const int face = 2 * dimension + (coordinates[dimension] >= size);
                            if (boundary.modes[face] == 0u || (boundary.modes[face] == 2u && axis == dimension)) constant = true;
                        }
                        if (!constant) atomicAdd(values + detail::cuda::mapped_face_index(coordinates[0], coordinates[1], coordinates[2], grid, axis, boundary), adjoint * wx[dx] * wy[dy] * wz[dz]);
                    }
        }

        __device__ detail::cuda::Vector sample_velocity_value(const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::Vector position, const detail::cuda::Grid grid, const detail::cuda::VelocityBoundaryData boundary) {
            return {sample_face(velocity.x, 0, position, grid, boundary).value, sample_face(velocity.y, 1, position, grid, boundary).value, sample_face(velocity.z, 2, position, grid, boundary).value};
        }

        __device__ bool solid_at(detail::cuda::Vector position, const std::uint32_t* collider_ids, const detail::cuda::Grid grid, const detail::cuda::VelocityBoundaryData boundary) {
            if (detail::cuda::periodic(boundary, 0)) position.x -= floorf(position.x / (grid.nx * grid.cell_size)) * grid.nx * grid.cell_size;
            if (detail::cuda::periodic(boundary, 1)) position.y -= floorf(position.y / (grid.ny * grid.cell_size)) * grid.ny * grid.cell_size;
            if (detail::cuda::periodic(boundary, 2)) position.z -= floorf(position.z / (grid.nz * grid.cell_size)) * grid.nz * grid.cell_size;
            const int x = max(0, min(grid.nx - 1, static_cast<int>(floorf(position.x / grid.cell_size))));
            const int y = max(0, min(grid.ny - 1, static_cast<int>(floorf(position.y / grid.cell_size))));
            const int z = max(0, min(grid.nz - 1, static_cast<int>(floorf(position.z / grid.cell_size))));
            return collider_ids[detail::cuda::index3(x, y, z, grid.nx, grid.ny)] != 0u;
        }

        __device__ Trace trace_rk2(const detail::cuda::Vector start, const detail::cuda::StaggeredVectorView<const float> velocity, const std::uint32_t* collider_ids, const detail::cuda::Grid grid, const detail::cuda::VelocityBoundaryData boundary) {
            const detail::cuda::Vector value0 = sample_velocity_value(velocity, start, grid, boundary);
            const detail::cuda::Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const detail::cuda::Vector value1 = sample_velocity_value(velocity, midpoint, grid, boundary);
            detail::cuda::Vector raw{start.x - grid.time_step * value1.x, start.y - grid.time_step * value1.y, start.z - grid.time_step * value1.z};
            detail::cuda::Vector derivative{1.0F, 1.0F, 1.0F};
            const detail::cuda::Vector maximum{grid.nx * grid.cell_size, grid.ny * grid.cell_size, grid.nz * grid.cell_size};
            if (!detail::cuda::periodic(boundary, 0) && (raw.x < 0.0F || raw.x > maximum.x)) {
                raw.x        = fminf(fmaxf(raw.x, 0.0F), maximum.x);
                derivative.x = 0.0F;
            }
            if (!detail::cuda::periodic(boundary, 1) && (raw.y < 0.0F || raw.y > maximum.y)) {
                raw.y        = fminf(fmaxf(raw.y, 0.0F), maximum.y);
                derivative.y = 0.0F;
            }
            if (!detail::cuda::periodic(boundary, 2) && (raw.z < 0.0F || raw.z > maximum.z)) {
                raw.z        = fminf(fmaxf(raw.z, 0.0F), maximum.z);
                derivative.z = 0.0F;
            }
            if (!solid_at(raw, collider_ids, grid, boundary)) return {raw, derivative};
            float lo = 0.0F;
            float hi = 1.0F;
            for (int iteration = 0; iteration < 8; ++iteration) {
                const float fraction = 0.5F * (lo + hi);
                const detail::cuda::Vector test{start.x + fraction * (raw.x - start.x), start.y + fraction * (raw.y - start.y), start.z + fraction * (raw.z - start.z)};
                if (solid_at(test, collider_ids, grid, boundary)) hi = fraction;
                else lo = fraction;
            }
            return {{start.x + lo * (raw.x - start.x), start.y + lo * (raw.y - start.y), start.z + lo * (raw.z - start.z)}, {lo * derivative.x, lo * derivative.y, lo * derivative.z}};
        }

    } // namespace
} // namespace physica::fluids::gas::operators::cuda_backend

#endif
