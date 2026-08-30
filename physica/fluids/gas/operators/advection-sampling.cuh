#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_SAMPLING_CUH
#define PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_SAMPLING_CUH

#include <fluids/gas/device.cuh>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>

namespace physica::fluids::gas::operators::kernels {
    namespace {
        struct Sample final {
            float value;
            Vector3<float> gradient;
        };

        struct Trace final {
            Vector3<float> position;
            Vector3<float> derivative;
        };

        __device__ std::uint64_t mapped_cell_index(int x, int y, int z, const device::Discretization grid, const bool periodic_x, const bool periodic_y, const bool periodic_z) {
            x = device::map_coordinate(x, grid.grid.nx, periodic_x, grid.grid.nx);
            y = device::map_coordinate(y, grid.grid.ny, periodic_y, grid.grid.ny);
            z = device::map_coordinate(z, grid.grid.nz, periodic_z, grid.grid.nz);
            return fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny);
        }

        __device__ float load_scalar(const float* values, const int x, const int y, const int z, const device::Discretization grid, const device::ScalarBoundary boundary) {
            if (x < 0 && boundary.faces[0].mode == 0u) return boundary.faces[0].value;
            if (x >= grid.grid.nx && boundary.faces[1].mode == 0u) return boundary.faces[1].value;
            if (y < 0 && boundary.faces[2].mode == 0u) return boundary.faces[2].value;
            if (y >= grid.grid.ny && boundary.faces[3].mode == 0u) return boundary.faces[3].value;
            if (z < 0 && boundary.faces[4].mode == 0u) return boundary.faces[4].value;
            if (z >= grid.grid.nz && boundary.faces[5].mode == 0u) return boundary.faces[5].value;
            return values[mapped_cell_index(x, y, z, grid, device::periodic(boundary, 0), device::periodic(boundary, 1), device::periodic(boundary, 2))];
        }

        __device__ Sample sample_scalar(const float* values, const Vector3<float> position, const device::Discretization grid, const device::ScalarBoundary boundary) {
            const float gx = position.x / grid.grid.cell_size - 0.5F;
            const float gy = position.y / grid.grid.cell_size - 0.5F;
            const float gz = position.z / grid.grid.cell_size - 0.5F;
            const int x0   = static_cast<int>(::cuda::std::floor(gx));
            const int y0   = static_cast<int>(::cuda::std::floor(gy));
            const int z0   = static_cast<int>(::cuda::std::floor(gz));
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
                    result.gradient.x += wy[dy] * wz[dz] * (values8[dz][dy][1] - values8[dz][dy][0]) / grid.grid.cell_size;
                }
                for (int dx = 0; dx < 2; ++dx) result.gradient.y += wx[dx] * wz[dz] * (values8[dz][1][dx] - values8[dz][0][dx]) / grid.grid.cell_size;
            }
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) result.gradient.z += wx[dx] * wy[dy] * (values8[1][dy][dx] - values8[0][dy][dx]) / grid.grid.cell_size;
            return result;
        }

        __device__ Sample sample_face(const float* values, const int axis, const Vector3<float> position, const device::Discretization grid, const device::VelocityBoundary boundary) {
            const float gx = position.x / grid.grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
            const float gy = position.y / grid.grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
            const float gz = position.z / grid.grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
            const int x0   = static_cast<int>(::cuda::std::floor(gx));
            const int y0   = static_cast<int>(::cuda::std::floor(gy));
            const int z0   = static_cast<int>(::cuda::std::floor(gz));
            const float tx = gx - x0;
            const float ty = gy - y0;
            const float tz = gz - z0;
            float values8[2][2][2];
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) values8[dz][dy][dx] = device::load_face(values, axis, x0 + dx, y0 + dy, z0 + dz, grid, boundary);
            const float wx[2]{1.0F - tx, tx};
            const float wy[2]{1.0F - ty, ty};
            const float wz[2]{1.0F - tz, tz};
            Sample result{};
            for (int dz = 0; dz < 2; ++dz) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) result.value += wx[dx] * wy[dy] * wz[dz] * values8[dz][dy][dx];
                    result.gradient.x += wy[dy] * wz[dz] * (values8[dz][dy][1] - values8[dz][dy][0]) / grid.grid.cell_size;
                }
                for (int dx = 0; dx < 2; ++dx) result.gradient.y += wx[dx] * wz[dz] * (values8[dz][1][dx] - values8[dz][0][dx]) / grid.grid.cell_size;
            }
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) result.gradient.z += wx[dx] * wy[dy] * (values8[1][dy][dx] - values8[0][dy][dx]) / grid.grid.cell_size;
            return result;
        }

        __device__ void scatter_scalar(double* values, const Vector3<float> position, const double adjoint, const device::Discretization grid, const device::ScalarBoundary boundary) {
            const float gx = position.x / grid.grid.cell_size - 0.5F;
            const float gy = position.y / grid.grid.cell_size - 0.5F;
            const float gz = position.z / grid.grid.cell_size - 0.5F;
            const int x0   = static_cast<int>(::cuda::std::floor(gx));
            const int y0   = static_cast<int>(::cuda::std::floor(gy));
            const int z0   = static_cast<int>(::cuda::std::floor(gz));
            const double wx[2]{1.0 - (gx - x0), gx - x0};
            const double wy[2]{1.0 - (gy - y0), gy - y0};
            const double wz[2]{1.0 - (gz - z0), gz - z0};
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int x = x0 + dx;
                        const int y = y0 + dy;
                        const int z = z0 + dz;
                        if ((x < 0 && boundary.faces[0].mode == 0u) || (x >= grid.grid.nx && boundary.faces[1].mode == 0u) || (y < 0 && boundary.faces[2].mode == 0u) || (y >= grid.grid.ny && boundary.faces[3].mode == 0u) || (z < 0 && boundary.faces[4].mode == 0u) || (z >= grid.grid.nz && boundary.faces[5].mode == 0u)) continue;
                        atomicAdd(values + mapped_cell_index(x, y, z, grid, device::periodic(boundary, 0), device::periodic(boundary, 1), device::periodic(boundary, 2)), adjoint * wx[dx] * wy[dy] * wz[dz]);
                    }
        }

        __device__ void scatter_face(double* values, const int axis, const Vector3<float> position, const double adjoint, const device::Discretization grid, const device::VelocityBoundary boundary) {
            const float gx = position.x / grid.grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
            const float gy = position.y / grid.grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
            const float gz = position.z / grid.grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
            const int x0   = static_cast<int>(::cuda::std::floor(gx));
            const int y0   = static_cast<int>(::cuda::std::floor(gy));
            const int z0   = static_cast<int>(::cuda::std::floor(gz));
            const double wx[2]{1.0 - (gx - x0), gx - x0};
            const double wy[2]{1.0 - (gy - y0), gy - y0};
            const double wz[2]{1.0 - (gz - z0), gz - z0};
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int coordinates[3]{x0 + dx, y0 + dy, z0 + dz};
                        bool constant = false;
                        for (int dimension = 0; dimension < 3; ++dimension) {
                            const int size = fluids::grid::device::extent(grid.grid, axis, dimension);
                            if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
                            const int face = 2 * dimension + (coordinates[dimension] >= size);
                            if (boundary.faces[face].mode == 0u || (boundary.faces[face].mode == 2u && axis == dimension)) constant = true;
                        }
                        if (!constant) atomicAdd(values + device::mapped_face_index(coordinates[0], coordinates[1], coordinates[2], grid, axis, boundary), adjoint * wx[dx] * wy[dy] * wz[dz]);
                    }
        }

        __device__ Vector3<float> sample_velocity_value(const simulation::VectorView<const float> velocity, const Vector3<float> position, const device::Discretization grid, const device::VelocityBoundary boundary) {
            return {sample_face(velocity.x, 0, position, grid, boundary).value, sample_face(velocity.y, 1, position, grid, boundary).value, sample_face(velocity.z, 2, position, grid, boundary).value};
        }

        __device__ bool solid_at(Vector3<float> position, const std::uint32_t* collider_ids, const device::Discretization grid, const device::VelocityBoundary boundary) {
            if (device::periodic(boundary, 0)) position.x -= ::cuda::std::floor(position.x / (grid.grid.nx * grid.grid.cell_size)) * grid.grid.nx * grid.grid.cell_size;
            if (device::periodic(boundary, 1)) position.y -= ::cuda::std::floor(position.y / (grid.grid.ny * grid.grid.cell_size)) * grid.grid.ny * grid.grid.cell_size;
            if (device::periodic(boundary, 2)) position.z -= ::cuda::std::floor(position.z / (grid.grid.nz * grid.grid.cell_size)) * grid.grid.nz * grid.grid.cell_size;
            const int x = ::cuda::std::clamp(static_cast<int>(::cuda::std::floor(position.x / grid.grid.cell_size)), 0, static_cast<int>(grid.grid.nx) - 1);
            const int y = ::cuda::std::clamp(static_cast<int>(::cuda::std::floor(position.y / grid.grid.cell_size)), 0, static_cast<int>(grid.grid.ny) - 1);
            const int z = ::cuda::std::clamp(static_cast<int>(::cuda::std::floor(position.z / grid.grid.cell_size)), 0, static_cast<int>(grid.grid.nz) - 1);
            return collider_ids[fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny)] != 0u;
        }

        __device__ Trace trace_rk2(const Vector3<float> start, const simulation::VectorView<const float> velocity, const std::uint32_t* collider_ids, const device::Discretization grid, const device::VelocityBoundary boundary) {
            const Vector3<float> value0 = sample_velocity_value(velocity, start, grid, boundary);
            const Vector3<float> midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Vector3<float> value1 = sample_velocity_value(velocity, midpoint, grid, boundary);
            Vector3<float> raw{start.x - grid.time_step * value1.x, start.y - grid.time_step * value1.y, start.z - grid.time_step * value1.z};
            Vector3<float> derivative{1.0F, 1.0F, 1.0F};
            const Vector3<float> maximum{grid.grid.nx * grid.grid.cell_size, grid.grid.ny * grid.grid.cell_size, grid.grid.nz * grid.grid.cell_size};
            if (!device::periodic(boundary, 0) && (raw.x < 0.0F || raw.x > maximum.x)) {
                raw.x        = ::cuda::std::min(::cuda::std::max(raw.x, 0.0F), maximum.x);
                derivative.x = 0.0F;
            }
            if (!device::periodic(boundary, 1) && (raw.y < 0.0F || raw.y > maximum.y)) {
                raw.y        = ::cuda::std::min(::cuda::std::max(raw.y, 0.0F), maximum.y);
                derivative.y = 0.0F;
            }
            if (!device::periodic(boundary, 2) && (raw.z < 0.0F || raw.z > maximum.z)) {
                raw.z        = ::cuda::std::min(::cuda::std::max(raw.z, 0.0F), maximum.z);
                derivative.z = 0.0F;
            }
            if (!solid_at(raw, collider_ids, grid, boundary)) return {raw, derivative};
            float lo = 0.0F;
            float hi = 1.0F;
            for (int iteration = 0; iteration < 8; ++iteration) {
                const float fraction = 0.5F * (lo + hi);
                const Vector3<float> test{start.x + fraction * (raw.x - start.x), start.y + fraction * (raw.y - start.y), start.z + fraction * (raw.z - start.z)};
                if (solid_at(test, collider_ids, grid, boundary)) hi = fraction;
                else lo = fraction;
            }
            return {{start.x + lo * (raw.x - start.x), start.y + lo * (raw.y - start.y), start.z + lo * (raw.z - start.z)}, {lo * derivative.x, lo * derivative.y, lo * derivative.z}};
        }

    } // namespace
} // namespace physica::fluids::gas::operators::kernels

#endif
