module;

#include <physica/cuda.h>

module physica.fluids.gas.domain;

import std;

namespace physica::fluids::gas {
    namespace {
        bool contains(const Ellipsoid& ellipsoid, const Vector3<float> point) {
            const float x = (point.x - ellipsoid.center.x) / ellipsoid.radius.x;
            const float y = (point.y - ellipsoid.center.y) / ellipsoid.radius.y;
            const float z = (point.z - ellipsoid.center.z) / ellipsoid.radius.z;
            return x * x + y * y + z * z <= 1.0F;
        }

    } // namespace

    Domain::Domain(DomainConfiguration next_configuration, const ::cuda::stream_ref source_stream) : configuration(std::move(next_configuration)), grid(configuration.grid, source_stream), collider_ids(grid.allocate_cell_field<std::uint32_t>()), collider_velocity(grid.allocate_mac_field<float>()), collider_indices(grid.cell_count) {
        std::vector<Vector3<float>> collider_cell_velocity(grid.cell_count);
        const std::uint32_t nx = configuration.grid.resolution[0];
        const std::uint32_t ny = configuration.grid.resolution[1];
        const std::uint32_t nz = configuration.grid.resolution[2];
        for (std::uint32_t z = 0u; z < nz; ++z)
            for (std::uint32_t y = 0u; y < ny; ++y)
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    const std::size_t index = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                    const Vector3<float> point{
                        configuration.grid.origin.x + (static_cast<float>(x) + 0.5F) * configuration.grid.cell_size,
                        configuration.grid.origin.y + (static_cast<float>(y) + 0.5F) * configuration.grid.cell_size,
                        configuration.grid.origin.z + (static_cast<float>(z) + 0.5F) * configuration.grid.cell_size,
                    };
                    for (std::uint32_t collider_index = 0u; collider_index < configuration.colliders.size(); ++collider_index) {
                        const Collider& collider = configuration.colliders[collider_index];
                        if (!std::visit([point](const auto& shape) { return contains(shape, point); }, collider.shape)) continue;
                        collider_indices[index]       = collider_index + 1u;
                        collider_cell_velocity[index] = collider.velocity;
                    }
                }
        while (collider_indices[first_fluid_cell] != 0u) ++first_fluid_cell;
        ::cuda::copy_bytes(grid.stream, collider_indices, collider_ids.values);

        std::vector<float> x_velocity(grid.face_counts[0]);
        std::vector<float> y_velocity(grid.face_counts[1]);
        std::vector<float> z_velocity(grid.face_counts[2]);
        for (std::uint32_t z = 0u; z < nz; ++z)
            for (std::uint32_t y = 0u; y < ny; ++y)
                for (std::uint32_t x = 0u; x <= nx; ++x) {
                    float sum{};
                    float count{};
                    if (x > 0u) {
                        const std::size_t cell = x - 1u + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (collider_indices[cell] != 0u) {
                            sum += collider_cell_velocity[cell].x;
                            count += 1.0F;
                        }
                    }
                    if (x < nx) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (collider_indices[cell] != 0u) {
                            sum += collider_cell_velocity[cell].x;
                            count += 1.0F;
                        }
                    }
                    x_velocity[x + static_cast<std::size_t>(nx + 1u) * (y + static_cast<std::size_t>(ny) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
        for (std::uint32_t z = 0u; z < nz; ++z)
            for (std::uint32_t y = 0u; y <= ny; ++y)
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    float sum{};
                    float count{};
                    if (y > 0u) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y - 1u + static_cast<std::size_t>(ny) * z);
                        if (collider_indices[cell] != 0u) {
                            sum += collider_cell_velocity[cell].y;
                            count += 1.0F;
                        }
                    }
                    if (y < ny) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (collider_indices[cell] != 0u) {
                            sum += collider_cell_velocity[cell].y;
                            count += 1.0F;
                        }
                    }
                    y_velocity[x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny + 1u) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
        for (std::uint32_t z = 0u; z <= nz; ++z)
            for (std::uint32_t y = 0u; y < ny; ++y)
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    float sum{};
                    float count{};
                    if (z > 0u) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * (z - 1u));
                        if (collider_indices[cell] != 0u) {
                            sum += collider_cell_velocity[cell].z;
                            count += 1.0F;
                        }
                    }
                    if (z < nz) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (collider_indices[cell] != 0u) {
                            sum += collider_cell_velocity[cell].z;
                            count += 1.0F;
                        }
                    }
                    z_velocity[x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
        ::cuda::copy_bytes(grid.stream, x_velocity, collider_velocity.x);
        ::cuda::copy_bytes(grid.stream, y_velocity, collider_velocity.y);
        ::cuda::copy_bytes(grid.stream, z_velocity, collider_velocity.z);
        grid.stream.sync();
    }

    simulation::ScalarField<float> Domain::allocate_collider_field(const std::span<const float> values) const {
        std::vector<float> host_values(grid.cell_count);
        for (std::size_t cell = 0u; cell < grid.cell_count; ++cell)
            if (collider_indices[cell] != 0u) host_values[cell] = values[collider_indices[cell] - 1u];
        simulation::ScalarField<float> result = grid.allocate_cell_field<float>();
        ::cuda::copy_bytes(grid.stream, host_values, result.values);
        return result;
    }

} // namespace physica::fluids::gas
