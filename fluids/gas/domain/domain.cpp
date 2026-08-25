module;

#include <physica/cuda.h>

module physica.fluids.gas.domain;

import std;

namespace physica::fluids::gas {
    namespace {
        bool contains(const Ellipsoid& ellipsoid, const Vector3 point) {
            const float x = (point.x - ellipsoid.center.x) / ellipsoid.radius.x;
            const float y = (point.y - ellipsoid.center.y) / ellipsoid.radius.y;
            const float z = (point.z - ellipsoid.center.z) / ellipsoid.radius.z;
            return x * x + y * y + z * z <= 1.0F;
        }

        bool contains(const Box& box, const Vector3 point) {
            return std::abs(point.x - box.center.x) <= box.half_extent.x && std::abs(point.y - box.center.y) <= box.half_extent.y && std::abs(point.z - box.center.z) <= box.half_extent.z;
        }
    } // namespace

    Domain::Domain(DomainConfiguration next_configuration, const ::cuda::stream_ref source_stream)
        : configuration(std::move(next_configuration)), stream(source_stream), cell_count(static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * configuration.resolution[2]),
          face_counts{
              static_cast<std::size_t>(configuration.resolution[0] + 1u) * configuration.resolution[1] * configuration.resolution[2],
              static_cast<std::size_t>(configuration.resolution[0]) * (configuration.resolution[1] + 1u) * configuration.resolution[2],
              static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * (configuration.resolution[2] + 1u),
          },
          collider_ids(allocate_cell_field<std::uint32_t>()), collider_velocity(allocate_staggered_vector_field<float>()), collider_indices(cell_count) {
        std::vector<Vector3> collider_cell_velocity(cell_count);
        const std::uint32_t nx = configuration.resolution[0];
        const std::uint32_t ny = configuration.resolution[1];
        const std::uint32_t nz = configuration.resolution[2];
        for (std::uint32_t z = 0u; z < nz; ++z)
            for (std::uint32_t y = 0u; y < ny; ++y)
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    const std::size_t index = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                    const Vector3 point{(static_cast<float>(x) + 0.5F) * configuration.cell_size, (static_cast<float>(y) + 0.5F) * configuration.cell_size, (static_cast<float>(z) + 0.5F) * configuration.cell_size};
                    for (std::uint32_t collider_index = 0u; collider_index < configuration.colliders.size(); ++collider_index) {
                        const Collider& collider = configuration.colliders[collider_index];
                        if (!std::visit([point](const auto& shape) { return contains(shape, point); }, collider.shape)) continue;
                        collider_indices[index]       = collider_index + 1u;
                        collider_cell_velocity[index] = collider.velocity;
                    }
                }
        while (collider_indices[first_fluid_cell] != 0u) ++first_fluid_cell;
        ::cuda::copy_bytes(stream, collider_indices, collider_ids.values);

        std::vector<float> x_velocity(face_counts[0]);
        std::vector<float> y_velocity(face_counts[1]);
        std::vector<float> z_velocity(face_counts[2]);
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
        ::cuda::copy_bytes(stream, x_velocity, collider_velocity.x);
        ::cuda::copy_bytes(stream, y_velocity, collider_velocity.y);
        ::cuda::copy_bytes(stream, z_velocity, collider_velocity.z);
        stream.sync();
    }

    CellField<float> Domain::allocate_collider_field(const std::span<const float> values) const {
        std::vector<float> host_values(cell_count);
        for (std::size_t cell = 0u; cell < cell_count; ++cell)
            if (collider_indices[cell] != 0u) host_values[cell] = values[collider_indices[cell] - 1u];
        CellField<float> result = allocate_cell_field<float>();
        ::cuda::copy_bytes(stream, host_values, result.values);
        return result;
    }

} // namespace physica::fluids::gas
