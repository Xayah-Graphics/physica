module;

#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>

module physica.fluids.gas.smoke.domain;

import std;

namespace physica::fluids::gas::smoke {
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
          cell_mask(stream, ::cuda::device_default_memory_pool(stream.device()), cell_count, ::cuda::no_init), collider_velocity(allocate_staggered_vector_field()), collider_density(allocate_scalar_field()), collider_temperature(allocate_scalar_field()) {
        std::vector<std::uint32_t> host_cell_mask(cell_count);
        std::vector<Vector3> host_collider_velocity(cell_count);
        std::vector<float> host_collider_density(cell_count);
        std::vector<float> host_collider_temperature(cell_count);
        const std::uint32_t nx = configuration.resolution[0];
        const std::uint32_t ny = configuration.resolution[1];
        const std::uint32_t nz = configuration.resolution[2];
        for (std::uint32_t z = 0u; z < nz; ++z) {
            for (std::uint32_t y = 0u; y < ny; ++y) {
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    const std::size_t index = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                    const Vector3 point{(static_cast<float>(x) + 0.5F) * configuration.cell_size, (static_cast<float>(y) + 0.5F) * configuration.cell_size, (static_cast<float>(z) + 0.5F) * configuration.cell_size};
                    for (const Collider& collider : configuration.colliders) {
                        const bool inside = std::visit([point](const auto& shape) { return contains(shape, point); }, collider.shape);
                        if (!inside) continue;
                        host_cell_mask[index]            = 1u;
                        host_collider_velocity[index]    = collider.velocity;
                        host_collider_density[index]     = collider.density;
                        host_collider_temperature[index] = collider.temperature;
                    }
                }
            }
        }
        const std::array pressure_faces{configuration.pressure_boundary.x_min, configuration.pressure_boundary.x_max, configuration.pressure_boundary.y_min, configuration.pressure_boundary.y_max, configuration.pressure_boundary.z_min, configuration.pressure_boundary.z_max};
        const bool fixed_pressure = std::ranges::any_of(pressure_faces, [](const ScalarBoundaryFace& face) { return face.mode == ScalarBoundaryMode::fixed_value; });
        if (fixed_pressure) pressure_anchor = static_cast<std::uint32_t>(cell_count);
        else while (host_cell_mask[pressure_anchor] != 0u) ++pressure_anchor;
        ::cuda::copy_bytes(stream, host_cell_mask, cell_mask);

        std::vector<float> x_face_velocity(face_counts[0]);
        std::vector<float> y_face_velocity(face_counts[1]);
        std::vector<float> z_face_velocity(face_counts[2]);
        for (std::uint32_t z = 0u; z < nz; ++z) {
            for (std::uint32_t y = 0u; y < ny; ++y) {
                for (std::uint32_t x = 0u; x <= nx; ++x) {
                    float sum{};
                    float count{};
                    if (x > 0u) {
                        const std::size_t cell = x - 1u + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (host_cell_mask[cell] != 0u) {
                            sum += host_collider_velocity[cell].x;
                            count += 1.0F;
                        }
                    }
                    if (x < nx) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (host_cell_mask[cell] != 0u) {
                            sum += host_collider_velocity[cell].x;
                            count += 1.0F;
                        }
                    }
                    x_face_velocity[x + static_cast<std::size_t>(nx + 1u) * (y + static_cast<std::size_t>(ny) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
            }
        }
        for (std::uint32_t z = 0u; z < nz; ++z) {
            for (std::uint32_t y = 0u; y <= ny; ++y) {
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    float sum{};
                    float count{};
                    if (y > 0u) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y - 1u + static_cast<std::size_t>(ny) * z);
                        if (host_cell_mask[cell] != 0u) {
                            sum += host_collider_velocity[cell].y;
                            count += 1.0F;
                        }
                    }
                    if (y < ny) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (host_cell_mask[cell] != 0u) {
                            sum += host_collider_velocity[cell].y;
                            count += 1.0F;
                        }
                    }
                    y_face_velocity[x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny + 1u) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
            }
        }
        for (std::uint32_t z = 0u; z <= nz; ++z) {
            for (std::uint32_t y = 0u; y < ny; ++y) {
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    float sum{};
                    float count{};
                    if (z > 0u) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * (z - 1u));
                        if (host_cell_mask[cell] != 0u) {
                            sum += host_collider_velocity[cell].z;
                            count += 1.0F;
                        }
                    }
                    if (z < nz) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (host_cell_mask[cell] != 0u) {
                            sum += host_collider_velocity[cell].z;
                            count += 1.0F;
                        }
                    }
                    z_face_velocity[x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
            }
        }
        ::cuda::copy_bytes(stream, x_face_velocity, collider_velocity.x);
        ::cuda::copy_bytes(stream, y_face_velocity, collider_velocity.y);
        ::cuda::copy_bytes(stream, z_face_velocity, collider_velocity.z);
        ::cuda::copy_bytes(stream, host_collider_density, collider_density.values);
        ::cuda::copy_bytes(stream, host_collider_temperature, collider_temperature.values);
        stream.sync();
    }

    ScalarField Domain::allocate_scalar_field() const {
        return {.values = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), cell_count, ::cuda::no_init}};
    }

    CenteredVectorField Domain::allocate_centered_vector_field() const {
        return {.x = allocate_scalar_field(), .y = allocate_scalar_field(), .z = allocate_scalar_field()};
    }

    StaggeredVectorField Domain::allocate_staggered_vector_field() const {
        return {
            .x = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[0], ::cuda::no_init},
            .y = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[1], ::cuda::no_init},
            .z = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[2], ::cuda::no_init},
        };
    }

    ScalarAdjointField Domain::allocate_scalar_adjoint_field() const {
        return {.values = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), cell_count, ::cuda::no_init}};
    }

    CenteredVectorAdjointField Domain::allocate_centered_vector_adjoint_field() const {
        return {.x = allocate_scalar_adjoint_field(), .y = allocate_scalar_adjoint_field(), .z = allocate_scalar_adjoint_field()};
    }

    StaggeredVectorAdjointField Domain::allocate_staggered_vector_adjoint_field() const {
        return {
            .x = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[0], ::cuda::no_init},
            .y = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[1], ::cuda::no_init},
            .z = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[2], ::cuda::no_init},
        };
    }

    void Domain::clear(ScalarField& field) const {
        ::cuda::fill_bytes(stream, field.values, 0u);
    }

    void Domain::clear(CenteredVectorField& field) const {
        clear(field.x);
        clear(field.y);
        clear(field.z);
    }

    void Domain::clear(StaggeredVectorField& field) const {
        ::cuda::fill_bytes(stream, field.x, 0u);
        ::cuda::fill_bytes(stream, field.y, 0u);
        ::cuda::fill_bytes(stream, field.z, 0u);
    }

    void Domain::clear(ScalarAdjointField& field) const {
        ::cuda::fill_bytes(stream, field.values, 0u);
    }

    void Domain::clear(CenteredVectorAdjointField& field) const {
        clear(field.x);
        clear(field.y);
        clear(field.z);
    }

    void Domain::clear(StaggeredVectorAdjointField& field) const {
        ::cuda::fill_bytes(stream, field.x, 0u);
        ::cuda::fill_bytes(stream, field.y, 0u);
        ::cuda::fill_bytes(stream, field.z, 0u);
    }

    void Domain::copy(const ScalarField& source, ScalarField& destination) const {
        ::cuda::copy_bytes(stream, source.values, destination.values);
    }

    void Domain::copy(const StaggeredVectorField& source, StaggeredVectorField& destination) const {
        ::cuda::copy_bytes(stream, source.x, destination.x);
        ::cuda::copy_bytes(stream, source.y, destination.y);
        ::cuda::copy_bytes(stream, source.z, destination.z);
    }

    void Domain::copy(const ScalarAdjointField& source, ScalarAdjointField& destination) const {
        ::cuda::copy_bytes(stream, source.values, destination.values);
    }

    void Domain::copy(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) const {
        ::cuda::copy_bytes(stream, source.x, destination.x);
        ::cuda::copy_bytes(stream, source.y, destination.y);
        ::cuda::copy_bytes(stream, source.z, destination.z);
    }

    void Domain::accumulate(const ScalarAdjointField& source, ScalarAdjointField& destination) const {
        cuda_detail::accumulate(stream, source.values.data(), destination.values.data(), source.values.size());
    }

    void Domain::accumulate(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) const {
        cuda_detail::accumulate(stream, source.x.data(), destination.x.data(), source.x.size());
        cuda_detail::accumulate(stream, source.y.data(), destination.y.data(), source.y.size());
        cuda_detail::accumulate(stream, source.z.data(), destination.z.data(), source.z.size());
    }
} // namespace physica::fluids::gas::smoke
