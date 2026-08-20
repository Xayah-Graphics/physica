module;

#include "interop.h"
#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>

module physica.deformables.cloth.domain;

import std;

namespace physica::deformables::cloth {
    Domain::Domain(DomainConfiguration next_configuration, const ::cuda::stream_ref source_stream)
        : configuration(std::move(next_configuration)), stream(source_stream), particle_count(configuration.rest_positions.size()) {}

    IndexField Domain::allocate_index_field(const std::size_t size) const {
        return {.values = ::cuda::device_buffer<std::uint32_t>{stream, ::cuda::device_default_memory_pool(stream.device()), size, ::cuda::no_init}};
    }

    ScalarField Domain::allocate_scalar_field(const std::size_t size) const {
        return {.values = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), size, ::cuda::no_init}};
    }

    ScalarAdjointField Domain::allocate_scalar_adjoint_field(const std::size_t size) const {
        return {.values = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), size, ::cuda::no_init}};
    }

    VectorField Domain::allocate_vector_field() const {
        return {.x = allocate_scalar_field(particle_count), .y = allocate_scalar_field(particle_count), .z = allocate_scalar_field(particle_count)};
    }

    VectorAdjointField Domain::allocate_vector_adjoint_field() const {
        return {.x = allocate_scalar_adjoint_field(particle_count), .y = allocate_scalar_adjoint_field(particle_count), .z = allocate_scalar_adjoint_field(particle_count)};
    }

    void Domain::clear(ScalarField& field) const {
        ::cuda::fill_bytes(stream, field.values, 0u);
    }

    void Domain::clear(ScalarAdjointField& field) const {
        ::cuda::fill_bytes(stream, field.values, 0u);
    }

    void Domain::clear(VectorField& field) const {
        clear(field.x);
        clear(field.y);
        clear(field.z);
    }

    void Domain::clear(VectorAdjointField& field) const {
        clear(field.x);
        clear(field.y);
        clear(field.z);
    }

    void Domain::copy(const VectorField& source, VectorField& destination) const {
        ::cuda::copy_bytes(stream, source.x.values, destination.x.values);
        ::cuda::copy_bytes(stream, source.y.values, destination.y.values);
        ::cuda::copy_bytes(stream, source.z.values, destination.z.values);
    }

    void Domain::copy(const VectorAdjointField& source, VectorAdjointField& destination) const {
        ::cuda::copy_bytes(stream, source.x.values, destination.x.values);
        ::cuda::copy_bytes(stream, source.y.values, destination.y.values);
        ::cuda::copy_bytes(stream, source.z.values, destination.z.values);
    }

    void Domain::accumulate(const VectorAdjointField& source, VectorAdjointField& destination) const {
        cuda_detail::accumulate(stream, cuda_detail::adjoint_field(source), cuda_detail::adjoint_field(destination), particle_count);
    }

    void Domain::upload(const std::span<const Vector3> source, VectorField& destination) const {
        std::vector<float> x(source.size());
        std::vector<float> y(source.size());
        std::vector<float> z(source.size());
        for (std::size_t index = 0uz; index < source.size(); ++index) {
            x[index] = source[index].x;
            y[index] = source[index].y;
            z[index] = source[index].z;
        }
        ::cuda::copy_bytes(stream, x, destination.x.values);
        ::cuda::copy_bytes(stream, y, destination.y.values);
        ::cuda::copy_bytes(stream, z, destination.z.values);
        stream.sync();
    }
} // namespace physica::deformables::cloth
