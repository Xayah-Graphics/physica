module;

#include <physica/cuda.h>

export module physica.deformables.cloth.domain;

import std;

export namespace physica::deformables::cloth {
    struct Vector3 final {
        float x;
        float y;
        float z;
    };

    struct Triangle final {
        std::uint32_t first;
        std::uint32_t second;
        std::uint32_t third;
    };

    struct DomainConfiguration final {
        std::vector<Vector3> rest_positions;
        std::vector<Triangle> triangles;
    };

    template <class Scalar>
    struct ScalarField final {
        ::cuda::device_buffer<Scalar> values;
    };

    template <class Scalar>
    struct VectorField final {
        ::cuda::device_buffer<Scalar> x;
        ::cuda::device_buffer<Scalar> y;
        ::cuda::device_buffer<Scalar> z;
    };

    struct Domain final {
        const DomainConfiguration configuration;
        ::cuda::stream_ref stream;
        const std::size_t particle_count;

        Domain(DomainConfiguration configuration, ::cuda::stream_ref stream);

        Domain(const Domain&)            = delete;
        Domain& operator=(const Domain&) = delete;
        Domain(Domain&&)                 = delete;
        Domain& operator=(Domain&&)      = delete;

        template <class Value>
        [[nodiscard]] ScalarField<Value> allocate_scalar_field(const std::size_t count) const {
            return {.values = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}};
        }

        template <class Value>
        [[nodiscard]] VectorField<Value> allocate_vector_field(const std::size_t count) const {
            return {
                .x = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .y = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .z = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
            };
        }

        template <class Value>
        void clear(ScalarField<Value>& field) const {
            ::cuda::fill_bytes(stream, field.values, 0u);
        }

        template <class Value>
        void clear(VectorField<Value>& field) const {
            ::cuda::fill_bytes(stream, field.x, 0u);
            ::cuda::fill_bytes(stream, field.y, 0u);
            ::cuda::fill_bytes(stream, field.z, 0u);
        }

        template <class Value>
        void copy(const ScalarField<Value>& source, ScalarField<Value>& destination) const {
            ::cuda::copy_bytes(stream, source.values, destination.values);
        }

        template <class Value>
        void copy(const VectorField<Value>& source, VectorField<Value>& destination) const {
            ::cuda::copy_bytes(stream, source.x, destination.x);
            ::cuda::copy_bytes(stream, source.y, destination.y);
            ::cuda::copy_bytes(stream, source.z, destination.z);
        }

        void accumulate(const VectorField<double>& source, VectorField<double>& destination) const;
        void upload(std::span<const Vector3> source, VectorField<float>& destination) const;
    };
} // namespace physica::deformables::cloth
