module;

#include <physica/cuda.h>

export module physica.fluids.liquid.domain;

import std;

export namespace physica::fluids::liquid {
    struct Vector3 final {
        float x{};
        float y{};
        float z{};
    };

    struct BoxBoundary final {
        Vector3 minimum;
        Vector3 maximum;
        Vector3 velocity;
        bool no_slip{true};
    };

    struct BoundaryParticle final {
        Vector3 position;
        Vector3 velocity;
        float volume;
    };

    struct DomainConfiguration final {
        std::uint32_t particle_count;
        float time_step;
        float support_radius;
        float particle_radius;
        BoxBoundary boundary;
        std::vector<BoundaryParticle> boundary_particles;
    };

    template <class Value>
    struct ScalarField final {
        ::cuda::device_buffer<Value> values;
    };

    template <class Value>
    struct VectorField final {
        ::cuda::device_buffer<Value> x;
        ::cuda::device_buffer<Value> y;
        ::cuda::device_buffer<Value> z;
    };

    struct DeviceBoundary final {
        VectorField<float> positions;
        VectorField<float> velocities;
        ScalarField<float> volumes;
    };

    struct Domain final {
        const DomainConfiguration configuration;
        ::cuda::stream_ref stream;
        DeviceBoundary boundary;

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

        void accumulate(const ScalarField<double>& source, ScalarField<double>& destination) const;
        void accumulate(const VectorField<double>& source, VectorField<double>& destination) const;
    };
} // namespace physica::fluids::liquid
