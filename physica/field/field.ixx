module;

#include <physica/cuda.h>

export module physica.field;

import std;
export import physica.math;

export namespace physica {
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

    template <class Value>
    struct Matrix3Field final {
        ::cuda::device_buffer<Value> c00;
        ::cuda::device_buffer<Value> c01;
        ::cuda::device_buffer<Value> c02;
        ::cuda::device_buffer<Value> c10;
        ::cuda::device_buffer<Value> c11;
        ::cuda::device_buffer<Value> c12;
        ::cuda::device_buffer<Value> c20;
        ::cuda::device_buffer<Value> c21;
        ::cuda::device_buffer<Value> c22;
    };

    struct FieldContext final {
        ::cuda::stream_ref stream;

        explicit FieldContext(::cuda::stream_ref stream);

        FieldContext(const FieldContext&)            = delete;
        FieldContext& operator=(const FieldContext&) = delete;
        FieldContext(FieldContext&&)                 = delete;
        FieldContext& operator=(FieldContext&&)      = delete;

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
        [[nodiscard]] Matrix3Field<Value> allocate_matrix3_field(const std::size_t count) const {
            return {
                .c00 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .c01 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .c02 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .c10 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .c11 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .c12 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .c20 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .c21 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
                .c22 = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
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
        void clear(Matrix3Field<Value>& field) const {
            ::cuda::fill_bytes(stream, field.c00, 0u);
            ::cuda::fill_bytes(stream, field.c01, 0u);
            ::cuda::fill_bytes(stream, field.c02, 0u);
            ::cuda::fill_bytes(stream, field.c10, 0u);
            ::cuda::fill_bytes(stream, field.c11, 0u);
            ::cuda::fill_bytes(stream, field.c12, 0u);
            ::cuda::fill_bytes(stream, field.c20, 0u);
            ::cuda::fill_bytes(stream, field.c21, 0u);
            ::cuda::fill_bytes(stream, field.c22, 0u);
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

        template <class Value>
        void copy(const Matrix3Field<Value>& source, Matrix3Field<Value>& destination) const {
            ::cuda::copy_bytes(stream, source.c00, destination.c00);
            ::cuda::copy_bytes(stream, source.c01, destination.c01);
            ::cuda::copy_bytes(stream, source.c02, destination.c02);
            ::cuda::copy_bytes(stream, source.c10, destination.c10);
            ::cuda::copy_bytes(stream, source.c11, destination.c11);
            ::cuda::copy_bytes(stream, source.c12, destination.c12);
            ::cuda::copy_bytes(stream, source.c20, destination.c20);
            ::cuda::copy_bytes(stream, source.c21, destination.c21);
            ::cuda::copy_bytes(stream, source.c22, destination.c22);
        }

        void accumulate(const ScalarField<double>& source, ScalarField<double>& destination) const;
        void accumulate(const VectorField<double>& source, VectorField<double>& destination) const;
        void upload(std::span<const Vector3<float>> source, VectorField<float>& destination) const;
    };
}
