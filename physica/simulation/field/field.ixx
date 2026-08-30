module;

#include <physica/cuda.h>

export module physica.simulation.field;

import std;
export import physica.math;

export namespace physica::simulation {
    template <class Value>
    struct ScalarField final {
        ::cuda::device_buffer<Value> values;

        ScalarField(const ::cuda::stream_ref stream, const std::size_t count) : values{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init} {}
    };

    template <class Value>
    struct VectorField final {
        ::cuda::device_buffer<Value> x;
        ::cuda::device_buffer<Value> y;
        ::cuda::device_buffer<Value> z;

        VectorField(const ::cuda::stream_ref stream, const std::size_t count) : VectorField(stream, {count, count, count}) {}

        VectorField(const ::cuda::stream_ref stream, const std::array<std::size_t, 3u> counts) : x{stream, ::cuda::device_default_memory_pool(stream.device()), counts[0], ::cuda::no_init}, y{stream, ::cuda::device_default_memory_pool(stream.device()), counts[1], ::cuda::no_init}, z{stream, ::cuda::device_default_memory_pool(stream.device()), counts[2], ::cuda::no_init} {}
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

        Matrix3Field(const ::cuda::stream_ref stream, const std::size_t count) : c00{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, c01{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, c02{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, c10{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, c11{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, c12{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, c20{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, c21{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}, c22{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init} {}
    };

    template <class Value>
    void clear(const ::cuda::stream_ref stream, ScalarField<Value>& field) {
        ::cuda::fill_bytes(stream, field.values, 0u);
    }

    template <class Value>
    void clear(const ::cuda::stream_ref stream, VectorField<Value>& field) {
        ::cuda::fill_bytes(stream, field.x, 0u);
        ::cuda::fill_bytes(stream, field.y, 0u);
        ::cuda::fill_bytes(stream, field.z, 0u);
    }

    template <class Value>
    void clear(const ::cuda::stream_ref stream, Matrix3Field<Value>& field) {
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
    void copy(const ::cuda::stream_ref stream, const ScalarField<Value>& source, ScalarField<Value>& destination) {
        ::cuda::copy_bytes(stream, source.values, destination.values);
    }

    template <class Value>
    void copy(const ::cuda::stream_ref stream, const VectorField<Value>& source, VectorField<Value>& destination) {
        ::cuda::copy_bytes(stream, source.x, destination.x);
        ::cuda::copy_bytes(stream, source.y, destination.y);
        ::cuda::copy_bytes(stream, source.z, destination.z);
    }

    template <class Value>
    void copy(const ::cuda::stream_ref stream, const Matrix3Field<Value>& source, Matrix3Field<Value>& destination) {
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

    void accumulate(::cuda::stream_ref stream, const ScalarField<double>& source, ScalarField<double>& destination);
    void accumulate(::cuda::stream_ref stream, const VectorField<double>& source, VectorField<double>& destination);
    void upload(::cuda::stream_ref stream, std::span<const Vector3<float>> source, VectorField<float>& destination);
} // namespace physica::simulation
