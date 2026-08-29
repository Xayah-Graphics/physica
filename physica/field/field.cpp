module;

#include "field-kernels.h"
#include <physica/cuda.h>

module physica.field;

import std;

namespace physica {
    FieldContext::FieldContext(const ::cuda::stream_ref source_stream) : stream(source_stream) {}

    void FieldContext::accumulate(const ScalarField<double>& source, ScalarField<double>& destination) const {
        field::kernels::accumulate(stream, source.values.data(), destination.values.data(), source.values.size());
    }

    void FieldContext::accumulate(const VectorField<double>& source, VectorField<double>& destination) const {
        field::kernels::accumulate(stream, field::view(source), field::view(destination), source.x.size());
    }

    void FieldContext::upload(const std::span<const Vector3<float>> source, VectorField<float>& destination) const {
        std::vector<float> x(source.size());
        std::vector<float> y(source.size());
        std::vector<float> z(source.size());
        for (std::size_t index = 0uz; index < source.size(); ++index) {
            x[index] = source[index].x;
            y[index] = source[index].y;
            z[index] = source[index].z;
        }
        ::cuda::copy_bytes(stream, x, destination.x);
        ::cuda::copy_bytes(stream, y, destination.y);
        ::cuda::copy_bytes(stream, z, destination.z);
        stream.sync();
    }
}
