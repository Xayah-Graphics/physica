module;

#include "field-kernels.h"
#include <physica/cuda.h>

module physica.simulation.field;

import std;

namespace physica::simulation {
    void accumulate(const ::cuda::stream_ref stream, const ScalarField<double>& source, ScalarField<double>& destination) {
        kernels::accumulate(stream, source.values.data(), destination.values.data(), source.values.size());
    }

    void accumulate(const ::cuda::stream_ref stream, const VectorField<double>& source, VectorField<double>& destination) {
        kernels::accumulate(stream, view(source), view(destination), source.x.size());
    }

    void upload(const ::cuda::stream_ref stream, const std::span<const Vector3<float>> source, VectorField<float>& destination) {
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
} // namespace physica::simulation
