module;

#include "../detail/cuda/interop.h"
#include "domain-kernels.h"
#include <physica/cuda.h>

module physica.deformables.cloth.domain;

import std;

namespace physica::deformables::cloth {
    Domain::Domain(DomainConfiguration next_configuration, const ::cuda::stream_ref source_stream) : configuration(std::move(next_configuration)), stream(source_stream), particle_count(configuration.rest_positions.size()) {}

    void Domain::accumulate(const VectorField<double>& source, VectorField<double>& destination) const {
        cuda_detail::accumulate(stream, cuda_detail::field<double>(source), cuda_detail::field<double>(destination), particle_count);
    }

    void Domain::upload(const std::span<const Vector3> source, VectorField<float>& destination) const {
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
} // namespace physica::deformables::cloth
