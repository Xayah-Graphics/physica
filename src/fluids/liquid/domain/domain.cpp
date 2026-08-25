module;

#include "../detail/cuda/interop.h"
#include "domain-kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.domain;

import std;

namespace physica::fluids::liquid {
    Domain::Domain(DomainConfiguration next_configuration, const ::cuda::stream_ref source_stream) :
        configuration(std::move(next_configuration)),
        stream(source_stream),
        boundary{
            .positions  = allocate_vector_field<float>(configuration.boundary_particles.size()),
            .velocities = allocate_vector_field<float>(configuration.boundary_particles.size()),
            .volumes    = allocate_scalar_field<float>(configuration.boundary_particles.size()),
        } {
        std::vector<float> position_x(configuration.boundary_particles.size());
        std::vector<float> position_y(configuration.boundary_particles.size());
        std::vector<float> position_z(configuration.boundary_particles.size());
        std::vector<float> velocity_x(configuration.boundary_particles.size());
        std::vector<float> velocity_y(configuration.boundary_particles.size());
        std::vector<float> velocity_z(configuration.boundary_particles.size());
        std::vector<float> volumes(configuration.boundary_particles.size());
        for (std::size_t index = 0uz; index < configuration.boundary_particles.size(); ++index) {
            position_x[index] = configuration.boundary_particles[index].position.x;
            position_y[index] = configuration.boundary_particles[index].position.y;
            position_z[index] = configuration.boundary_particles[index].position.z;
            velocity_x[index] = configuration.boundary_particles[index].velocity.x;
            velocity_y[index] = configuration.boundary_particles[index].velocity.y;
            velocity_z[index] = configuration.boundary_particles[index].velocity.z;
            volumes[index]    = configuration.boundary_particles[index].volume;
        }
        ::cuda::copy_bytes(stream, position_x, boundary.positions.x);
        ::cuda::copy_bytes(stream, position_y, boundary.positions.y);
        ::cuda::copy_bytes(stream, position_z, boundary.positions.z);
        ::cuda::copy_bytes(stream, velocity_x, boundary.velocities.x);
        ::cuda::copy_bytes(stream, velocity_y, boundary.velocities.y);
        ::cuda::copy_bytes(stream, velocity_z, boundary.velocities.z);
        ::cuda::copy_bytes(stream, volumes, boundary.volumes.values);
        stream.sync();
    }

    void Domain::accumulate(const ScalarField<double>& source, ScalarField<double>& destination) const {
        cuda_detail::accumulate(stream, source.values.data(), destination.values.data(), source.values.size());
    }

    void Domain::accumulate(const VectorField<double>& source, VectorField<double>& destination) const {
        cuda_detail::accumulate(stream, cuda_detail::vector(source), cuda_detail::vector(destination), source.x.size());
    }

} // namespace physica::fluids::liquid
