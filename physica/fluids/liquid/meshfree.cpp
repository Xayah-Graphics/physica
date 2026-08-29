module;

#include <physica/cuda.h>

module physica.fluids.liquid.meshfree;

import std;

namespace physica::fluids::liquid::meshfree {
    Model::Model(Configuration next_configuration, const ::cuda::stream_ref source_stream)
        : configuration(std::move(next_configuration)), fields(source_stream), boundary{
            .positions  = fields.allocate_vector_field<float>(configuration.boundary_particles.size()),
            .velocities = fields.allocate_vector_field<float>(configuration.boundary_particles.size()),
            .volumes    = fields.allocate_scalar_field<float>(configuration.boundary_particles.size()),
        } {
        std::vector<Vector3<float>> positions(configuration.boundary_particles.size());
        std::vector<Vector3<float>> velocities(configuration.boundary_particles.size());
        std::vector<float> volumes(configuration.boundary_particles.size());
        for (std::size_t index = 0uz; index < configuration.boundary_particles.size(); ++index) {
            positions[index]  = configuration.boundary_particles[index].position;
            velocities[index] = configuration.boundary_particles[index].velocity;
            volumes[index]    = configuration.boundary_particles[index].volume;
        }
        fields.upload(positions, boundary.positions);
        fields.upload(velocities, boundary.velocities);
        ::cuda::copy_bytes(fields.stream, volumes, boundary.volumes.values);
        fields.stream.sync();
    }
}
