module;

#include "interop.h"
#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>

module physica.fluids.liquid.particle.domain;

import std;

namespace physica::fluids::liquid::particle {
    Domain::Domain(DomainConfiguration next_configuration, const ::cuda::stream_ref source_stream)
        : configuration(std::move(next_configuration)), stream(source_stream), boundary{
            .positions = allocate_vector_field(configuration.boundary_particles.size()),
            .velocities = allocate_vector_field(configuration.boundary_particles.size()),
            .volumes = allocate_scalar_field(configuration.boundary_particles.size()),
        } {
        std::vector<Vector3> positions(configuration.boundary_particles.size());
        std::vector<Vector3> velocities(configuration.boundary_particles.size());
        std::vector<float> volumes(configuration.boundary_particles.size());
        for (std::size_t index = 0uz; index < configuration.boundary_particles.size(); ++index) {
            positions[index] = configuration.boundary_particles[index].position;
            velocities[index] = configuration.boundary_particles[index].velocity;
            volumes[index] = configuration.boundary_particles[index].volume;
        }
        upload(positions, boundary.positions);
        upload(velocities, boundary.velocities);
        upload(volumes, boundary.volumes);
        stream.sync();
    }

    ScalarField Domain::allocate_scalar_field(const std::size_t count) const {
        return {.values = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}};
    }

    ScalarAdjointField Domain::allocate_scalar_adjoint_field(const std::size_t count) const {
        return {.values = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init}};
    }

    VectorField Domain::allocate_vector_field(const std::size_t count) const {
        return {
            .x = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
            .y = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
            .z = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
        };
    }

    VectorAdjointField Domain::allocate_vector_adjoint_field(const std::size_t count) const {
        return {
            .x = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
            .y = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
            .z = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init},
        };
    }

    ParticleState Domain::allocate_particle_state() const {
        ParticleState state{.positions = allocate_vector_field(configuration.particle_count), .velocities = allocate_vector_field(configuration.particle_count)};
        clear(state.positions);
        clear(state.velocities);
        return state;
    }

    Control Domain::allocate_control() const {
        Control control{.external_accelerations = allocate_vector_field(configuration.particle_count)};
        clear(control.external_accelerations);
        return control;
    }

    ParticleStateTangent Domain::allocate_particle_state_tangent() const {
        ParticleStateTangent tangent{.positions = allocate_vector_field(configuration.particle_count), .velocities = allocate_vector_field(configuration.particle_count)};
        clear(tangent.positions);
        clear(tangent.velocities);
        return tangent;
    }

    ControlTangent Domain::allocate_control_tangent() const {
        ControlTangent tangent{.external_accelerations = allocate_vector_field(configuration.particle_count)};
        clear(tangent.external_accelerations);
        return tangent;
    }

    ParticleStateAdjoint Domain::allocate_particle_state_adjoint() const {
        ParticleStateAdjoint adjoint{.positions = allocate_vector_adjoint_field(configuration.particle_count), .velocities = allocate_vector_adjoint_field(configuration.particle_count)};
        clear(adjoint.positions);
        clear(adjoint.velocities);
        return adjoint;
    }

    ControlAdjoint Domain::allocate_control_adjoint() const {
        ControlAdjoint adjoint{.external_accelerations = allocate_vector_adjoint_field(configuration.particle_count)};
        clear(adjoint.external_accelerations);
        return adjoint;
    }

    ParticleParameters Domain::allocate_particle_parameters() const {
        return {
            .masses = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .rest_densities = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .viscosities = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .surface_tensions = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
        };
    }

    ParticleParameterTangent Domain::allocate_particle_parameter_tangent() const {
        ParticleParameterTangent tangent{
            .masses = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .rest_densities = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .viscosities = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .surface_tensions = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
        };
        clear(tangent);
        return tangent;
    }

    ParticleParameterAdjoint Domain::allocate_particle_parameter_adjoint() const {
        ParticleParameterAdjoint adjoint{
            .masses = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .rest_densities = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .viscosities = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
            .surface_tensions = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.particle_count, ::cuda::no_init},
        };
        clear(adjoint);
        return adjoint;
    }

    void Domain::clear(ScalarField& field) const { ::cuda::fill_bytes(stream, field.values, 0u); }
    void Domain::clear(ScalarAdjointField& field) const { ::cuda::fill_bytes(stream, field.values, 0u); }

    void Domain::clear(VectorField& field) const {
        ::cuda::fill_bytes(stream, field.x, 0u);
        ::cuda::fill_bytes(stream, field.y, 0u);
        ::cuda::fill_bytes(stream, field.z, 0u);
    }

    void Domain::clear(VectorAdjointField& field) const {
        ::cuda::fill_bytes(stream, field.x, 0u);
        ::cuda::fill_bytes(stream, field.y, 0u);
        ::cuda::fill_bytes(stream, field.z, 0u);
    }

    void Domain::clear(ParticleParameterTangent& tangent) const {
        ::cuda::fill_bytes(stream, tangent.masses, 0u);
        ::cuda::fill_bytes(stream, tangent.rest_densities, 0u);
        ::cuda::fill_bytes(stream, tangent.viscosities, 0u);
        ::cuda::fill_bytes(stream, tangent.surface_tensions, 0u);
    }

    void Domain::clear(ParticleParameterAdjoint& adjoint) const {
        ::cuda::fill_bytes(stream, adjoint.masses, 0u);
        ::cuda::fill_bytes(stream, adjoint.rest_densities, 0u);
        ::cuda::fill_bytes(stream, adjoint.viscosities, 0u);
        ::cuda::fill_bytes(stream, adjoint.surface_tensions, 0u);
    }

    void Domain::copy(const ScalarField& source, ScalarField& destination) const { ::cuda::copy_bytes(stream, source.values, destination.values); }
    void Domain::copy(const ScalarAdjointField& source, ScalarAdjointField& destination) const { ::cuda::copy_bytes(stream, source.values, destination.values); }

    void Domain::copy(const VectorField& source, VectorField& destination) const {
        ::cuda::copy_bytes(stream, source.x, destination.x);
        ::cuda::copy_bytes(stream, source.y, destination.y);
        ::cuda::copy_bytes(stream, source.z, destination.z);
    }

    void Domain::copy(const VectorAdjointField& source, VectorAdjointField& destination) const {
        ::cuda::copy_bytes(stream, source.x, destination.x);
        ::cuda::copy_bytes(stream, source.y, destination.y);
        ::cuda::copy_bytes(stream, source.z, destination.z);
    }

    void Domain::copy(const ParticleState& source, ParticleState& destination) const {
        copy(source.positions, destination.positions);
        copy(source.velocities, destination.velocities);
        destination.step_index = source.step_index;
    }

    void Domain::copy(const ParticleStateTangent& source, ParticleStateTangent& destination) const {
        copy(source.positions, destination.positions);
        copy(source.velocities, destination.velocities);
    }

    void Domain::copy(const ParticleStateAdjoint& source, ParticleStateAdjoint& destination) const {
        copy(source.positions, destination.positions);
        copy(source.velocities, destination.velocities);
    }

    void Domain::accumulate(const ScalarAdjointField& source, ScalarAdjointField& destination) const {
        cuda_detail::accumulate(stream, source.values.data(), destination.values.data(), source.values.size());
    }

    void Domain::accumulate(const VectorAdjointField& source, VectorAdjointField& destination) const {
        cuda_detail::accumulate(stream, cuda_detail::adjoint_vector(source), cuda_detail::adjoint_vector(destination), source.x.size());
    }

    void Domain::accumulate(const ParticleStateAdjoint& source, ParticleStateAdjoint& destination) const {
        accumulate(source.positions, destination.positions);
        accumulate(source.velocities, destination.velocities);
    }

    void Domain::upload(const std::span<const float> source, ScalarField& destination) const {
        ::cuda::copy_bytes(stream, source, destination.values);
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
        ::cuda::copy_bytes(stream, x, destination.x);
        ::cuda::copy_bytes(stream, y, destination.y);
        ::cuda::copy_bytes(stream, z, destination.z);
    }
} // namespace physica::fluids::liquid::particle
