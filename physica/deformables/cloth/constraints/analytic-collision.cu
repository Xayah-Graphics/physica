#include "analytic-collision-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __device__ Vector3<float> collide_velocity(const Vector3<float> velocity, const Vector3<float> normal, const float restitution, const float friction) {
            const float normal_velocity = dot(velocity, normal);
            if (normal_velocity >= 0.0F) return velocity;
            const Vector3<float> tangential_velocity = velocity - normal_velocity * normal;
            const float tangential_speed             = length(tangential_velocity);
            const float normal_impulse_speed         = -(1.0F + restitution) * normal_velocity;
            const float friction_scale               = tangential_speed == 0.0F ? 0.0F : max(0.0F, 1.0F - friction * normal_impulse_speed / tangential_speed);
            return friction_scale * tangential_velocity - restitution * normal_velocity * normal;
        }

        __global__ void initialize_kernel(const std::uint32_t particle_count, const simulation::VectorView<const float> integrated_positions, const simulation::VectorView<const float> integrated_velocities, const simulation::VectorView<float> constrained_positions, const simulation::VectorView<float> constrained_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(constrained_positions, particle, load(integrated_positions, particle));
            store(constrained_velocities, particle, load(integrated_velocities, particle));
        }

        __global__ void plane_kernel(const std::uint32_t particle_count, const Vector3<float> normal, const float offset, const float thickness, const float restitution, const float friction, const simulation::VectorView<float> positions, const simulation::VectorView<float> velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = load(positions, particle);
            const float distance          = dot(position, normal) - offset;
            if (distance >= thickness) return;
            store(positions, particle, position + (thickness - distance) * normal);
            store(velocities, particle, collide_velocity(load(velocities, particle), normal, restitution, friction));
        }

        __global__ void sphere_kernel(const std::uint32_t particle_count, const Vector3<float> center, const float collision_radius, const float restitution, const float friction, const simulation::VectorView<float> positions, const simulation::VectorView<float> velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> displacement = load(positions, particle) - center;
            const float distance              = length(displacement);
            if (distance >= collision_radius) return;
            const Vector3<float> normal = displacement / distance;
            store(positions, particle, center + collision_radius * normal);
            store(velocities, particle, collide_velocity(load(velocities, particle), normal, restitution, friction));
        }
    } // namespace

    void analytic_collision_initialize(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const simulation::VectorView<const float> integrated_positions, const simulation::VectorView<const float> integrated_velocities, const simulation::VectorView<float> constrained_positions, const simulation::VectorView<float> constrained_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), initialize_kernel, particle_count, integrated_positions, integrated_velocities, constrained_positions, constrained_velocities);
    }

    void analytic_collision_plane(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const Vector3<float> normal, const float offset, const float thickness, const float restitution, const float friction, const simulation::VectorView<float> positions, const simulation::VectorView<float> velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), plane_kernel, particle_count, normal, offset, thickness, restitution, friction, positions, velocities);
    }

    void analytic_collision_sphere(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const Vector3<float> center, const float radius, const float thickness, const float restitution, const float friction, const simulation::VectorView<float> positions, const simulation::VectorView<float> velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), sphere_kernel, particle_count, center, radius + thickness, restitution, friction, positions, velocities);
    }
} // namespace physica::deformables::cloth::kernels
