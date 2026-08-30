#include "mass-spring-kernels.h"
#include <cuda/launch>
#include <cuda/std/cmath>

namespace physica::deformables::cloth::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __device__ Vector3<float> spring_force(const Vector3<float> first_position, const Vector3<float> second_position, const Vector3<float> first_velocity, const Vector3<float> second_velocity, const float stiffness, const float damping, const float rest_length) {
            const Vector3<float> displacement      = second_position - first_position;
            const float length                     = ::cuda::std::sqrt(dot(displacement, displacement));
            const Vector3<float> direction         = displacement / length;
            const Vector3<float> relative_velocity = second_velocity - first_velocity;
            const float magnitude                  = stiffness * (length - rest_length) + damping * dot(relative_velocity, direction);
            return magnitude * direction;
        }

        __device__ Vector3<float> spring_force_tangent(const Vector3<float> first_position, const Vector3<float> second_position, const Vector3<float> first_velocity, const Vector3<float> second_velocity, const Vector3<float> first_position_tangent, const Vector3<float> second_position_tangent, const Vector3<float> first_velocity_tangent, const Vector3<float> second_velocity_tangent, const float stiffness, const float damping, const float rest_length, const float stiffness_tangent, const float damping_tangent, const float rest_length_tangent) {
            const Vector3<float> displacement              = second_position - first_position;
            const Vector3<float> displacement_tangent      = second_position_tangent - first_position_tangent;
            const float length                             = ::cuda::std::sqrt(dot(displacement, displacement));
            const Vector3<float> direction                 = displacement / length;
            const float length_tangent                     = dot(direction, displacement_tangent);
            const Vector3<float> direction_tangent         = (displacement_tangent - length_tangent * direction) / length;
            const Vector3<float> relative_velocity         = second_velocity - first_velocity;
            const Vector3<float> relative_velocity_tangent = second_velocity_tangent - first_velocity_tangent;
            const float axial_velocity                     = dot(relative_velocity, direction);
            const float axial_velocity_tangent             = dot(relative_velocity_tangent, direction) + dot(relative_velocity, direction_tangent);
            const float magnitude                          = stiffness * (length - rest_length) + damping * axial_velocity;
            const float magnitude_tangent                  = stiffness_tangent * (length - rest_length) + stiffness * (length_tangent - rest_length_tangent) + damping_tangent * axial_velocity + damping * axial_velocity_tangent;
            return magnitude_tangent * direction + magnitude * direction_tangent;
        }

        __device__ void spring_vjp(const Vector3<float> first_position, const Vector3<float> second_position, const Vector3<float> first_velocity, const Vector3<float> second_velocity, const float stiffness, const float damping, const float rest_length, const Vector3<double> force_adjoint, Vector3<double>& displacement_adjoint, Vector3<double>& relative_velocity_adjoint, double& stiffness_adjoint, double& damping_adjoint, double& rest_length_adjoint) {
            const Vector3<double> displacement{static_cast<double>(second_position.x) - first_position.x, static_cast<double>(second_position.y) - first_position.y, static_cast<double>(second_position.z) - first_position.z};
            const double length             = ::cuda::std::sqrt(dot(displacement, displacement));
            const Vector3<double> direction = displacement / length;
            const Vector3<double> relative_velocity{static_cast<double>(second_velocity.x) - first_velocity.x, static_cast<double>(second_velocity.y) - first_velocity.y, static_cast<double>(second_velocity.z) - first_velocity.z};
            const double axial_velocity       = dot(relative_velocity, direction);
            const double magnitude            = static_cast<double>(stiffness) * (length - rest_length) + static_cast<double>(damping) * axial_velocity;
            const double magnitude_adjoint    = dot(force_adjoint, direction);
            Vector3<double> direction_adjoint = magnitude * force_adjoint;
            const double axial_adjoint        = static_cast<double>(damping) * magnitude_adjoint;
            const double length_adjoint       = static_cast<double>(stiffness) * magnitude_adjoint;
            stiffness_adjoint                 = (length - rest_length) * magnitude_adjoint;
            damping_adjoint                   = axial_velocity * magnitude_adjoint;
            rest_length_adjoint               = -static_cast<double>(stiffness) * magnitude_adjoint;
            relative_velocity_adjoint         = axial_adjoint * direction;
            direction_adjoint                 = direction_adjoint + axial_adjoint * relative_velocity;
            displacement_adjoint              = length_adjoint * direction + (direction_adjoint - dot(direction, direction_adjoint) * direction) / length;
        }

        __device__ Vector3<float> gathered_force(const std::uint32_t particle, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const SpringTopologyView topology, const SpringParametersView parameters) {
            Vector3<float> result{};
            for (std::uint32_t entry = topology.offsets[particle]; entry < topology.offsets[particle + 1u]; ++entry) {
                const std::uint32_t spring = topology.indices[entry];
                const Vector3<float> force = spring_force(load(positions, topology.first[spring]), load(positions, topology.second[spring]), load(velocities, topology.first[spring]), load(velocities, topology.second[spring]), parameters.stiffnesses[spring], parameters.dampings[spring], parameters.rest_lengths[spring]);
                result                     = result + (topology.first[spring] == particle ? force : -1.0F * force);
            }
            return result;
        }

        __device__ Vector3<float> gathered_force_tangent(const std::uint32_t particle, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const SpringTopologyView topology, const SpringParametersView parameters, const SpringParametersView parameter_tangent) {
            Vector3<float> result{};
            for (std::uint32_t entry = topology.offsets[particle]; entry < topology.offsets[particle + 1u]; ++entry) {
                const std::uint32_t spring   = topology.indices[entry];
                const std::uint32_t first    = topology.first[spring];
                const std::uint32_t second   = topology.second[spring];
                const Vector3<float> tangent = spring_force_tangent(load(positions, first), load(positions, second), load(velocities, first), load(velocities, second), load(position_tangent, first), load(position_tangent, second), load(velocity_tangent, first), load(velocity_tangent, second), parameters.stiffnesses[spring], parameters.dampings[spring], parameters.rest_lengths[spring], parameter_tangent.stiffnesses[spring], parameter_tangent.dampings[spring], parameter_tangent.rest_lengths[spring]);
                result                       = result + (first == particle ? tangent : -1.0F * tangent);
            }
            return result;
        }

        __device__ void gathered_state_vjp(const std::uint32_t particle, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const double> force_adjoint, const SpringTopologyView topology, const SpringParametersView parameters, Vector3<double>& position_adjoint, Vector3<double>& velocity_adjoint) {
            for (std::uint32_t entry = topology.offsets[particle]; entry < topology.offsets[particle + 1u]; ++entry) {
                const std::uint32_t spring = topology.indices[entry];
                const std::uint32_t first  = topology.first[spring];
                const std::uint32_t second = topology.second[spring];
                Vector3<double> displacement_adjoint;
                Vector3<double> relative_velocity_adjoint;
                double stiffness_adjoint;
                double damping_adjoint;
                double rest_length_adjoint;
                spring_vjp(load(positions, first), load(positions, second), load(velocities, first), load(velocities, second), parameters.stiffnesses[spring], parameters.dampings[spring], parameters.rest_lengths[spring], load(force_adjoint, first) - load(force_adjoint, second), displacement_adjoint, relative_velocity_adjoint, stiffness_adjoint, damping_adjoint, rest_length_adjoint);
                const double sign = first == particle ? -1.0 : 1.0;
                position_adjoint  = position_adjoint + sign * displacement_adjoint;
                velocity_adjoint  = velocity_adjoint + sign * relative_velocity_adjoint;
            }
        }

        __global__ void force_forward_kernel(const std::uint32_t particle_count, const Vector3<float> gravity, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> controls, const float* masses, const SpringTopologyView stretch_topology, const SpringParametersView stretch_parameters, const SpringTopologyView bending_topology, const SpringParametersView bending_parameters, const simulation::VectorView<float> forces) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(forces, particle, load(controls, particle) + masses[particle] * gravity + gathered_force(particle, positions, velocities, stretch_topology, stretch_parameters) + gathered_force(particle, positions, velocities, bending_topology, bending_parameters));
        }

        __global__ void force_jvp_kernel(const std::uint32_t particle_count, const Vector3<float> gravity, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> control_tangent, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const float* mass_tangent, const SpringTopologyView stretch_topology, const SpringParametersView stretch_parameters, const SpringParametersView stretch_tangent, const SpringTopologyView bending_topology, const SpringParametersView bending_parameters, const SpringParametersView bending_tangent, const simulation::VectorView<float> output) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(output, particle, load(control_tangent, particle) + mass_tangent[particle] * gravity + gathered_force_tangent(particle, positions, velocities, position_tangent, velocity_tangent, stretch_topology, stretch_parameters, stretch_tangent) + gathered_force_tangent(particle, positions, velocities, position_tangent, velocity_tangent, bending_topology, bending_parameters, bending_tangent));
        }

        __global__ void force_state_vjp_kernel(const std::uint32_t particle_count, const Vector3<float> gravity, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const double> force_adjoint, const SpringTopologyView stretch_topology, const SpringParametersView stretch_parameters, const SpringTopologyView bending_topology, const SpringParametersView bending_parameters, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint, const simulation::VectorView<double> control_adjoint, double* mass_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Vector3<double> local_position_adjoint{};
            Vector3<double> local_velocity_adjoint{};
            gathered_state_vjp(particle, positions, velocities, force_adjoint, stretch_topology, stretch_parameters, local_position_adjoint, local_velocity_adjoint);
            gathered_state_vjp(particle, positions, velocities, force_adjoint, bending_topology, bending_parameters, local_position_adjoint, local_velocity_adjoint);
            accumulate(position_adjoint, particle, local_position_adjoint);
            accumulate(velocity_adjoint, particle, local_velocity_adjoint);
            const Vector3<double> local_force_adjoint = load(force_adjoint, particle);
            accumulate(control_adjoint, particle, local_force_adjoint);
            mass_adjoint[particle] += gravity.x * local_force_adjoint.x + gravity.y * local_force_adjoint.y + gravity.z * local_force_adjoint.z;
        }

        __global__ void force_parameter_vjp_kernel(const std::uint32_t spring_count, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const double> force_adjoint, const SpringTopologyView topology, const SpringParametersView parameters, const SpringParameterAdjointView parameter_adjoint) {
            const std::uint32_t spring = blockIdx.x * blockDim.x + threadIdx.x;
            if (spring >= spring_count) return;
            const std::uint32_t first  = topology.first[spring];
            const std::uint32_t second = topology.second[spring];
            Vector3<double> displacement_adjoint;
            Vector3<double> relative_velocity_adjoint;
            double stiffness_adjoint;
            double damping_adjoint;
            double rest_length_adjoint;
            spring_vjp(load(positions, first), load(positions, second), load(velocities, first), load(velocities, second), parameters.stiffnesses[spring], parameters.dampings[spring], parameters.rest_lengths[spring], load(force_adjoint, first) - load(force_adjoint, second), displacement_adjoint, relative_velocity_adjoint, stiffness_adjoint, damping_adjoint, rest_length_adjoint);
            parameter_adjoint.stiffnesses[spring] += stiffness_adjoint;
            parameter_adjoint.dampings[spring] += damping_adjoint;
            parameter_adjoint.rest_lengths[spring] += rest_length_adjoint;
        }
    } // namespace

    void force_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const Vector3<float> gravity, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> controls, const float* masses, const SpringTopologyView stretch_topology, const SpringParametersView stretch_parameters, const SpringTopologyView bending_topology, const SpringParametersView bending_parameters, const simulation::VectorView<float> forces) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), force_forward_kernel, particle_count, gravity, positions, velocities, controls, masses, stretch_topology, stretch_parameters, bending_topology, bending_parameters, forces);
    }

    void force_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const Vector3<float> gravity, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> control_tangent, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const float* mass_tangent, const SpringTopologyView stretch_topology, const SpringParametersView stretch_parameters, const SpringParametersView stretch_tangent, const SpringTopologyView bending_topology, const SpringParametersView bending_parameters, const SpringParametersView bending_tangent, const simulation::VectorView<float> force_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), force_jvp_kernel, particle_count, gravity, positions, velocities, control_tangent, position_tangent, velocity_tangent, mass_tangent, stretch_topology, stretch_parameters, stretch_tangent, bending_topology, bending_parameters, bending_tangent, force_tangent);
    }

    void force_state_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const Vector3<float> gravity, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const double> force_adjoint, const SpringTopologyView stretch_topology, const SpringParametersView stretch_parameters, const SpringTopologyView bending_topology, const SpringParametersView bending_parameters, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint, const simulation::VectorView<double> control_adjoint, double* mass_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), force_state_vjp_kernel, particle_count, gravity, positions, velocities, force_adjoint, stretch_topology, stretch_parameters, bending_topology, bending_parameters, position_adjoint, velocity_adjoint, control_adjoint, mass_adjoint);
    }

    void force_parameter_vjp(const ::cuda::stream_ref stream, const std::uint32_t spring_count, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const double> force_adjoint, const SpringTopologyView topology, const SpringParametersView parameters, const SpringParameterAdjointView parameter_adjoint) {
        if (spring_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(spring_count), force_parameter_vjp_kernel, spring_count, positions, velocities, force_adjoint, topology, parameters, parameter_adjoint);
    }
} // namespace physica::deformables::cloth::kernels
