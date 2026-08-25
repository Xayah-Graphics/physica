#include "../domain/device.cuh"
#include "kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::cuda_detail {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void forward_kernel(const std::uint32_t particle_count, const float time_step, const ConstFieldView<float> positions, const ConstFieldView<float> velocities, const ConstFieldView<float> forces, const float* masses, const FieldView<float> integrated_positions, const FieldView<float> integrated_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector<float> velocity = load(velocities, particle) + (time_step / masses[particle]) * load(forces, particle);
            store(integrated_velocities, particle, velocity);
            store(integrated_positions, particle, load(positions, particle) + time_step * velocity);
        }

        __global__ void jvp_kernel(const std::uint32_t particle_count, const float time_step, const ConstFieldView<float> forces, const float* masses, const ConstFieldView<float> position_tangent, const ConstFieldView<float> velocity_tangent, const ConstFieldView<float> force_tangent, const float* mass_tangent, const FieldView<float> integrated_position_tangent, const FieldView<float> integrated_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float mass             = masses[particle];
            const Vector<float> velocity = load(velocity_tangent, particle) + time_step * (load(force_tangent, particle) / mass - (mass_tangent[particle] / (mass * mass)) * load(forces, particle));
            store(integrated_velocity_tangent, particle, velocity);
            store(integrated_position_tangent, particle, load(position_tangent, particle) + time_step * velocity);
        }

        __global__ void vjp_kernel(const std::uint32_t particle_count, const float time_step, const ConstFieldView<float> forces, const float* masses, const ConstFieldView<double> integrated_position_adjoint, const ConstFieldView<double> integrated_velocity_adjoint, const FieldView<double> position_adjoint, const FieldView<double> velocity_adjoint, const FieldView<double> force_adjoint, double* mass_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector<double> local_position_adjoint = load(integrated_position_adjoint, particle);
            const Vector<double> local_velocity_adjoint = load(integrated_velocity_adjoint, particle) + static_cast<double>(time_step) * local_position_adjoint;
            add(position_adjoint, particle, local_position_adjoint);
            add(velocity_adjoint, particle, local_velocity_adjoint);
            add(force_adjoint, particle, (static_cast<double>(time_step) / masses[particle]) * local_velocity_adjoint);
            const Vector<float> force = load(forces, particle);
            mass_adjoint[particle] -= static_cast<double>(time_step) * (force.x * local_velocity_adjoint.x + force.y * local_velocity_adjoint.y + force.z * local_velocity_adjoint.z) / (static_cast<double>(masses[particle]) * masses[particle]);
        }
    } // namespace

    void semi_implicit_euler_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const ConstFieldView<float> positions, const ConstFieldView<float> velocities, const ConstFieldView<float> forces, const float* masses, const FieldView<float> integrated_positions, const FieldView<float> integrated_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), forward_kernel, particle_count, time_step, positions, velocities, forces, masses, integrated_positions, integrated_velocities);
    }

    void semi_implicit_euler_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const ConstFieldView<float> forces, const float* masses, const ConstFieldView<float> position_tangent, const ConstFieldView<float> velocity_tangent, const ConstFieldView<float> force_tangent, const float* mass_tangent, const FieldView<float> integrated_position_tangent, const FieldView<float> integrated_velocity_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), jvp_kernel, particle_count, time_step, forces, masses, position_tangent, velocity_tangent, force_tangent, mass_tangent, integrated_position_tangent, integrated_velocity_tangent);
    }

    void semi_implicit_euler_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const ConstFieldView<float> forces, const float* masses, const ConstFieldView<double> integrated_position_adjoint, const ConstFieldView<double> integrated_velocity_adjoint, const FieldView<double> position_adjoint, const FieldView<double> velocity_adjoint, const FieldView<double> force_adjoint, double* mass_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), vjp_kernel, particle_count, time_step, forces, masses, integrated_position_adjoint, integrated_velocity_adjoint, position_adjoint, velocity_adjoint, force_adjoint, mass_adjoint);
    }
} // namespace physica::deformables::cloth::cuda_detail
