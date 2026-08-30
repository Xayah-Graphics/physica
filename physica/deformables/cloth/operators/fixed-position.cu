#include <simulation/field/device.cuh>
#include "fixed-position-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void forward_kernel(const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const simulation::VectorView<const float> anchor_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<float> constrained_positions, const simulation::VectorView<float> constrained_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            if (anchor_mask[particle] != 0u) {
                store(constrained_positions, particle, load(anchor_positions, particle));
                store(constrained_velocities, particle, Vector3<float>{});
                return;
            }
            store(constrained_positions, particle, load(positions, particle));
            store(constrained_velocities, particle, load(velocities, particle));
        }

        __global__ void jvp_kernel(const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const simulation::VectorView<float> constrained_position_tangent, const simulation::VectorView<float> constrained_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            if (anchor_mask[particle] != 0u) {
                store(constrained_position_tangent, particle, Vector3<float>{});
                store(constrained_velocity_tangent, particle, Vector3<float>{});
                return;
            }
            store(constrained_position_tangent, particle, load(position_tangent, particle));
            store(constrained_velocity_tangent, particle, load(velocity_tangent, particle));
        }

        __global__ void vjp_kernel(const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const simulation::VectorView<const double> constrained_position_adjoint, const simulation::VectorView<const double> constrained_velocity_adjoint, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            if (anchor_mask[particle] != 0u) {
                store(position_adjoint, particle, Vector3<double>{});
                store(velocity_adjoint, particle, Vector3<double>{});
                return;
            }
            store(position_adjoint, particle, load(constrained_position_adjoint, particle));
            store(velocity_adjoint, particle, load(constrained_velocity_adjoint, particle));
        }
    } // namespace

    void fixed_position_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const simulation::VectorView<const float> anchor_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<float> constrained_positions, const simulation::VectorView<float> constrained_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), forward_kernel, particle_count, anchor_mask, anchor_positions, positions, velocities, constrained_positions, constrained_velocities);
    }

    void fixed_position_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const simulation::VectorView<float> constrained_position_tangent, const simulation::VectorView<float> constrained_velocity_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), jvp_kernel, particle_count, anchor_mask, position_tangent, velocity_tangent, constrained_position_tangent, constrained_velocity_tangent);
    }

    void fixed_position_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const simulation::VectorView<const double> constrained_position_adjoint, const simulation::VectorView<const double> constrained_velocity_adjoint, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), vjp_kernel, particle_count, anchor_mask, constrained_position_adjoint, constrained_velocity_adjoint, position_adjoint, velocity_adjoint);
    }
} // namespace physica::deformables::cloth::kernels
