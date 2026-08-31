#include "position-dynamics-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::position_dynamics::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void predict_kernel(const std::uint32_t particle_count, const float time_step, const Vector3<float> gravity, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> external_forces, const float* masses, const simulation::VectorView<float> predicted_positions) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            if (fixed_vertex_mask[particle] != 0u) {
                store(predicted_positions, particle, load(fixed_positions, particle));
                return;
            }
            const Vector3<float> predicted_velocity = load(velocities, particle) + time_step * (gravity + load(external_forces, particle) / masses[particle]);
            store(predicted_positions, particle, load(positions, particle) + time_step * predicted_velocity);
        }

        __global__ void reconstruct_velocities_kernel(const std::uint32_t particle_count, const float inverse_time_step, const simulation::VectorView<const float> previous_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<float> velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(velocities, particle, inverse_time_step * (load(positions, particle) - load(previous_positions, particle)));
        }
    } // namespace

    void predict(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const Vector3<float> gravity, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> external_forces, const float* masses, const simulation::VectorView<float> predicted_positions) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), predict_kernel, particle_count, time_step, gravity, fixed_vertex_mask, fixed_positions, positions, velocities, external_forces, masses, predicted_positions);
    }

    void reconstruct_velocities(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> previous_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<float> velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), reconstruct_velocities_kernel, particle_count, 1.0F / time_step, previous_positions, positions, velocities);
    }
} // namespace physica::deformables::cloth::solvers::position_dynamics::kernels
