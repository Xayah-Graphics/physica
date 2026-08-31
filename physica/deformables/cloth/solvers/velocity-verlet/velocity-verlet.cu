#include "velocity-verlet-kernels.h"
#include <cuda/launch>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::velocity_verlet::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        template <class Value>
        __global__ void predict_kernel(const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const Value> positions, const simulation::VectorView<const Value> velocities, const Value* masses, const simulation::VectorView<const Value> forces, const simulation::VectorView<Value> predicted_positions, const simulation::VectorView<Value> predicted_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Value step              = static_cast<Value>(time_step);
            const Vector3<Value> velocity = load(velocities, particle) + (static_cast<Value>(0.5F) * step / masses[particle]) * load(forces, particle);
            store(predicted_positions, particle, load(positions, particle) + step * velocity);
            store(predicted_velocities, particle, velocity);
        }

        template <class Value>
        __global__ void second_half_kick_kernel(const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const Value> predicted_positions, const simulation::VectorView<const Value> predicted_velocities, const Value* masses, const simulation::VectorView<const Value> forces, const simulation::VectorView<Value> final_positions, const simulation::VectorView<Value> final_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Value step = static_cast<Value>(time_step);
            store(final_positions, particle, load(predicted_positions, particle));
            store(final_velocities, particle, load(predicted_velocities, particle) + (static_cast<Value>(0.5F) * step / masses[particle]) * load(forces, particle));
        }
    } // namespace

    void velocity_verlet_predict(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const float* masses, const simulation::VectorView<const float> forces, const simulation::VectorView<float> predicted_positions, const simulation::VectorView<float> predicted_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), predict_kernel<float>, particle_count, time_step, positions, velocities, masses, forces, predicted_positions, predicted_velocities);
    }

    void velocity_verlet_predict(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const double> positions, const simulation::VectorView<const double> velocities, const double* masses, const simulation::VectorView<const double> forces, const simulation::VectorView<double> predicted_positions, const simulation::VectorView<double> predicted_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), predict_kernel<double>, particle_count, time_step, positions, velocities, masses, forces, predicted_positions, predicted_velocities);
    }

    void velocity_verlet_second_half_kick(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> predicted_velocities, const float* masses, const simulation::VectorView<const float> forces, const simulation::VectorView<float> final_positions, const simulation::VectorView<float> final_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), second_half_kick_kernel<float>, particle_count, time_step, predicted_positions, predicted_velocities, masses, forces, final_positions, final_velocities);
    }

    void velocity_verlet_second_half_kick(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const double> predicted_positions, const simulation::VectorView<const double> predicted_velocities, const double* masses, const simulation::VectorView<const double> forces, const simulation::VectorView<double> final_positions, const simulation::VectorView<double> final_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), second_half_kick_kernel<double>, particle_count, time_step, predicted_positions, predicted_velocities, masses, forces, final_positions, final_velocities);
    }
} // namespace physica::deformables::cloth::solvers::velocity_verlet::kernels
