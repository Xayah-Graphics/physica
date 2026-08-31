#include "forward-euler-kernels.h"
#include <cuda/launch>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void forward_kernel(const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> forces, const float* masses, const simulation::VectorView<float> integrated_positions, const simulation::VectorView<float> integrated_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(integrated_positions, particle, load(positions, particle) + time_step * load(velocities, particle));
            store(integrated_velocities, particle, load(velocities, particle) + (time_step / masses[particle]) * load(forces, particle));
        }
    } // namespace

    void forward_euler_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> forces, const float* masses, const simulation::VectorView<float> integrated_positions, const simulation::VectorView<float> integrated_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), forward_kernel, particle_count, time_step, positions, velocities, forces, masses, integrated_positions, integrated_velocities);
    }
} // namespace physica::deformables::cloth::kernels
