#include "kernels.h"
#include <cuda/launch>

namespace physica::examples::pbf_dam_break::spectra_cuda {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void write_vectors_kernel(const std::uint32_t particle_count, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            positions[particle]  = {.x = position_x[particle], .y = position_y[particle], .z = position_z[particle]};
            velocities[particle] = {.x = velocity_x[particle], .y = velocity_y[particle], .z = velocity_z[particle]};
        }
    } // namespace

    void write_vectors(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), write_vectors_kernel, particle_count, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, positions, velocities);
    }
} // namespace physica::examples::pbf_dam_break::spectra_cuda
