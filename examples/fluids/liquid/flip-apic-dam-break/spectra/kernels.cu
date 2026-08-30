#include "kernels.h"
#include <cuda/launch>
#include <cuda/std/cmath>

namespace physica::examples::flip_apic_dam_break::spectra_cuda {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void write_flip_particles_kernel(const std::uint32_t particle_count, const float offset_x, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities, float* speeds) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            positions[particle]  = {.x = position_x[particle] + offset_x, .y = position_y[particle], .z = position_z[particle]};
            velocities[particle] = {.x = velocity_x[particle], .y = velocity_y[particle], .z = velocity_z[particle]};
            speeds[particle]     = ::cuda::std::sqrt(velocity_x[particle] * velocity_x[particle] + velocity_y[particle] * velocity_y[particle] + velocity_z[particle] * velocity_z[particle]);
        }

        __global__ void write_apic_particles_kernel(const std::uint32_t particle_count, const float offset_x, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, const float* c00, const float* c01, const float* c02, const float* c10, const float* c11, const float* c12, const float* c20, const float* c21, const float* c22, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities, float* speeds, float* affine_magnitudes) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            positions[particle]         = {.x = position_x[particle] + offset_x, .y = position_y[particle], .z = position_z[particle]};
            velocities[particle]        = {.x = velocity_x[particle], .y = velocity_y[particle], .z = velocity_z[particle]};
            speeds[particle]            = ::cuda::std::sqrt(velocity_x[particle] * velocity_x[particle] + velocity_y[particle] * velocity_y[particle] + velocity_z[particle] * velocity_z[particle]);
            affine_magnitudes[particle] = ::cuda::std::sqrt(c00[particle] * c00[particle] + c01[particle] * c01[particle] + c02[particle] * c02[particle] + c10[particle] * c10[particle] + c11[particle] * c11[particle] + c12[particle] * c12[particle] + c20[particle] * c20[particle] + c21[particle] * c21[particle] + c22[particle] * c22[particle]);
        }
    } // namespace

    void write_flip_particles(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float offset_x, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities, float* speeds) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), write_flip_particles_kernel, particle_count, offset_x, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, positions, velocities, speeds);
    }

    void write_apic_particles(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float offset_x, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, const float* c00, const float* c01, const float* c02, const float* c10, const float* c11, const float* c12, const float* c20, const float* c21, const float* c22, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities, float* speeds, float* affine_magnitudes) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), write_apic_particles_kernel, particle_count, offset_x, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, c00, c01, c02, c10, c11, c12, c20, c21, c22, positions, velocities, speeds, affine_magnitudes);
    }
} // namespace physica::examples::flip_apic_dam_break::spectra_cuda
