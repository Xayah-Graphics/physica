#ifndef PHYSICA_EXAMPLES_FLUIDS_LIQUID_FLIP_APIC_DAM_BREAK_SPECTRA_KERNELS_H
#define PHYSICA_EXAMPLES_FLUIDS_LIQUID_FLIP_APIC_DAM_BREAK_SPECTRA_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <spectra/sdk/cuda_types.h>

namespace physica::examples::flip_apic_dam_break::spectra_cuda {
    void write_flip_particles(::cuda::stream_ref stream, std::uint32_t particle_count, float offset_x, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities, float* speeds);
    void write_apic_particles(::cuda::stream_ref stream, std::uint32_t particle_count, float offset_x, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, const float* c00, const float* c01, const float* c02, const float* c10, const float* c11, const float* c12, const float* c20, const float* c21, const float* c22, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities, float* speeds, float* affine_magnitudes);
} // namespace physica::examples::flip_apic_dam_break::spectra_cuda

#endif
