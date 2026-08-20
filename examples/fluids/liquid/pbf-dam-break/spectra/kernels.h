#ifndef PHYSICA_EXAMPLES_FLUIDS_LIQUID_PBF_DAM_BREAK_SPECTRA_KERNELS_H
#define PHYSICA_EXAMPLES_FLUIDS_LIQUID_PBF_DAM_BREAK_SPECTRA_KERNELS_H

#include <cstdint>
#include <cuda/stream>
#include <spectra/sdk/cuda_types.h>

namespace physica::examples::pbf_dam_break::spectra_cuda {
    void write_vectors(::cuda::stream_ref stream, std::uint32_t particle_count, const float* position_x, const float* position_y, const float* position_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* velocities);
} // namespace physica::examples::pbf_dam_break::spectra_cuda

#endif
