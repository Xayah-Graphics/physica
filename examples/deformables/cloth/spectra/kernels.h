#ifndef PHYSICA_EXAMPLES_DEFORMABLES_CLOTH_SPECTRA_KERNELS_H
#define PHYSICA_EXAMPLES_DEFORMABLES_CLOTH_SPECTRA_KERNELS_H

#include <cstdint>
#include <cuda/stream>
#include <spectra/sdk/cuda_types.h>

namespace physica::examples::cloth::spectra_cuda {
    void write_surface(::cuda::stream_ref stream, std::uint32_t rows, std::uint32_t columns, const float* position_x, const float* position_y, const float* position_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* normals);
} // namespace physica::examples::cloth::spectra_cuda

#endif
