#ifndef PHYSICA_EXAMPLES_DEFORMABLES_CLOTH_FLAG_MODULE_KERNELS_H
#define PHYSICA_EXAMPLES_DEFORMABLES_CLOTH_FLAG_MODULE_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <spectra/sdk/cuda_types.h>

namespace physica::examples::cloth_flag::module_cuda {
    void write_vectors(::cuda::stream_ref stream, std::uint32_t count, const float* x, const float* y, const float* z, spectra::sdk::Float3* output);
    void write_strain(::cuda::stream_ref stream, std::uint32_t particle_count, std::uint32_t edge_count, const std::uint32_t* first, const std::uint32_t* second, const float* rest_lengths, const float* position_x, const float* position_y, const float* position_z, float* strain);
} // namespace physica::examples::cloth_flag::module_cuda

#endif
