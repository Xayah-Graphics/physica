#ifndef PHYSICA_RECONSTRUCTION_INSTANT_NGP_CUDA_DENSITY_ACTIVATION_CUH
#define PHYSICA_RECONSTRUCTION_INSTANT_NGP_CUDA_DENSITY_ACTIVATION_CUH

#include <cuda/std/cmath>

namespace physica::reconstruction::instant_ngp::cuda_detail {
inline __device__ float exponential_density(const float value) {
    return ::cuda::std::exp(value);
}
} // namespace physica::reconstruction::instant_ngp::cuda_detail

#endif
