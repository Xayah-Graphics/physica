#ifndef PHYSICA_RECONSTRUCTION_INSTANT_NGP_CUDA_RANDOM_CUH
#define PHYSICA_RECONSTRUCTION_INSTANT_NGP_CUDA_RANDOM_CUH

#include <cstdint>
#include <cuda/std/random>

namespace physica::reconstruction::instant_ngp::cuda_detail {
enum class RandomStream : std::uint32_t {
    density_grid = 0u,
    training_pixel = 1u,
    raymarch = 2u,
    background = 3u,
    mlp_parameters = 4u,
    grid_parameters = 5u,
};

inline __host__ __device__ ::cuda::std::philox4x32 make_random_engine(const std::uint32_t seed, const RandomStream stream, const std::uint32_t phase, const std::uint32_t index) {
    ::cuda::std::philox4x32 random{seed};
    random.set_counter({static_cast<std::uint32_t>(stream), phase, index, 0u});
    return random;
}

inline __host__ __device__ float random_float(::cuda::std::philox4x32& random) {
    return static_cast<float>(random() >> 8u) * 0x1p-24F;
}
} // namespace physica::reconstruction::instant_ngp::cuda_detail

#endif
