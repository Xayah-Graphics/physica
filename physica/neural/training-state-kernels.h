#ifndef PHYSICA_NEURAL_TRAINING_STATE_KERNELS_H
#define PHYSICA_NEURAL_TRAINING_STATE_KERNELS_H

#include <physica/cuda_stream.h>

#include <cstddef>
#include <cstdint>

namespace physica::neural::kernels {
    void optimize(::cuda::stream_ref stream, float* parameters, float* gradients, float* first_moments, float* second_moments, float* ema, std::size_t count, float learning_rate, float first_decay, float second_decay, float first_correction, float second_correction, float epsilon, float weight_decay, float exponential_average_decay);
    void optimize(::cuda::stream_ref stream, float* parameters, float* gradients, float* first_moments, float* second_moments, float* ema, std::size_t count, float learning_rate, float first_decay, float second_decay, const std::uint64_t* step, const std::uint64_t* processed_samples, float* step_scalars, float epsilon, float weight_decay, std::uint32_t samples_per_step, std::uint64_t half_life_samples, float ramp_up_ratio);
} // namespace physica::neural::kernels

#endif
