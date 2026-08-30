#ifndef PHYSICA_GENERATIVE_FLOW_MATCHING_KERNELS_H
#define PHYSICA_GENERATIVE_FLOW_MATCHING_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::generative::flow_matching::kernels {
    void make_training_batch(::cuda::stream_ref stream, const std::uint8_t* images, const std::uint8_t* dataset_labels, float* path, float* target, float* times, std::uint8_t* labels, const std::uint64_t* step, const std::uint64_t* seed, std::uint32_t batch);
    void make_sampling_noise(::cuda::stream_ref stream, float* state, std::uint64_t seed, std::uint32_t batch);
    void make_sampling_time(::cuda::stream_ref stream, float* times, float time, std::uint32_t batch);
    void make_time_embedding(::cuda::stream_ref stream, const float* times, float* embedding, std::uint32_t batch, std::uint32_t width);
    void silu_forward(::cuda::stream_ref stream, const float* input, float* output, std::size_t count);
    void silu_backward(::cuda::stream_ref stream, const float* input, const float* output_gradient, float* input_gradient, std::size_t count);
    void add_position(::cuda::stream_ref stream, float* tokens, const float* position, std::size_t count);
    void make_condition(::cuda::stream_ref stream, float* time_condition, const float* class_embedding, const std::uint8_t* labels, std::uint32_t batch, std::uint32_t width);
    void class_embedding_backward(::cuda::stream_ref stream, const float* condition_gradient, const std::uint8_t* labels, float* class_embedding_gradient, std::uint32_t batch, std::uint32_t width);
    void final_adaln_forward(::cuda::stream_ref stream, const float* input, const float* modulation, float* output, float* means, float* inverse_standard_deviations, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width);
    void final_adaln_backward(::cuda::stream_ref stream, const float* input, const float* modulation, const float* output_gradient, const float* means, const float* inverse_standard_deviations, float* input_gradient, float* modulation_gradient, std::uint32_t batch, std::uint32_t sequence, std::uint32_t width);
    void flow_matching_loss(::cuda::stream_ref stream, const float* prediction, const float* target, float* prediction_gradient, float* sample_loss, float* loss, std::uint32_t batch);
    void add_loss(::cuda::stream_ref stream, const float* loss, float* loss_sum);
    void advance_training_state(::cuda::stream_ref stream, std::uint64_t* step, std::uint64_t* processed_samples, std::uint32_t samples_per_step);
    void make_labels(::cuda::stream_ref stream, std::uint8_t* labels, std::uint32_t batch, std::uint32_t class_index);
    void combine_guidance(::cuda::stream_ref stream, const float* conditional, const float* unconditional, float* output, float guidance, std::size_t count);
    void euler_step(::cuda::stream_ref stream, float* state, const float* velocity, float step_size, std::size_t count);
    void heun_predict(::cuda::stream_ref stream, const float* state, const float* velocity, float* prediction, float step_size, std::size_t count);
    void heun_step(::cuda::stream_ref stream, float* state, const float* first_velocity, const float* second_velocity, float step_size, std::size_t count);
    void rk4_intermediate(::cuda::stream_ref stream, const float* state, const float* velocity, float* intermediate, float step_size, std::size_t count);
    void rk4_step(::cuda::stream_ref stream, float* state, const float* first, const float* second, const float* third, const float* fourth, float step_size, std::size_t count);
    void unpatchify(::cuda::stream_ref stream, const float* patches, std::uint8_t* rgba, std::uint32_t batch);
} // namespace physica::generative::flow_matching::kernels

#endif
