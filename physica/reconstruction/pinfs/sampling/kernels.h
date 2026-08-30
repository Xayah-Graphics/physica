#ifndef PHYSICA_RECONSTRUCTION_PINFS_SAMPLING_KERNELS_H
#define PHYSICA_RECONSTRUCTION_PINFS_SAMPLING_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <math/math.h>

namespace physica::reconstruction::pinfs::kernels {
    void sample_coarse(::cuda::stream_ref stream, const Ray3<float>* rays, float* z, Vector3<float>* positions, float* dynamic_points, Vector3<float>* directions, std::uint32_t ray_count, std::uint32_t sample_count, float near_distance, float far_distance, float time, std::uint32_t seed, std::uint32_t step, std::uint32_t random_offset, bool perturb);
    void warp_points(::cuda::stream_ref stream, float* dynamic_points, const float* velocity, std::uint32_t sample_count, float amount, std::uint32_t seed, std::uint32_t step, std::uint32_t random_offset);
    void sample_pdf(::cuda::stream_ref stream, const float* z, const float* weights, float* output, std::uint32_t ray_count, std::uint32_t z_count, std::uint32_t weight_offset, std::uint32_t weight_count, std::uint32_t output_count, std::uint32_t seed, std::uint32_t step, std::uint32_t ray_offset, bool deterministic);
    void sort_samples(::cuda::stream_ref stream, float* samples, std::uint32_t ray_count, std::uint32_t sample_count);
    void extract_sdf(::cuda::stream_ref stream, const float* sdf_output, float* sdf, std::uint32_t sample_count);
    void neus_pdf(::cuda::stream_ref stream, const float* z, const float* sdf, float* output, std::uint32_t ray_count, std::uint32_t sample_count, std::uint32_t output_count, float inverse_deviation);
    void merge_samples(::cuda::stream_ref stream, const float* first_z, const float* first_sdf, std::uint32_t first_count, const float* second_z, const float* second_sdf, std::uint32_t second_count, float* output_z, float* output_sdf, std::uint32_t ray_count);
    void positions_from_z(::cuda::stream_ref stream, const Ray3<float>* rays, const float* z, Vector3<float>* positions, std::uint32_t ray_count, std::uint32_t sample_count);
    void samples_from_z(::cuda::stream_ref stream, const Ray3<float>* rays, const float* z, Vector3<float>* positions, float* dynamic_points, Vector3<float>* directions, std::uint32_t ray_count, std::uint32_t sample_count, float time);
} // namespace physica::reconstruction::pinfs::kernels

#endif
