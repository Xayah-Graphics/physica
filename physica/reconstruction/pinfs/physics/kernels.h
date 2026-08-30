#ifndef PHYSICA_RECONSTRUCTION_PINFS_PHYSICS_KERNELS_H
#define PHYSICA_RECONSTRUCTION_PINFS_PHYSICS_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <math/math.h>

namespace physica::reconstruction::pinfs::kernels {
    void sample_physics(::cuda::stream_ref stream, const Vector3<float>* voxel_positions, std::uint32_t voxel_count, float* points, float* derivatives, Vector3<float>* positions, std::uint32_t sample_count, float time, std::uint32_t seed, std::uint32_t step);
    void physics_loss(::cuda::stream_ref stream, const float* density, const float* density_derivatives, const float* velocity, const float* velocity_derivatives, const float* static_sdf, const float* static_sdf_derivatives, const float* inverse_deviation, float* velocity_adjoints, float* velocity_derivative_adjoints, double* losses, std::uint32_t sample_count, float physics_weight, float velocity_fading, float neumann_weight);
} // namespace physica::reconstruction::pinfs::kernels

#endif
