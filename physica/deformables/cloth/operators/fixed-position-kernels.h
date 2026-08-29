#ifndef PHYSICA_DEFORMABLES_CLOTH_OPERATORS_FIXED_POSITION_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_OPERATORS_FIXED_POSITION_KERNELS_H

#include <physica/field/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::deformables::cloth::kernels {
    void fixed_position_forward(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, field::VectorView<const float> anchor_positions, field::VectorView<const float> positions, field::VectorView<const float> velocities, field::VectorView<float> constrained_positions, field::VectorView<float> constrained_velocities);
    void fixed_position_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, field::VectorView<const float> position_tangent, field::VectorView<const float> velocity_tangent, field::VectorView<float> constrained_position_tangent, field::VectorView<float> constrained_velocity_tangent);
    void fixed_position_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, field::VectorView<const double> constrained_position_adjoint, field::VectorView<const double> constrained_velocity_adjoint, field::VectorView<double> position_adjoint, field::VectorView<double> velocity_adjoint);
} // namespace physica::deformables::cloth::kernels

#endif
