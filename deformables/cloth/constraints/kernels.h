#ifndef PHYSICA_DEFORMABLES_CLOTH_CONSTRAINTS_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_CONSTRAINTS_KERNELS_H

#include "../domain/device.h"
#include <cstdint>
#include <cuda/stream>

namespace physica::deformables::cloth::cuda_detail {
    void fixed_position_forward(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, ConstFieldView<float> anchor_positions, ConstFieldView<float> positions, ConstFieldView<float> velocities, FieldView<float> constrained_positions, FieldView<float> constrained_velocities);
    void fixed_position_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, ConstFieldView<float> position_tangent, ConstFieldView<float> velocity_tangent, FieldView<float> constrained_position_tangent, FieldView<float> constrained_velocity_tangent);
    void fixed_position_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, ConstFieldView<double> constrained_position_adjoint, ConstFieldView<double> constrained_velocity_adjoint, FieldView<double> position_adjoint, FieldView<double> velocity_adjoint);
} // namespace physica::deformables::cloth::cuda_detail

#endif
