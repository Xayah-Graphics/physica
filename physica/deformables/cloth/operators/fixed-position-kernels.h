#ifndef PHYSICA_DEFORMABLES_CLOTH_OPERATORS_FIXED_POSITION_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_OPERATORS_FIXED_POSITION_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::kernels {
    void fixed_position_forward(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, simulation::VectorView<const float> anchor_positions, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::VectorView<float> constrained_positions, simulation::VectorView<float> constrained_velocities);
    void fixed_position_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, simulation::VectorView<const float> position_tangent, simulation::VectorView<const float> velocity_tangent, simulation::VectorView<float> constrained_position_tangent, simulation::VectorView<float> constrained_velocity_tangent);
    void fixed_position_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, simulation::VectorView<const double> constrained_position_adjoint, simulation::VectorView<const double> constrained_velocity_adjoint, simulation::VectorView<double> position_adjoint, simulation::VectorView<double> velocity_adjoint);
} // namespace physica::deformables::cloth::kernels

#endif
