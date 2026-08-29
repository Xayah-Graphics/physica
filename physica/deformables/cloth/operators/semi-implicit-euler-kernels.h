#ifndef PHYSICA_DEFORMABLES_CLOTH_OPERATORS_SEMI_IMPLICIT_EULER_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_OPERATORS_SEMI_IMPLICIT_EULER_KERNELS_H

#include <physica/field/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::deformables::cloth::kernels {
    void semi_implicit_euler_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, field::VectorView<const float> positions, field::VectorView<const float> velocities, field::VectorView<const float> forces, const float* masses, field::VectorView<float> integrated_positions, field::VectorView<float> integrated_velocities);
    void semi_implicit_euler_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, field::VectorView<const float> forces, const float* masses, field::VectorView<const float> position_tangent, field::VectorView<const float> velocity_tangent, field::VectorView<const float> force_tangent, const float* mass_tangent, field::VectorView<float> integrated_position_tangent, field::VectorView<float> integrated_velocity_tangent);
    void semi_implicit_euler_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, field::VectorView<const float> forces, const float* masses, field::VectorView<const double> integrated_position_adjoint, field::VectorView<const double> integrated_velocity_adjoint, field::VectorView<double> position_adjoint, field::VectorView<double> velocity_adjoint, field::VectorView<double> force_adjoint, double* mass_adjoint);
} // namespace physica::deformables::cloth::kernels

#endif
