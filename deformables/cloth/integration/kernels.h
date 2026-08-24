#ifndef PHYSICA_DEFORMABLES_CLOTH_INTEGRATION_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_INTEGRATION_KERNELS_H

#include "../domain/device.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::deformables::cloth::cuda_detail {
    void semi_implicit_euler_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, ConstFieldView<float> positions, ConstFieldView<float> velocities, ConstFieldView<float> forces, const float* masses, FieldView<float> integrated_positions, FieldView<float> integrated_velocities);
    void semi_implicit_euler_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, ConstFieldView<float> forces, const float* masses, ConstFieldView<float> position_tangent, ConstFieldView<float> velocity_tangent, ConstFieldView<float> force_tangent, const float* mass_tangent, FieldView<float> integrated_position_tangent, FieldView<float> integrated_velocity_tangent);
    void semi_implicit_euler_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, ConstFieldView<float> forces, const float* masses, ConstFieldView<double> integrated_position_adjoint, ConstFieldView<double> integrated_velocity_adjoint, FieldView<double> position_adjoint, FieldView<double> velocity_adjoint, FieldView<double> force_adjoint, double* mass_adjoint);
} // namespace physica::deformables::cloth::cuda_detail

#endif
