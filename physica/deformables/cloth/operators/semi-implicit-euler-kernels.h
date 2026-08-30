#ifndef PHYSICA_DEFORMABLES_CLOTH_OPERATORS_SEMI_IMPLICIT_EULER_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_OPERATORS_SEMI_IMPLICIT_EULER_KERNELS_H

#include <simulation/field/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::deformables::cloth::kernels {
    void semi_implicit_euler_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::VectorView<const float> forces, const float* masses, simulation::VectorView<float> integrated_positions, simulation::VectorView<float> integrated_velocities);
    void semi_implicit_euler_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> forces, const float* masses, simulation::VectorView<const float> position_tangent, simulation::VectorView<const float> velocity_tangent, simulation::VectorView<const float> force_tangent, const float* mass_tangent, simulation::VectorView<float> integrated_position_tangent, simulation::VectorView<float> integrated_velocity_tangent);
    void semi_implicit_euler_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> forces, const float* masses, simulation::VectorView<const double> integrated_position_adjoint, simulation::VectorView<const double> integrated_velocity_adjoint, simulation::VectorView<double> position_adjoint, simulation::VectorView<double> velocity_adjoint, simulation::VectorView<double> force_adjoint, double* mass_adjoint);
} // namespace physica::deformables::cloth::kernels

#endif
