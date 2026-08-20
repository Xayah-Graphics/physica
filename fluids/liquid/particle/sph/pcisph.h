#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_SPH_PCISPH_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_SPH_PCISPH_H

#include "kernels.h"

#include <cstdint>

namespace physica::fluids::liquid::particle::cuda_detail::pcisph {

    void launch_predict_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, Box domain, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> non_pressure_accelerations, ConstVectorView<float> pressure_accelerations, VectorView<float> predicted_positions, VectorView<float> predicted_velocities);
    void launch_predict_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, Box domain, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> non_pressure_accelerations, ConstVectorView<float> pressure_accelerations, ConstVectorView<float> position_tangent, ConstVectorView<float> velocity_tangent, ConstVectorView<float> non_pressure_acceleration_tangent, ConstVectorView<float> pressure_acceleration_tangent, VectorView<float> predicted_position_tangent, VectorView<float> predicted_velocity_tangent);
    void launch_predict_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, Box domain, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> non_pressure_accelerations, ConstVectorView<float> pressure_accelerations, ConstVectorView<double> predicted_position_adjoint, ConstVectorView<double> predicted_velocity_adjoint, VectorView<double> position_adjoint, VectorView<double> velocity_adjoint, VectorView<double> non_pressure_acceleration_adjoint, VectorView<double> pressure_acceleration_adjoint);

    void launch_pressure_update_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, float reference_gradient_norm, ParticleParameterView particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, float* pressures);
    void launch_pressure_update_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, float reference_gradient_norm, ParticleParameterView particles, ParticleParameterTangentView particle_tangent, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const float* previous_pressure_tangent, const float* predicted_density_tangent, const float* pressure_relaxation_tangent, float* pressure_tangent);
    void launch_pressure_update_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, float reference_gradient_norm, ParticleParameterView particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const double* pressure_adjoint, ParticleParameterAdjointView particle_adjoint, double* previous_pressure_adjoint, double* predicted_density_adjoint, double* pressure_relaxation_adjoint);


} // namespace physica::fluids::liquid::particle::cuda_detail::pcisph

#endif
