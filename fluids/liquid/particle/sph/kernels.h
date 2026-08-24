#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_SPH_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_SPH_KERNELS_H

#include "../domain/device.h"
#include "../neighborhood/device.h"
#include <physica/cuda_stream.h>
#include <cstdint>

namespace physica::fluids::liquid::particle::cuda_detail::sph {
    void non_pressure_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, float gravity_x, float gravity_y, float gravity_z, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> external_accelerations, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, VectorView<float> accelerations);
    void non_pressure_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> external_acceleration_tangent, ConstVectorView<float> position_tangent, ConstVectorView<float> velocity_tangent, ParticleParameterView parameters, ParticleParameterTangentView parameter_tangent, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, const float* density_tangent, VectorView<float> acceleration_tangent);
    void non_pressure_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> positions, ConstVectorView<float> velocities, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, ConstVectorView<double> acceleration_adjoint, VectorView<double> position_adjoint, VectorView<double> velocity_adjoint, VectorView<double> control_adjoint, double* density_adjoint, ParticleParameterAdjointView parameter_adjoint);

    void pressure_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> positions, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, const float* pressures, VectorView<float> accelerations);
    void pressure_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> positions, ConstVectorView<float> position_tangent, ParticleParameterView parameters, ParticleParameterTangentView parameter_tangent, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, const float* density_tangent, const float* pressures, const float* pressure_tangent, VectorView<float> acceleration_tangent);
    void pressure_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> positions, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, const float* pressures, ConstVectorView<double> acceleration_adjoint, VectorView<double> position_adjoint, double* density_adjoint, double* pressure_adjoint, ParticleParameterAdjointView parameter_adjoint);

    void integrate_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, Box domain, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> accelerations, VectorView<float> next_positions, VectorView<float> next_velocities);
    void integrate_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, Box domain, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> accelerations, ConstVectorView<float> position_tangent, ConstVectorView<float> velocity_tangent, ConstVectorView<float> acceleration_tangent, VectorView<float> next_position_tangent, VectorView<float> next_velocity_tangent);
    void integrate_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, Box domain, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> accelerations, ConstVectorView<double> next_position_adjoint, ConstVectorView<double> next_velocity_adjoint, VectorView<double> position_adjoint, VectorView<double> velocity_adjoint, VectorView<double> acceleration_adjoint);

    void add(::cuda::stream_ref stream, std::uint32_t particle_count, ConstVectorView<float> first, ConstVectorView<float> second, VectorView<float> output);
    void add_adjoint(::cuda::stream_ref stream, std::uint32_t particle_count, ConstVectorView<double> output_adjoint, VectorView<double> first_adjoint, VectorView<double> second_adjoint);
} // namespace physica::fluids::liquid::particle::cuda_detail::sph

#endif
