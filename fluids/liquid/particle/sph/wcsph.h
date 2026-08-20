#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_SPH_WCSPH_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_SPH_WCSPH_H

#include "kernels.h"

#include <cstdint>

namespace physica::fluids::liquid::particle::cuda_detail::wcsph {

    void launch_external_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float gravity_x, float gravity_y, float gravity_z, ConstVectorView<float> controls, VectorView<float> accelerations);

    void launch_eos_forward(::cuda::stream_ref stream, std::uint32_t particle_count, const float* densities, ParticleParameterView particles, const float* speed_of_sound, const float* tait_exponent, float* pressures);
    void launch_eos_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, const float* densities, const float* density_tangent, ParticleParameterView particles, ParticleParameterTangentView particle_tangent, const float* speed_of_sound, const float* speed_of_sound_tangent, const float* tait_exponent, const float* tait_exponent_tangent, float* pressure_tangent);
    void launch_eos_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, const float* densities, ParticleParameterView particles, const float* speed_of_sound, const float* tait_exponent, const double* pressure_adjoint, double* density_adjoint, ParticleParameterAdjointView particle_adjoint, double* speed_of_sound_adjoint, double* tait_exponent_adjoint);

    void launch_artificial_viscosity_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> positions, ConstVectorView<float> velocities, ParticleParameterView particles, const float* speed_of_sound, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, VectorView<float> accelerations);
    void launch_artificial_viscosity_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> positions, ConstVectorView<float> velocities, ConstVectorView<float> position_tangent, ConstVectorView<float> velocity_tangent, ParticleParameterView particles, ParticleParameterTangentView particle_tangent, const float* speed_of_sound, const float* speed_of_sound_tangent, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, const float* density_tangent, VectorView<float> acceleration_tangent);
    void launch_artificial_viscosity_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> positions, ConstVectorView<float> velocities, ParticleParameterView particles, const float* speed_of_sound, NeighborhoodView neighborhood, BoundaryView boundary, const float* densities, ConstVectorView<double> acceleration_adjoint, VectorView<double> position_adjoint, VectorView<double> velocity_adjoint, double* density_adjoint, ParticleParameterAdjointView particle_adjoint, double* speed_of_sound_adjoint);

    void launch_surface_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, float particle_radius, ConstVectorView<float> positions, ParticleParameterView particles, const float* boundary_surface_tension, NeighborhoodView neighborhood, BoundaryView boundary, VectorView<float> accelerations);
    void launch_surface_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, float particle_radius, ConstVectorView<float> positions, ConstVectorView<float> position_tangent, ParticleParameterView particles, ParticleParameterTangentView particle_tangent, const float* boundary_surface_tension, const float* boundary_surface_tension_tangent, NeighborhoodView neighborhood, BoundaryView boundary, VectorView<float> acceleration_tangent);
    void launch_surface_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, float particle_radius, ConstVectorView<float> positions, ParticleParameterView particles, const float* boundary_surface_tension, NeighborhoodView neighborhood, BoundaryView boundary, ConstVectorView<double> acceleration_adjoint, VectorView<double> position_adjoint, ParticleParameterAdjointView particle_adjoint, double* boundary_surface_tension_adjoint);

} // namespace physica::fluids::liquid::particle::cuda_detail::wcsph

#endif
