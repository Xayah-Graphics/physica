#ifndef PHYSICA_FLUIDS_LIQUID_OPERATORS_DENSITY_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_OPERATORS_DENSITY_KERNELS_H

#include <physica/fluids/liquid/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::liquid::operators::kernels::density {
    void cubic_density_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, field::VectorView<const float> topology_positions, field::VectorView<const float> positions, device::ParticleParameterView parameters, device::NeighborhoodView neighborhood, device::BoundaryView boundary, float* densities);
    void cubic_density_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, field::VectorView<const float> topology_positions, field::VectorView<const float> positions, field::VectorView<const float> position_tangent, device::ParticleParameterView parameters, device::ParticleParameterTangentView parameter_tangent, device::NeighborhoodView neighborhood, device::BoundaryView boundary, float* density_tangent);
    void cubic_density_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, field::VectorView<const float> topology_positions, field::VectorView<const float> positions, device::ParticleParameterView parameters, device::NeighborhoodView neighborhood, device::BoundaryView boundary, const double* density_adjoint, field::VectorView<double> position_adjoint, device::ParticleParameterAdjointView parameter_adjoint);
    void poly6_density_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, field::VectorView<const float> topology_positions, field::VectorView<const float> positions, device::ParticleParameterView parameters, device::NeighborhoodView neighborhood, device::BoundaryView boundary, float* densities);
    void poly6_density_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, field::VectorView<const float> topology_positions, field::VectorView<const float> positions, field::VectorView<const float> position_tangent, device::ParticleParameterView parameters, device::ParticleParameterTangentView parameter_tangent, device::NeighborhoodView neighborhood, device::BoundaryView boundary, float* density_tangent);
    void poly6_density_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, field::VectorView<const float> topology_positions, field::VectorView<const float> positions, device::ParticleParameterView parameters, device::NeighborhoodView neighborhood, device::BoundaryView boundary, const double* density_adjoint, field::VectorView<double> position_adjoint, device::ParticleParameterAdjointView parameter_adjoint);
} // namespace physica::fluids::liquid::operators::kernels::density

#endif
