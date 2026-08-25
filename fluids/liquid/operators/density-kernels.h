#ifndef PHYSICA_FLUIDS_LIQUID_OPERATORS_DENSITY_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_OPERATORS_DENSITY_KERNELS_H

#include "../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::liquid::cuda_detail {
    void cubic_density_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, float* densities);
    void cubic_density_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ConstVectorView<float> position_tangent, ParticleParameterView parameters, ParticleParameterTangentView parameter_tangent, NeighborhoodView neighborhood, BoundaryView boundary, float* density_tangent);
    void cubic_density_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, const double* density_adjoint, VectorView<double> position_adjoint, ParticleParameterAdjointView parameter_adjoint);
    void poly6_density_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, float* densities);
    void poly6_density_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ConstVectorView<float> position_tangent, ParticleParameterView parameters, ParticleParameterTangentView parameter_tangent, NeighborhoodView neighborhood, BoundaryView boundary, float* density_tangent);
    void poly6_density_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, const double* density_adjoint, VectorView<double> position_adjoint, ParticleParameterAdjointView parameter_adjoint);
} // namespace physica::fluids::liquid::cuda_detail

#endif
