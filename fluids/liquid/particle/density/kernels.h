#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_DENSITY_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_DENSITY_KERNELS_H

#include "../domain/device.h"
#include "../neighborhood/device.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::liquid::particle::cuda_detail {
    void density_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, bool pbf_kernel, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, float* densities);
    void density_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, bool pbf_kernel, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ConstVectorView<float> position_tangent, ParticleParameterView parameters, ParticleParameterTangentView parameter_tangent, NeighborhoodView neighborhood, BoundaryView boundary, float* density_tangent);
    void density_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, float support_radius, bool pbf_kernel, ConstVectorView<float> topology_positions, ConstVectorView<float> positions, ParticleParameterView parameters, NeighborhoodView neighborhood, BoundaryView boundary, const double* density_adjoint, VectorView<double> position_adjoint, ParticleParameterAdjointView parameter_adjoint);
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
