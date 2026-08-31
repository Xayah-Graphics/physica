#ifndef PHYSICA_DEFORMABLES_CLOTH_CONSTRAINTS_ANALYTIC_COLLISION_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_CONSTRAINTS_ANALYTIC_COLLISION_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::kernels {
    void analytic_collision_initialize(::cuda::stream_ref stream, std::uint32_t particle_count, simulation::VectorView<const float> integrated_positions, simulation::VectorView<const float> integrated_velocities, simulation::VectorView<float> constrained_positions, simulation::VectorView<float> constrained_velocities);
    void analytic_collision_plane(::cuda::stream_ref stream, std::uint32_t particle_count, Vector3<float> normal, float offset, float thickness, float restitution, float friction, simulation::VectorView<float> positions, simulation::VectorView<float> velocities);
    void analytic_collision_sphere(::cuda::stream_ref stream, std::uint32_t particle_count, Vector3<float> center, float radius, float thickness, float restitution, float friction, simulation::VectorView<float> positions, simulation::VectorView<float> velocities);
} // namespace physica::deformables::cloth::kernels

#endif
