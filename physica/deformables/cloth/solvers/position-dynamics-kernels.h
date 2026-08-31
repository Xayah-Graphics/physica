#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_POSITION_DYNAMICS_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_POSITION_DYNAMICS_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::position_dynamics::kernels {
    void predict(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, Vector3<float> gravity, const std::uint32_t* fixed_vertex_mask, simulation::VectorView<const float> fixed_positions, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::VectorView<const float> external_forces, const float* masses, simulation::VectorView<float> predicted_positions);
    void reconstruct_velocities(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> previous_positions, simulation::VectorView<const float> positions, simulation::VectorView<float> velocities);
} // namespace physica::deformables::cloth::solvers::position_dynamics::kernels

#endif
