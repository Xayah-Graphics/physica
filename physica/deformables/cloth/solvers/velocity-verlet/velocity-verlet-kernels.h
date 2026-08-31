#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_VELOCITY_VERLET_VELOCITY_VERLET_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_VELOCITY_VERLET_VELOCITY_VERLET_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::velocity_verlet::kernels {
    void velocity_verlet_predict(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, const float* masses, simulation::VectorView<const float> forces, simulation::VectorView<float> predicted_positions, simulation::VectorView<float> predicted_velocities);
    void velocity_verlet_predict(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const double> positions, simulation::VectorView<const double> velocities, const double* masses, simulation::VectorView<const double> forces, simulation::VectorView<double> predicted_positions, simulation::VectorView<double> predicted_velocities);
    void velocity_verlet_second_half_kick(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> predicted_positions, simulation::VectorView<const float> predicted_velocities, const float* masses, simulation::VectorView<const float> forces, simulation::VectorView<float> final_positions, simulation::VectorView<float> final_velocities);
    void velocity_verlet_second_half_kick(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const double> predicted_positions, simulation::VectorView<const double> predicted_velocities, const double* masses, simulation::VectorView<const double> forces, simulation::VectorView<double> final_positions, simulation::VectorView<double> final_velocities);
} // namespace physica::deformables::cloth::solvers::velocity_verlet::kernels

#endif
