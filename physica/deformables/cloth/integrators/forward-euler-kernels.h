#ifndef PHYSICA_DEFORMABLES_CLOTH_INTEGRATORS_FORWARD_EULER_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_INTEGRATORS_FORWARD_EULER_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::kernels {
    void forward_euler_forward(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::VectorView<const float> forces, const float* masses, simulation::VectorView<float> integrated_positions, simulation::VectorView<float> integrated_velocities);
} // namespace physica::deformables::cloth::kernels

#endif
