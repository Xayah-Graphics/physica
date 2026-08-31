#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_XPBD_XPBD_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_XPBD_XPBD_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::xpbd::kernels {
    void project(::cuda::stream_ref stream, std::uint32_t constraint_count, std::uint32_t color_offset, float inverse_time_step_squared, const std::uint32_t* colored_constraints, const std::uint32_t* constraint_first, const std::uint32_t* constraint_second, const float* rest_lengths, const float* compliances, const std::uint32_t* fixed_vertex_mask, const float* masses, float* lambdas, simulation::VectorView<float> positions);
} // namespace physica::deformables::cloth::solvers::xpbd::kernels

#endif
