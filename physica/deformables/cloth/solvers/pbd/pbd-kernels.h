#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_PBD_PBD_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_PBD_PBD_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::pbd::kernels {
    void project_distance(::cuda::stream_ref stream, std::uint32_t edge_count, std::uint32_t color_offset, const std::uint32_t* colored_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, const float* masses, simulation::VectorView<float> positions);
} // namespace physica::deformables::cloth::solvers::pbd::kernels

#endif
