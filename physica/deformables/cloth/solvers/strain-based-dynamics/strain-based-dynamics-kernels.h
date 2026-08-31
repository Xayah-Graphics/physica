#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_STRAIN_BASED_DYNAMICS_STRAIN_BASED_DYNAMICS_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_STRAIN_BASED_DYNAMICS_STRAIN_BASED_DYNAMICS_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::strain_based_dynamics::kernels {
    void project_strain(::cuda::stream_ref stream, std::uint32_t triangle_count, std::uint32_t color_offset, const std::uint32_t* colored_triangles, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const float* inverse_rest_00, const float* inverse_rest_01, const float* inverse_rest_10, const float* inverse_rest_11, const std::uint32_t* fixed_vertex_mask, const float* masses, float stretch_stiffness_u, float stretch_stiffness_v, float shear_stiffness, simulation::VectorView<float> positions);
} // namespace physica::deformables::cloth::solvers::strain_based_dynamics::kernels

#endif
