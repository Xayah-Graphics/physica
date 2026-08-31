#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_FAST_MASS_SPRING_FAST_MASS_SPRING_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_FAST_MASS_SPRING_FAST_MASS_SPRING_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::fast_mass_spring::kernels {
    void project_springs(::cuda::stream_ref stream, std::uint32_t edge_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, simulation::VectorView<const float> positions, simulation::VectorView<float> projected_springs);
    void assemble_right_hand_sides(::cuda::stream_ref stream, std::uint32_t free_particle_count, float inverse_time_step_squared, float spring_stiffness, const std::uint32_t* free_particles, const std::uint32_t* vertex_edge_offsets, const std::uint32_t* vertex_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* fixed_vertex_mask, const float* masses, simulation::VectorView<const float> predicted_positions, simulation::VectorView<const float> fixed_positions, simulation::VectorView<const float> projected_springs, float* right_hand_sides);
    void scatter_solution(::cuda::stream_ref stream, std::uint32_t free_particle_count, const std::uint32_t* free_particles, const float* solutions, simulation::VectorView<float> positions);
} // namespace physica::deformables::cloth::solvers::fast_mass_spring::kernels

#endif
