#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_PROJECTIVE_DYNAMICS_PROJECTIVE_DYNAMICS_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_PROJECTIVE_DYNAMICS_PROJECTIVE_DYNAMICS_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::projective_dynamics::kernels {
    void project_deformation_gradients(::cuda::stream_ref stream, std::uint32_t deformation_gradient_count, simulation::VectorView<const float> deformation_gradient_first_columns, simulation::VectorView<const float> deformation_gradient_second_columns, simulation::VectorView<float> projected_frame_first_columns, simulation::VectorView<float> projected_frame_second_columns);
    void project_membranes(::cuda::stream_ref stream, std::uint32_t triangle_count, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const float* material_inverse_00, const float* material_inverse_01, const float* material_inverse_10, const float* material_inverse_11, simulation::VectorView<const float> positions, simulation::VectorView<float> projected_frame_first_columns, simulation::VectorView<float> projected_frame_second_columns);
    void project_bending(::cuda::stream_ref stream, std::uint32_t hinge_count, const std::uint32_t* hinge_first_opposite, const std::uint32_t* hinge_second_opposite, simulation::VectorView<const float> positions, simulation::VectorView<float> bending_directions);
    void assemble_right_hand_sides(::cuda::stream_ref stream, std::uint32_t free_particle_count, float inverse_time_step_squared, float bending_stiffness, const std::uint32_t* free_particles, const std::uint32_t* vertex_triangle_offsets, const std::uint32_t* vertex_triangles, const std::uint32_t* vertex_hinge_offsets, const std::uint32_t* vertex_hinges, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const std::uint32_t* hinge_first_opposite, const std::uint32_t* hinge_second_opposite, const float* material_inverse_00, const float* material_inverse_01, const float* material_inverse_10, const float* material_inverse_11, const float* membrane_weights, const float* bending_rest_lengths, const float* masses, simulation::VectorView<const float> predicted_positions, simulation::VectorView<const float> fixed_right_hand_sides, simulation::VectorView<const float> projected_frame_first_columns, simulation::VectorView<const float> projected_frame_second_columns, simulation::VectorView<const float> bending_directions, float* right_hand_sides);
    void scatter_solution(::cuda::stream_ref stream, std::uint32_t free_particle_count, const std::uint32_t* free_particles, const float* solutions, simulation::VectorView<float> positions);
} // namespace physica::deformables::cloth::solvers::projective_dynamics::kernels

#endif
