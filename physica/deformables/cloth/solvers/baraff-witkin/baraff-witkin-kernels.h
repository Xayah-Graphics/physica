#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_BARAFF_WITKIN_BARAFF_WITKIN_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_BARAFF_WITKIN_BARAFF_WITKIN_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::baraff_witkin::kernels {
    void initialize_forces(::cuda::stream_ref stream, std::uint32_t particle_count, Vector3<float> gravity, const float* masses, simulation::VectorView<const float> external_forces, simulation::VectorView<float> forces);
    void assemble_triangles(::cuda::stream_ref stream, std::uint32_t batch_count, const std::uint32_t* triangle_indices, float stretch_u_target, float stretch_v_target, float stretch_u_stiffness, float stretch_v_stiffness, float shear_stiffness, float stretch_u_damping, float stretch_v_damping, float shear_damping, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, simulation::VectorView<const float> triangle_u_coefficients, simulation::VectorView<const float> triangle_v_coefficients, const float* triangle_areas, const std::uint32_t* triangle_block_indices, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::VectorView<float> triangle_conditions, simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative);
    void assemble_hinges(::cuda::stream_ref stream, std::uint32_t batch_count, const std::uint32_t* hinge_indices, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* first_opposite, const std::uint32_t* second_opposite, const float* rest_angles, const float* stiffnesses, const float* dampings, const std::uint32_t* hinge_block_indices, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, float* bending_angles, simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative);
    void build_system(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* masses, const float* force_position_derivative, const float* force_velocity_derivative, simulation::VectorView<const float> velocities, simulation::VectorView<const float> forces, float* system, simulation::VectorView<float> right_hand_side);
    void build_constraint_velocity_change(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, const std::uint32_t* fixed_vertex_mask, simulation::VectorView<const float> fixed_positions, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::VectorView<float> constraint_velocity_change);
    void subtract(::cuda::stream_ref stream, std::uint32_t particle_count, simulation::VectorView<const float> first, simulation::VectorView<const float> second, simulation::VectorView<float> result);
    void finalize(::cuda::stream_ref stream, std::uint32_t particle_count, float time_step, simulation::VectorView<const float> positions, simulation::VectorView<const float> velocities, simulation::VectorView<const float> free_velocity_change, simulation::VectorView<const float> constraint_velocity_change, simulation::VectorView<float> velocity_increment, simulation::VectorView<float> next_positions, simulation::VectorView<float> next_velocities);
} // namespace physica::deformables::cloth::solvers::baraff_witkin::kernels

#endif
