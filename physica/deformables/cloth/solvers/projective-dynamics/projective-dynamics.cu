#include "projective-dynamics-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::projective_dynamics::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __device__ void thin_polar(const Vector3<float> deformation_gradient_first_column, const Vector3<float> deformation_gradient_second_column, Vector3<float>& projected_frame_first_column, Vector3<float>& projected_frame_second_column) {
            const Vector3<float> oriented_area  = cross(deformation_gradient_first_column, deformation_gradient_second_column);
            const float square_root_determinant = length(oriented_area);
            const Vector3<float> normal         = oriented_area / square_root_determinant;
            const float trace_factor            = sqrtf(dot(deformation_gradient_first_column, deformation_gradient_first_column) + dot(deformation_gradient_second_column, deformation_gradient_second_column) + 2.0F * square_root_determinant);
            projected_frame_first_column        = (deformation_gradient_first_column - cross(normal, deformation_gradient_second_column)) / trace_factor;
            projected_frame_second_column       = (deformation_gradient_second_column + cross(normal, deformation_gradient_first_column)) / trace_factor;
        }

        __global__ void project_deformation_gradients_kernel(const std::uint32_t deformation_gradient_count, const simulation::VectorView<const float> deformation_gradient_first_columns, const simulation::VectorView<const float> deformation_gradient_second_columns, const simulation::VectorView<float> projected_frame_first_columns, const simulation::VectorView<float> projected_frame_second_columns) {
            const std::uint32_t deformation_gradient = blockIdx.x * blockDim.x + threadIdx.x;
            if (deformation_gradient >= deformation_gradient_count) return;
            Vector3<float> projected_frame_first_column;
            Vector3<float> projected_frame_second_column;
            thin_polar(load(deformation_gradient_first_columns, deformation_gradient), load(deformation_gradient_second_columns, deformation_gradient), projected_frame_first_column, projected_frame_second_column);
            store(projected_frame_first_columns, deformation_gradient, projected_frame_first_column);
            store(projected_frame_second_columns, deformation_gradient, projected_frame_second_column);
        }

        __global__ void project_membranes_kernel(const std::uint32_t triangle_count, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const float* material_inverse_00, const float* material_inverse_01, const float* material_inverse_10, const float* material_inverse_11, const simulation::VectorView<const float> positions, const simulation::VectorView<float> projected_frame_first_columns, const simulation::VectorView<float> projected_frame_second_columns) {
            const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
            if (triangle >= triangle_count) return;
            const Vector3<float> first_edge  = load(positions, triangle_second[triangle]) - load(positions, triangle_first[triangle]);
            const Vector3<float> second_edge = load(positions, triangle_third[triangle]) - load(positions, triangle_first[triangle]);
            const Vector3<float> deformation_gradient_first_column  = material_inverse_00[triangle] * first_edge + material_inverse_10[triangle] * second_edge;
            const Vector3<float> deformation_gradient_second_column = material_inverse_01[triangle] * first_edge + material_inverse_11[triangle] * second_edge;
            Vector3<float> projected_frame_first_column;
            Vector3<float> projected_frame_second_column;
            thin_polar(deformation_gradient_first_column, deformation_gradient_second_column, projected_frame_first_column, projected_frame_second_column);
            store(projected_frame_first_columns, triangle, projected_frame_first_column);
            store(projected_frame_second_columns, triangle, projected_frame_second_column);
        }

        __global__ void project_bending_kernel(const std::uint32_t hinge_count, const std::uint32_t* hinge_first_opposite, const std::uint32_t* hinge_second_opposite, const simulation::VectorView<const float> positions, const simulation::VectorView<float> bending_directions) {
            const std::uint32_t hinge = blockIdx.x * blockDim.x + threadIdx.x;
            if (hinge >= hinge_count) return;
            const Vector3<float> opposite_displacement = load(positions, hinge_second_opposite[hinge]) - load(positions, hinge_first_opposite[hinge]);
            store(bending_directions, hinge, opposite_displacement / length(opposite_displacement));
        }

        __global__ void assemble_right_hand_sides_kernel(const std::uint32_t free_particle_count, const float inverse_time_step_squared, const float bending_stiffness, const std::uint32_t* free_particles, const std::uint32_t* vertex_triangle_offsets, const std::uint32_t* vertex_triangles, const std::uint32_t* vertex_hinge_offsets, const std::uint32_t* vertex_hinges, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const std::uint32_t* hinge_first_opposite, const std::uint32_t* hinge_second_opposite, const float* material_inverse_00, const float* material_inverse_01, const float* material_inverse_10, const float* material_inverse_11, const float* membrane_weights, const float* bending_rest_lengths, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> fixed_right_hand_sides, const simulation::VectorView<const float> projected_frame_first_columns, const simulation::VectorView<const float> projected_frame_second_columns, const simulation::VectorView<const float> bending_directions, float* right_hand_sides) {
            const std::uint32_t free_particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (free_particle >= free_particle_count) return;
            const std::uint32_t particle = free_particles[free_particle];
            Vector3<float> local_right_hand_side{};
            for (std::uint32_t adjacency = vertex_triangle_offsets[particle]; adjacency < vertex_triangle_offsets[particle + 1u]; ++adjacency) {
                const std::uint32_t triangle = vertex_triangles[adjacency];
                float gradient_u;
                float gradient_v;
                if (particle == triangle_first[triangle]) {
                    gradient_u = -material_inverse_00[triangle] - material_inverse_10[triangle];
                    gradient_v = -material_inverse_01[triangle] - material_inverse_11[triangle];
                } else if (particle == triangle_second[triangle]) {
                    gradient_u = material_inverse_00[triangle];
                    gradient_v = material_inverse_01[triangle];
                } else {
                    gradient_u = material_inverse_10[triangle];
                    gradient_v = material_inverse_11[triangle];
                }
                local_right_hand_side = local_right_hand_side + membrane_weights[triangle] * (gradient_u * load(projected_frame_first_columns, triangle) + gradient_v * load(projected_frame_second_columns, triangle));
            }
            for (std::uint32_t adjacency = vertex_hinge_offsets[particle]; adjacency < vertex_hinge_offsets[particle + 1u]; ++adjacency) {
                const std::uint32_t hinge = vertex_hinges[adjacency];
                if (particle != hinge_first_opposite[hinge] && particle != hinge_second_opposite[hinge]) continue;
                const Vector3<float> projected_displacement = bending_rest_lengths[hinge] * load(bending_directions, hinge);
                if (particle == hinge_first_opposite[hinge]) local_right_hand_side = local_right_hand_side - bending_stiffness * projected_displacement;
                if (particle == hinge_second_opposite[hinge]) local_right_hand_side = local_right_hand_side + bending_stiffness * projected_displacement;
            }
            const Vector3<float> right_hand_side = (masses[particle] * inverse_time_step_squared) * load(predicted_positions, particle) + load(fixed_right_hand_sides, free_particle) + local_right_hand_side;
            right_hand_sides[free_particle]                            = right_hand_side.x;
            right_hand_sides[free_particle_count + free_particle]      = right_hand_side.y;
            right_hand_sides[2u * free_particle_count + free_particle] = right_hand_side.z;
        }

        __global__ void scatter_solution_kernel(const std::uint32_t free_particle_count, const std::uint32_t* free_particles, const float* solutions, const simulation::VectorView<float> positions) {
            const std::uint32_t free_particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (free_particle >= free_particle_count) return;
            const std::uint32_t particle = free_particles[free_particle];
            store(positions, particle, {.x = solutions[free_particle], .y = solutions[free_particle_count + free_particle], .z = solutions[2u * free_particle_count + free_particle]});
        }
    } // namespace

    void project_deformation_gradients(const ::cuda::stream_ref stream, const std::uint32_t deformation_gradient_count, const simulation::VectorView<const float> deformation_gradient_first_columns, const simulation::VectorView<const float> deformation_gradient_second_columns, const simulation::VectorView<float> projected_frame_first_columns, const simulation::VectorView<float> projected_frame_second_columns) {
        if (deformation_gradient_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(deformation_gradient_count), project_deformation_gradients_kernel, deformation_gradient_count, deformation_gradient_first_columns, deformation_gradient_second_columns, projected_frame_first_columns, projected_frame_second_columns);
    }

    void project_membranes(const ::cuda::stream_ref stream, const std::uint32_t triangle_count, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const float* material_inverse_00, const float* material_inverse_01, const float* material_inverse_10, const float* material_inverse_11, const simulation::VectorView<const float> positions, const simulation::VectorView<float> projected_frame_first_columns, const simulation::VectorView<float> projected_frame_second_columns) {
        if (triangle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(triangle_count), project_membranes_kernel, triangle_count, triangle_first, triangle_second, triangle_third, material_inverse_00, material_inverse_01, material_inverse_10, material_inverse_11, positions, projected_frame_first_columns, projected_frame_second_columns);
    }

    void project_bending(const ::cuda::stream_ref stream, const std::uint32_t hinge_count, const std::uint32_t* hinge_first_opposite, const std::uint32_t* hinge_second_opposite, const simulation::VectorView<const float> positions, const simulation::VectorView<float> bending_directions) {
        if (hinge_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(hinge_count), project_bending_kernel, hinge_count, hinge_first_opposite, hinge_second_opposite, positions, bending_directions);
    }

    void assemble_right_hand_sides(const ::cuda::stream_ref stream, const std::uint32_t free_particle_count, const float inverse_time_step_squared, const float bending_stiffness, const std::uint32_t* free_particles, const std::uint32_t* vertex_triangle_offsets, const std::uint32_t* vertex_triangles, const std::uint32_t* vertex_hinge_offsets, const std::uint32_t* vertex_hinges, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const std::uint32_t* hinge_first_opposite, const std::uint32_t* hinge_second_opposite, const float* material_inverse_00, const float* material_inverse_01, const float* material_inverse_10, const float* material_inverse_11, const float* membrane_weights, const float* bending_rest_lengths, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> fixed_right_hand_sides, const simulation::VectorView<const float> projected_frame_first_columns, const simulation::VectorView<const float> projected_frame_second_columns, const simulation::VectorView<const float> bending_directions, float* right_hand_sides) {
        if (free_particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(free_particle_count), assemble_right_hand_sides_kernel, free_particle_count, inverse_time_step_squared, bending_stiffness, free_particles, vertex_triangle_offsets, vertex_triangles, vertex_hinge_offsets, vertex_hinges, triangle_first, triangle_second, triangle_third, hinge_first_opposite, hinge_second_opposite, material_inverse_00, material_inverse_01, material_inverse_10, material_inverse_11, membrane_weights, bending_rest_lengths, masses, predicted_positions, fixed_right_hand_sides, projected_frame_first_columns, projected_frame_second_columns, bending_directions, right_hand_sides);
    }

    void scatter_solution(const ::cuda::stream_ref stream, const std::uint32_t free_particle_count, const std::uint32_t* free_particles, const float* solutions, const simulation::VectorView<float> positions) {
        if (free_particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(free_particle_count), scatter_solution_kernel, free_particle_count, free_particles, solutions, positions);
    }
} // namespace physica::deformables::cloth::solvers::projective_dynamics::kernels
