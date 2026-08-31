#include "fast-mass-spring-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::fast_mass_spring::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void project_springs_kernel(const std::uint32_t edge_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> positions, const simulation::VectorView<float> projected_springs) {
            const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
            if (edge >= edge_count) return;
            const std::uint32_t first  = edge_first[edge];
            const std::uint32_t second = edge_second[edge];
            if (fixed_vertex_mask[first] != 0u && fixed_vertex_mask[second] != 0u) {
                store(projected_springs, edge, {});
                return;
            }
            const Vector3<float> displacement = load(positions, first) - load(positions, second);
            store(projected_springs, edge, rest_lengths[edge] * displacement / length(displacement));
        }

        __global__ void assemble_right_hand_sides_kernel(const std::uint32_t free_particle_count, const float inverse_time_step_squared, const float spring_stiffness, const std::uint32_t* free_particles, const std::uint32_t* vertex_edge_offsets, const std::uint32_t* vertex_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> projected_springs, float* right_hand_sides) {
            const std::uint32_t free_particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (free_particle >= free_particle_count) return;
            const std::uint32_t particle = free_particles[free_particle];
            Vector3<float> spring_right_hand_side{};
            for (std::uint32_t adjacency = vertex_edge_offsets[particle]; adjacency < vertex_edge_offsets[particle + 1u]; ++adjacency) {
                const std::uint32_t edge   = vertex_edges[adjacency];
                const bool particle_first  = edge_first[edge] == particle;
                const std::uint32_t other  = particle_first ? edge_second[edge] : edge_first[edge];
                spring_right_hand_side     = spring_right_hand_side + (particle_first ? load(projected_springs, edge) : -load(projected_springs, edge));
                if (fixed_vertex_mask[other] != 0u) spring_right_hand_side = spring_right_hand_side + load(fixed_positions, other);
            }
            const Vector3<float> right_hand_side = (masses[particle] * inverse_time_step_squared) * load(predicted_positions, particle) + spring_stiffness * spring_right_hand_side;
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

    void project_springs(const ::cuda::stream_ref stream, const std::uint32_t edge_count, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const float* rest_lengths, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> positions, const simulation::VectorView<float> projected_springs) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(edge_count), project_springs_kernel, edge_count, edge_first, edge_second, rest_lengths, fixed_vertex_mask, positions, projected_springs);
    }

    void assemble_right_hand_sides(const ::cuda::stream_ref stream, const std::uint32_t free_particle_count, const float inverse_time_step_squared, const float spring_stiffness, const std::uint32_t* free_particles, const std::uint32_t* vertex_edge_offsets, const std::uint32_t* vertex_edges, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* fixed_vertex_mask, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> projected_springs, float* right_hand_sides) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(free_particle_count), assemble_right_hand_sides_kernel, free_particle_count, inverse_time_step_squared, spring_stiffness, free_particles, vertex_edge_offsets, vertex_edges, edge_first, edge_second, fixed_vertex_mask, masses, predicted_positions, fixed_positions, projected_springs, right_hand_sides);
    }

    void scatter_solution(const ::cuda::stream_ref stream, const std::uint32_t free_particle_count, const std::uint32_t* free_particles, const float* solutions, const simulation::VectorView<float> positions) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(free_particle_count), scatter_solution_kernel, free_particle_count, free_particles, solutions, positions);
    }
} // namespace physica::deformables::cloth::solvers::fast_mass_spring::kernels
