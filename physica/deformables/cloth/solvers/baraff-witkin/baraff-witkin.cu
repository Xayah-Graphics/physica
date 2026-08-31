#include "baraff-witkin-kernels.h"
#include "second-order.cuh"
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::baraff_witkin::kernels {
    namespace {
        constexpr std::uint32_t block_size = 128u;

        __device__ void accumulate_force(const simulation::VectorView<float> forces, const std::uint32_t particle, const Vector3<float> force) {
            forces.x[particle] += force.x;
            forces.y[particle] += force.y;
            forces.z[particle] += force.z;
        }

        __device__ void assemble_stretch_condition(const std::uint32_t* vertices, const std::uint32_t* block_indices, const Vector3<float> coefficients, const Vector3<float>* positions, const Vector3<float>* velocities, const float area, const float target, const float stiffness, const float damping, float& condition_output, const simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative) {
            const Vector3<float> derivative = coefficients.x * positions[0] + coefficients.y * positions[1] + coefficients.z * positions[2];
            const float derivative_length   = length(derivative);
            const Vector3<float> direction  = derivative / derivative_length;
            const float condition           = area * (derivative_length - target);
            const float local_coefficients[]{coefficients.x, coefficients.y, coefficients.z};
            Vector3<float> gradients[3];
            float condition_rate = 0.0F;
            for (std::uint32_t local_vertex = 0u; local_vertex < 3u; ++local_vertex) {
                gradients[local_vertex] = area * local_coefficients[local_vertex] * direction;
                condition_rate += dot(gradients[local_vertex], velocities[local_vertex]);
            }
            condition_output = condition;

            for (std::uint32_t local_vertex = 0u; local_vertex < 3u; ++local_vertex) accumulate_force(forces, vertices[local_vertex], -stiffness * condition * gradients[local_vertex] - damping * condition_rate * gradients[local_vertex]);
            for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < 3u; ++local_column) {
                    const std::uint32_t block = block_indices[3u * local_row + local_column];
                    for (std::uint32_t row = 0u; row < 3u; ++row) {
                        for (std::uint32_t column = 0u; column < 3u; ++column) {
                            const float hessian = area * local_coefficients[local_row] * local_coefficients[local_column] * ((row == column ? 1.0F : 0.0F) - direction[row] * direction[column]) / derivative_length;
                            force_position_derivative[9u * block + 3u * row + column] += -stiffness * (gradients[local_row][row] * gradients[local_column][column] + condition * hessian) - damping * condition_rate * hessian;
                            force_velocity_derivative[9u * block + 3u * row + column] += -damping * gradients[local_row][row] * gradients[local_column][column];
                        }
                    }
                }
            }
        }

        __device__ void assemble_shear_condition(const std::uint32_t* vertices, const std::uint32_t* block_indices, const Vector3<float> u_coefficients, const Vector3<float> v_coefficients, const Vector3<float>* positions, const Vector3<float>* velocities, const float area, const float stiffness, const float damping, float& condition_output, const simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative) {
            const Vector3<float> u_derivative = u_coefficients.x * positions[0] + u_coefficients.y * positions[1] + u_coefficients.z * positions[2];
            const Vector3<float> v_derivative = v_coefficients.x * positions[0] + v_coefficients.y * positions[1] + v_coefficients.z * positions[2];
            const float condition             = area * dot(u_derivative, v_derivative);
            const float local_u_coefficients[]{u_coefficients.x, u_coefficients.y, u_coefficients.z};
            const float local_v_coefficients[]{v_coefficients.x, v_coefficients.y, v_coefficients.z};
            Vector3<float> gradients[3];
            float condition_rate = 0.0F;
            for (std::uint32_t local_vertex = 0u; local_vertex < 3u; ++local_vertex) {
                gradients[local_vertex] = area * (local_u_coefficients[local_vertex] * v_derivative + local_v_coefficients[local_vertex] * u_derivative);
                condition_rate += dot(gradients[local_vertex], velocities[local_vertex]);
            }
            condition_output = condition;

            for (std::uint32_t local_vertex = 0u; local_vertex < 3u; ++local_vertex) accumulate_force(forces, vertices[local_vertex], -stiffness * condition * gradients[local_vertex] - damping * condition_rate * gradients[local_vertex]);
            for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < 3u; ++local_column) {
                    const std::uint32_t block              = block_indices[3u * local_row + local_column];
                    const float scalar_hessian             = area * (local_u_coefficients[local_row] * local_v_coefficients[local_column] + local_v_coefficients[local_row] * local_u_coefficients[local_column]);
                    for (std::uint32_t row = 0u; row < 3u; ++row) {
                        for (std::uint32_t column = 0u; column < 3u; ++column) {
                            const float hessian = row == column ? scalar_hessian : 0.0F;
                            force_position_derivative[9u * block + 3u * row + column] += -stiffness * (gradients[local_row][row] * gradients[local_column][column] + condition * hessian) - damping * condition_rate * hessian;
                            force_velocity_derivative[9u * block + 3u * row + column] += -damping * gradients[local_row][row] * gradients[local_column][column];
                        }
                    }
                }
            }
        }

        __device__ void assemble_bending_condition(const std::uint32_t* vertices, const std::uint32_t* block_indices, const Vector3<float>* velocities, const SecondOrder<12u>& condition, const float stiffness, const float damping, const simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative) {
            float condition_rate = 0.0F;
            for (std::uint32_t local_vertex = 0u; local_vertex < 4u; ++local_vertex)
                for (std::uint32_t component = 0u; component < 3u; ++component) condition_rate += condition.gradient[3u * local_vertex + component] * velocities[local_vertex][component];

            for (std::uint32_t local_vertex = 0u; local_vertex < 4u; ++local_vertex) {
                const Vector3<float> gradient{
                    .x = condition.gradient[3u * local_vertex],
                    .y = condition.gradient[3u * local_vertex + 1u],
                    .z = condition.gradient[3u * local_vertex + 2u],
                };
                accumulate_force(forces, vertices[local_vertex], -stiffness * condition.value * gradient - damping * condition_rate * gradient);
            }
            for (std::uint32_t local_row = 0u; local_row < 4u; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < 4u; ++local_column) {
                    const std::uint32_t block = block_indices[4u * local_row + local_column];
                    for (std::uint32_t row = 0u; row < 3u; ++row) {
                        for (std::uint32_t column = 0u; column < 3u; ++column) {
                            const float first_gradient  = condition.gradient[3u * local_row + row];
                            const float second_gradient = condition.gradient[3u * local_column + column];
                            const float hessian         = condition.hessian(3u * local_row + row, 3u * local_column + column);
                            force_position_derivative[9u * block + 3u * row + column] += -stiffness * (first_gradient * second_gradient + condition.value * hessian) - damping * condition_rate * hessian;
                            force_velocity_derivative[9u * block + 3u * row + column] += -damping * first_gradient * second_gradient;
                        }
                    }
                }
            }
        }

        __global__ void initialize_forces_kernel(const std::uint32_t particle_count, const Vector3<float> gravity, const float* masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<float> forces) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(forces, particle, masses[particle] * gravity + load(external_forces, particle));
        }

        __global__ void assemble_triangles_kernel(const std::uint32_t batch_count, const std::uint32_t* triangle_indices, const float stretch_u_target, const float stretch_v_target, const float stretch_u_stiffness, const float stretch_v_stiffness, const float shear_stiffness, const float stretch_u_damping, const float stretch_v_damping, const float shear_damping, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> triangle_u_coefficients, const simulation::VectorView<const float> triangle_v_coefficients, const float* triangle_areas, const std::uint32_t* triangle_block_indices, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<float> triangle_conditions, const simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative) {
            const std::uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
            if (element >= batch_count) return;
            const std::uint32_t triangle = triangle_indices[element];
            const std::uint32_t vertices[]{triangle_first[triangle], triangle_second[triangle], triangle_third[triangle]};
            const Vector3<float> local_positions[]{load(positions, vertices[0]), load(positions, vertices[1]), load(positions, vertices[2])};
            const Vector3<float> local_velocities[]{load(velocities, vertices[0]), load(velocities, vertices[1]), load(velocities, vertices[2])};
            float u_condition;
            float v_condition;
            float shear_condition;
            assemble_stretch_condition(vertices, triangle_block_indices + 9u * triangle, load(triangle_u_coefficients, triangle), local_positions, local_velocities, triangle_areas[triangle], stretch_u_target, stretch_u_stiffness, stretch_u_damping, u_condition, forces, force_position_derivative, force_velocity_derivative);
            assemble_stretch_condition(vertices, triangle_block_indices + 9u * triangle, load(triangle_v_coefficients, triangle), local_positions, local_velocities, triangle_areas[triangle], stretch_v_target, stretch_v_stiffness, stretch_v_damping, v_condition, forces, force_position_derivative, force_velocity_derivative);
            assemble_shear_condition(vertices, triangle_block_indices + 9u * triangle, load(triangle_u_coefficients, triangle), load(triangle_v_coefficients, triangle), local_positions, local_velocities, triangle_areas[triangle], shear_stiffness, shear_damping, shear_condition, forces, force_position_derivative, force_velocity_derivative);
            store(triangle_conditions, triangle, {.x = u_condition, .y = v_condition, .z = shear_condition});
        }

        __global__ void assemble_hinges_kernel(const std::uint32_t batch_count, const std::uint32_t* hinge_indices, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* first_opposite, const std::uint32_t* second_opposite, const float* rest_angles, const float* stiffnesses, const float* dampings, const std::uint32_t* hinge_block_indices, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, float* bending_angles, const simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative) {
            const std::uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
            if (element >= batch_count) return;
            const std::uint32_t hinge = hinge_indices[element];
            const std::uint32_t vertices[]{edge_first[hinge], edge_second[hinge], first_opposite[hinge], second_opposite[hinge]};
            const Vector3<float> local_positions[]{load(positions, vertices[0]), load(positions, vertices[1]), load(positions, vertices[2]), load(positions, vertices[3])};
            const Vector3<float> local_velocities[]{load(velocities, vertices[0]), load(velocities, vertices[1]), load(velocities, vertices[2]), load(velocities, vertices[3])};
            SecondVector3<12u> variables[4];
            for (std::uint32_t local_vertex = 0u; local_vertex < 4u; ++local_vertex) {
                variables[local_vertex] = {
                    .x = SecondOrder<12u>::variable(local_positions[local_vertex].x, 3u * local_vertex),
                    .y = SecondOrder<12u>::variable(local_positions[local_vertex].y, 3u * local_vertex + 1u),
                    .z = SecondOrder<12u>::variable(local_positions[local_vertex].z, 3u * local_vertex + 2u),
                };
            }
            const SecondVector3<12u> edge          = normalized(variables[1] - variables[0]);
            const SecondVector3<12u> first_normal  = normalized(cross(variables[1] - variables[0], variables[2] - variables[0]));
            const SecondVector3<12u> second_normal = normalized(cross(variables[0] - variables[1], variables[3] - variables[1]));
            SecondOrder<12u> angle = atan2(dot(cross(first_normal, second_normal), edge), dot(first_normal, second_normal));
            bending_angles[hinge] = angle.value;
            angle.value -= rest_angles[hinge];
            assemble_bending_condition(vertices, hinge_block_indices + 16u * hinge, local_velocities, angle, stiffnesses[hinge], dampings[hinge], forces, force_position_derivative, force_velocity_derivative);
        }

        __global__ void build_system_kernel(const std::uint32_t particle_count, const float time_step, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* masses, const float* force_position_derivative, const float* force_velocity_derivative, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> forces, float* system, const simulation::VectorView<float> right_hand_side) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= particle_count) return;
            Vector3<float> position_derivative_velocity{};
            for (std::uint32_t block = row_offsets[row]; block < row_offsets[row + 1u]; ++block) {
                const std::uint32_t column        = column_indices[block];
                const Vector3<float> velocity     = load(velocities, column);
                const float* const position_block = force_position_derivative + 9u * block;
                const float* const velocity_block = force_velocity_derivative + 9u * block;
                float* const system_block         = system + 9u * block;
                position_derivative_velocity.x += position_block[0] * velocity.x + position_block[1] * velocity.y + position_block[2] * velocity.z;
                position_derivative_velocity.y += position_block[3] * velocity.x + position_block[4] * velocity.y + position_block[5] * velocity.z;
                position_derivative_velocity.z += position_block[6] * velocity.x + position_block[7] * velocity.y + position_block[8] * velocity.z;
                for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row) {
                    for (std::uint32_t local_column = 0u; local_column < 3u; ++local_column) {
                        const std::uint32_t entry = 3u * local_row + local_column;
                        system_block[entry] = -time_step * velocity_block[entry] - time_step * time_step * position_block[entry];
                        if (column == row && local_column == local_row) system_block[entry] += masses[row];
                    }
                }
            }
            store(right_hand_side, row, time_step * (load(forces, row) + time_step * position_derivative_velocity));
        }

        __global__ void build_constraint_velocity_change_kernel(const std::uint32_t particle_count, const float inverse_time_step, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<float> constraint_velocity_change) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> value = fixed_vertex_mask[particle] != 0u ? inverse_time_step * (load(fixed_positions, particle) - load(positions, particle)) - load(velocities, particle) : Vector3<float>{};
            store(constraint_velocity_change, particle, value);
        }

        __global__ void subtract_kernel(const std::uint32_t particle_count, const simulation::VectorView<const float> first, const simulation::VectorView<const float> second, const simulation::VectorView<float> result) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(result, particle, load(first, particle) - load(second, particle));
        }

        __global__ void finalize_kernel(const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> free_velocity_change, const simulation::VectorView<const float> constraint_velocity_change, const simulation::VectorView<float> velocity_increment, const simulation::VectorView<float> next_positions, const simulation::VectorView<float> next_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> increment     = load(free_velocity_change, particle) + load(constraint_velocity_change, particle);
            const Vector3<float> next_velocity = load(velocities, particle) + increment;
            store(velocity_increment, particle, increment);
            store(next_velocities, particle, next_velocity);
            store(next_positions, particle, load(positions, particle) + time_step * next_velocity);
        }
    } // namespace

    void initialize_forces(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const Vector3<float> gravity, const float* masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<float> forces) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), initialize_forces_kernel, particle_count, gravity, masses, external_forces, forces);
    }

    void assemble_triangles(const ::cuda::stream_ref stream, const std::uint32_t batch_count, const std::uint32_t* triangle_indices, const float stretch_u_target, const float stretch_v_target, const float stretch_u_stiffness, const float stretch_v_stiffness, const float shear_stiffness, const float stretch_u_damping, const float stretch_v_damping, const float shear_damping, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> triangle_u_coefficients, const simulation::VectorView<const float> triangle_v_coefficients, const float* triangle_areas, const std::uint32_t* triangle_block_indices, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<float> triangle_conditions, const simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(batch_count), assemble_triangles_kernel, batch_count, triangle_indices, stretch_u_target, stretch_v_target, stretch_u_stiffness, stretch_v_stiffness, shear_stiffness, stretch_u_damping, stretch_v_damping, shear_damping, triangle_first, triangle_second, triangle_third, triangle_u_coefficients, triangle_v_coefficients, triangle_areas, triangle_block_indices, positions, velocities, triangle_conditions, forces, force_position_derivative, force_velocity_derivative);
    }

    void assemble_hinges(const ::cuda::stream_ref stream, const std::uint32_t batch_count, const std::uint32_t* hinge_indices, const std::uint32_t* edge_first, const std::uint32_t* edge_second, const std::uint32_t* first_opposite, const std::uint32_t* second_opposite, const float* rest_angles, const float* stiffnesses, const float* dampings, const std::uint32_t* hinge_block_indices, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, float* bending_angles, const simulation::VectorView<float> forces, float* force_position_derivative, float* force_velocity_derivative) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(batch_count), assemble_hinges_kernel, batch_count, hinge_indices, edge_first, edge_second, first_opposite, second_opposite, rest_angles, stiffnesses, dampings, hinge_block_indices, positions, velocities, bending_angles, forces, force_position_derivative, force_velocity_derivative);
    }

    void build_system(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const float* masses, const float* force_position_derivative, const float* force_velocity_derivative, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> forces, float* system, const simulation::VectorView<float> right_hand_side) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), build_system_kernel, particle_count, time_step, row_offsets, column_indices, masses, force_position_derivative, force_velocity_derivative, velocities, forces, system, right_hand_side);
    }

    void build_constraint_velocity_change(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<float> constraint_velocity_change) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), build_constraint_velocity_change_kernel, particle_count, 1.0F / time_step, fixed_vertex_mask, fixed_positions, positions, velocities, constraint_velocity_change);
    }

    void subtract(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const simulation::VectorView<const float> first, const simulation::VectorView<const float> second, const simulation::VectorView<float> result) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), subtract_kernel, particle_count, first, second, result);
    }

    void finalize(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> free_velocity_change, const simulation::VectorView<const float> constraint_velocity_change, const simulation::VectorView<float> velocity_increment, const simulation::VectorView<float> next_positions, const simulation::VectorView<float> next_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), finalize_kernel, particle_count, time_step, positions, velocities, free_velocity_change, constraint_velocity_change, velocity_increment, next_positions, next_velocities);
    }
} // namespace physica::deformables::cloth::solvers::baraff_witkin::kernels
