#include "choi-ko-kernels.h"
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::choi_ko::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        [[nodiscard]] __device__ Vector3<float> multiply_block(const float* const block, const Vector3<float> vector) {
            return {
                .x = block[0] * vector.x + block[1] * vector.y + block[2] * vector.z,
                .y = block[3] * vector.x + block[4] * vector.y + block[5] * vector.z,
                .z = block[6] * vector.x + block[7] * vector.y + block[8] * vector.z,
            };
        }

        __device__ void sinc_derivatives(const float angle, float& value, float& first_derivative, float& second_derivative) {
            const float squared_angle = angle * angle;
            if (angle < 0.05F) {
                value             = 1.0F - squared_angle / 6.0F + squared_angle * squared_angle / 120.0F - squared_angle * squared_angle * squared_angle / 5040.0F;
                first_derivative  = angle * (-1.0F / 3.0F + squared_angle / 30.0F - squared_angle * squared_angle / 840.0F + squared_angle * squared_angle * squared_angle / 45360.0F);
                second_derivative = -1.0F / 3.0F + squared_angle / 10.0F - squared_angle * squared_angle / 168.0F + squared_angle * squared_angle * squared_angle / 6480.0F;
                return;
            }
            const float sine   = sinf(angle);
            const float cosine = cosf(angle);
            value              = sine / angle;
            first_derivative   = (angle * cosine - sine) / squared_angle;
            second_derivative  = (-squared_angle * sine - 2.0F * angle * cosine + 2.0F * sine) / (squared_angle * angle);
        }

        __device__ void accumulate_membrane_direction(const std::uint32_t triangle, const std::uint32_t direction, const float stiffness, const float damping, const Vector3<float> coefficients, const Vector3<float>* const positions, const Vector3<float>* const velocities, const float area, float* const conditions, const simulation::VectorView<float> local_forces, float* const local_symmetric_force_position_derivatives, float* const local_force_velocity_derivatives) {
            const Vector3<float> derivative = coefficients.x * positions[0] + coefficients.y * positions[1] + coefficients.z * positions[2];
            const float derivative_length   = length(derivative);
            const float extension           = derivative_length - 1.0F;
            conditions[4u * triangle + direction] = extension > 0.0F ? extension : 0.0F;
            if (extension <= 0.0F) return;

            const Vector3<float> normal = derivative / derivative_length;
            const float coefficient_values[]{coefficients.x, coefficients.y, coefficients.z};
            float extension_rate = 0.0F;
            for (std::uint32_t local_vertex = 0u; local_vertex < 3u; ++local_vertex) extension_rate += coefficient_values[local_vertex] * dot(normal, velocities[local_vertex]);
            const float response = -area * (stiffness * extension + damping * extension_rate);
            for (std::uint32_t local_vertex = 0u; local_vertex < 3u; ++local_vertex) {
                const std::uint32_t local_force = 3u * triangle + local_vertex;
                store(local_forces, local_force, load(local_forces, local_force) + response * coefficient_values[local_vertex] * normal);
            }

            for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < 3u; ++local_column) {
                    const std::uint32_t local_block = 9u * triangle + 3u * local_row + local_column;
                    float* const position_block = local_symmetric_force_position_derivatives + 9u * local_block;
                    float* const velocity_block = local_force_velocity_derivatives + 9u * local_block;
                    const float coefficient_product = coefficient_values[local_row] * coefficient_values[local_column];
                    for (std::uint32_t row = 0u; row < 3u; ++row) {
                        for (std::uint32_t column = 0u; column < 3u; ++column) {
                            const std::uint32_t entry = 3u * row + column;
                            const float normal_outer = normal[row] * normal[column];
                            const float norm_hessian = coefficient_product * ((row == column ? 1.0F : 0.0F) - normal_outer) / derivative_length;
                            const float gradient_outer = coefficient_product * normal_outer;
                            position_block[entry] += -area * (stiffness * (gradient_outer + extension * norm_hessian) + damping * extension_rate * norm_hessian);
                            velocity_block[entry] += -area * damping * gradient_outer;
                        }
                    }
                }
            }
        }

        __global__ void assemble_triangles_kernel(const std::uint32_t triangle_count, const float stretch_u_stiffness, const float stretch_v_stiffness, const float diagonal_u_stiffness, const float diagonal_v_stiffness, const float stretch_u_damping, const float stretch_v_damping, const float diagonal_u_damping, const float diagonal_v_damping, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const simulation::VectorView<const float> direction_coefficients, const float* const triangle_areas, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, float* const triangle_conditions, const simulation::VectorView<float> local_forces, float* const local_symmetric_force_position_derivatives, float* const local_force_velocity_derivatives) {
            const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
            if (triangle >= triangle_count) return;
            const std::uint32_t vertices[]{triangle_first[triangle], triangle_second[triangle], triangle_third[triangle]};
            const Vector3<float> local_positions[]{load(positions, vertices[0]), load(positions, vertices[1]), load(positions, vertices[2])};
            const Vector3<float> local_velocities[]{load(velocities, vertices[0]), load(velocities, vertices[1]), load(velocities, vertices[2])};
            for (std::uint32_t local_vertex = 0u; local_vertex < 3u; ++local_vertex) store(local_forces, 3u * triangle + local_vertex, {});
            for (std::uint32_t entry = 0u; entry < 81u; ++entry) {
                local_symmetric_force_position_derivatives[81u * triangle + entry] = 0.0F;
                local_force_velocity_derivatives[81u * triangle + entry] = 0.0F;
            }
            accumulate_membrane_direction(triangle, 0u, stretch_u_stiffness, stretch_u_damping, load(direction_coefficients, 4u * triangle), local_positions, local_velocities, triangle_areas[triangle], triangle_conditions, local_forces, local_symmetric_force_position_derivatives, local_force_velocity_derivatives);
            accumulate_membrane_direction(triangle, 1u, stretch_v_stiffness, stretch_v_damping, load(direction_coefficients, 4u * triangle + 1u), local_positions, local_velocities, triangle_areas[triangle], triangle_conditions, local_forces, local_symmetric_force_position_derivatives, local_force_velocity_derivatives);
            accumulate_membrane_direction(triangle, 2u, diagonal_u_stiffness, diagonal_u_damping, load(direction_coefficients, 4u * triangle + 2u), local_positions, local_velocities, triangle_areas[triangle], triangle_conditions, local_forces, local_symmetric_force_position_derivatives, local_force_velocity_derivatives);
            accumulate_membrane_direction(triangle, 3u, diagonal_v_stiffness, diagonal_v_damping, load(direction_coefficients, 4u * triangle + 3u), local_positions, local_velocities, triangle_areas[triangle], triangle_conditions, local_forces, local_symmetric_force_position_derivatives, local_force_velocity_derivatives);
        }

        __global__ void assemble_hinges_kernel(const std::uint32_t hinge_count, const std::uint32_t local_force_offset, const std::uint32_t local_block_offset, const float imperfection_stiffness, const float bending_damping, const std::uint32_t* const first_opposite, const std::uint32_t* const second_opposite, const float* const rest_spans, const float* const area_sums, const float* const stiffnesses, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, float* const curvatures, float* const curvature_first_derivatives, float* const curvature_second_derivatives, float* const responses, float* const response_derivatives, const simulation::VectorView<float> local_forces, float* const local_symmetric_force_position_derivatives, float* const local_force_velocity_derivatives) {
            const std::uint32_t hinge = blockIdx.x * blockDim.x + threadIdx.x;
            if (hinge >= hinge_count) return;
            store(local_forces, local_force_offset + 2u * hinge, {});
            store(local_forces, local_force_offset + 2u * hinge + 1u, {});
            for (std::uint32_t entry = 0u; entry < 36u; ++entry) {
                local_symmetric_force_position_derivatives[9u * local_block_offset + 36u * hinge + entry] = 0.0F;
                local_force_velocity_derivatives[9u * local_block_offset + 36u * hinge + entry] = 0.0F;
            }
            curvatures[hinge] = 0.0F;
            curvature_first_derivatives[hinge] = 0.0F;
            curvature_second_derivatives[hinge] = 0.0F;
            responses[hinge] = 0.0F;
            response_derivatives[hinge] = 0.0F;

            const Vector3<float> relative_velocity = load(velocities, second_opposite[hinge]) - load(velocities, first_opposite[hinge]);
            const Vector3<float> damping_force = bending_damping * relative_velocity;
            store(local_forces, local_force_offset + 2u * hinge, damping_force);
            store(local_forces, local_force_offset + 2u * hinge + 1u, -damping_force);
            for (std::uint32_t local_row = 0u; local_row < 2u; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < 2u; ++local_column) {
                    const float sign = local_row == local_column ? -1.0F : 1.0F;
                    float* const velocity_block = local_force_velocity_derivatives + 9u * (local_block_offset + 4u * hinge + 2u * local_row + local_column);
                    for (std::uint32_t row = 0u; row < 3u; ++row)
                        for (std::uint32_t column = 0u; column < 3u; ++column) velocity_block[3u * row + column] = row == column ? sign * bending_damping : 0.0F;
                }
            }

            const Vector3<float> difference = load(positions, second_opposite[hinge]) - load(positions, first_opposite[hinge]);
            const float span = length(difference);
            const float rest_span = rest_spans[hinge];
            if (span >= rest_span) return;

            const Vector3<float> normal = difference / span;
            const float normalized_span = span / rest_span;
            const float missing_span = 1.0F - normalized_span;
            float angle = sqrtf(6.0F * missing_span) * (1.0F + missing_span * (0.15F + 0.05732142857142857F * missing_span));
            for (std::uint32_t iteration = 0u; iteration < 10u; ++iteration) {
                float sinc;
                float sinc_first;
                float sinc_second;
                sinc_derivatives(angle, sinc, sinc_first, sinc_second);
                angle -= (sinc - normalized_span) / sinc_first;
            }
            float sinc;
            float sinc_first;
            float sinc_second;
            sinc_derivatives(angle, sinc, sinc_first, sinc_second);
            const float curvature = 2.0F * angle / rest_span;
            const float curvature_first = 2.0F / (rest_span * rest_span * sinc_first);
            const float curvature_second = -2.0F * sinc_second / (rest_span * rest_span * rest_span * sinc_first * sinc_first * sinc_first);
            const float bending_response = area_sums[hinge] * stiffnesses[hinge] * curvature * curvature_first;
            const float bending_response_derivative = area_sums[hinge] * stiffnesses[hinge] * (curvature_first * curvature_first + curvature * curvature_second);
            const float imperfection_response = imperfection_stiffness * (span - rest_span);
            const bool imperfection_branch = bending_response < imperfection_response;
            const float response = imperfection_branch ? imperfection_response : bending_response;
            const float response_derivative = imperfection_branch ? imperfection_stiffness : bending_response_derivative;
            curvatures[hinge] = curvature;
            curvature_first_derivatives[hinge] = curvature_first;
            curvature_second_derivatives[hinge] = curvature_second;
            responses[hinge] = response;
            response_derivatives[hinge] = response_derivative;

            const Vector3<float> conservative_force = response * normal;
            store(local_forces, local_force_offset + 2u * hinge, damping_force + conservative_force);
            store(local_forces, local_force_offset + 2u * hinge + 1u, -damping_force - conservative_force);

            for (std::uint32_t local_row = 0u; local_row < 2u; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < 2u; ++local_column) {
                    const float sign = local_row == local_column ? -1.0F : 1.0F;
                    float* const position_block = local_symmetric_force_position_derivatives + 9u * (local_block_offset + 4u * hinge + 2u * local_row + local_column);
                    for (std::uint32_t row = 0u; row < 3u; ++row)
                        for (std::uint32_t column = 0u; column < 3u; ++column) {
                            const std::uint32_t entry = 3u * row + column;
                            position_block[entry] = sign * response_derivative * normal[row] * normal[column];
                        }
                }
            }
        }

        __global__ void gather_forces_kernel(const std::uint32_t particle_count, const Vector3<float> gravity, const float* const masses, const simulation::VectorView<const float> external_forces, const std::uint32_t* const contribution_offsets, const std::uint32_t* const contribution_indices, const simulation::VectorView<const float> local_forces, const simulation::VectorView<float> forces) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Vector3<float> force = masses[particle] * gravity + load(external_forces, particle);
            for (std::uint32_t contribution = contribution_offsets[particle]; contribution < contribution_offsets[particle + 1u]; ++contribution) force = force + load(local_forces, contribution_indices[contribution]);
            store(forces, particle, force);
        }

        __global__ void gather_blocks_kernel(const std::uint32_t block_count, const std::uint32_t* const contribution_offsets, const std::uint32_t* const contribution_indices, const float* const local_symmetric_force_position_derivatives, const float* const local_force_velocity_derivatives, float* const symmetric_force_position_derivatives, float* const force_velocity_derivatives) {
            const std::uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;
            if (block >= block_count) return;
            for (std::uint32_t entry = 0u; entry < 9u; ++entry) {
                float position_value = 0.0F;
                float velocity_value = 0.0F;
                for (std::uint32_t contribution = contribution_offsets[block]; contribution < contribution_offsets[block + 1u]; ++contribution) {
                    const std::uint32_t local_entry = 9u * contribution_indices[contribution] + entry;
                    position_value += local_symmetric_force_position_derivatives[local_entry];
                    velocity_value += local_force_velocity_derivatives[local_entry];
                }
                symmetric_force_position_derivatives[9u * block + entry] = position_value;
                force_velocity_derivatives[9u * block + entry] = velocity_value;
            }
        }

        __global__ void build_system_and_right_hand_side_kernel(const std::uint32_t particle_count, const float inverse_time_step, const std::uint32_t* const row_offsets, const std::uint32_t* const column_indices, const float* const masses, const float* const symmetric_force_position_derivatives, const float* const force_velocity_derivatives, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> previous_positions, const simulation::VectorView<const float> previous_velocities, const simulation::VectorView<const float> forces, float* const system, const simulation::VectorView<float> right_hand_side) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= particle_count) return;
            const float inverse_squared_time_step = inverse_time_step * inverse_time_step;
            const float mass_coefficient = 2.25F * inverse_squared_time_step;
            const float velocity_derivative_coefficient = 1.5F * inverse_time_step;
            Vector3<float> velocity_derivative_history{};
            for (std::uint32_t block = row_offsets[row]; block < row_offsets[row + 1u]; ++block) {
                const std::uint32_t column = column_indices[block];
                const float* const position_block = symmetric_force_position_derivatives + 9u * block;
                const float* const velocity_block = force_velocity_derivatives + 9u * block;
                float* const system_block = system + 9u * block;
                const Vector3<float> position_history = load(positions, column) - load(previous_positions, column);
                const Vector3<float> velocity_history = 0.5F * inverse_time_step * position_history + load(velocities, column);
                velocity_derivative_history = velocity_derivative_history + multiply_block(velocity_block, velocity_history);
                for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row) {
                    for (std::uint32_t local_column = 0u; local_column < 3u; ++local_column) {
                        const std::uint32_t entry = 3u * local_row + local_column;
                        system_block[entry] = -position_block[entry] - velocity_derivative_coefficient * velocity_block[entry];
                        if (row == column && local_row == local_column) system_block[entry] += mass_coefficient * masses[row];
                    }
                }
            }
            const Vector3<float> position_history = load(positions, row) - load(previous_positions, row);
            const Vector3<float> mass_history = masses[row] * (0.75F * inverse_squared_time_step * position_history + 0.5F * inverse_time_step * (4.0F * load(velocities, row) - load(previous_velocities, row)));
            store(right_hand_side, row, load(forces, row) + mass_history - velocity_derivative_history);
        }

        __global__ void build_prescribed_displacement_kernel(const std::uint32_t particle_count, const std::uint32_t* const fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<float> prescribed_displacement) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(prescribed_displacement, particle, fixed_vertex_mask[particle] != 0u ? load(fixed_positions, particle) - load(positions, particle) : Vector3<float>{});
        }

        __global__ void subtract_kernel(const std::uint32_t particle_count, const simulation::VectorView<const float> first, const simulation::VectorView<const float> second, const simulation::VectorView<float> result) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(result, particle, load(first, particle) - load(second, particle));
        }

        __global__ void finalize_kernel(const std::uint32_t particle_count, const float inverse_time_step, const std::uint32_t* const fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> previous_positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> solution, const simulation::VectorView<const float> prescribed_displacement, const simulation::VectorView<float> bdf2_displacement, const simulation::VectorView<float> next_positions, const simulation::VectorView<float> next_velocities, const simulation::VectorView<float> next_previous_positions, const simulation::VectorView<float> next_previous_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> displacement = load(solution, particle) + load(prescribed_displacement, particle);
            const Vector3<float> position_history = load(positions, particle) - load(previous_positions, particle);
            store(bdf2_displacement, particle, displacement);
            store(next_positions, particle, fixed_vertex_mask[particle] != 0u ? load(fixed_positions, particle) : load(positions, particle) + displacement);
            store(next_velocities, particle, 0.5F * inverse_time_step * (3.0F * displacement - position_history));
            store(next_previous_positions, particle, load(positions, particle));
            store(next_previous_velocities, particle, load(velocities, particle));
        }
    } // namespace

    void assemble_triangles(const ::cuda::stream_ref stream, const std::uint32_t triangle_count, const float stretch_u_stiffness, const float stretch_v_stiffness, const float diagonal_u_stiffness, const float diagonal_v_stiffness, const float stretch_u_damping, const float stretch_v_damping, const float diagonal_u_damping, const float diagonal_v_damping, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const simulation::VectorView<const float> direction_coefficients, const float* const triangle_areas, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, float* const triangle_conditions, const simulation::VectorView<float> local_forces, float* const local_symmetric_force_position_derivatives, float* const local_force_velocity_derivatives) {
        if (triangle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(triangle_count), assemble_triangles_kernel, triangle_count, stretch_u_stiffness, stretch_v_stiffness, diagonal_u_stiffness, diagonal_v_stiffness, stretch_u_damping, stretch_v_damping, diagonal_u_damping, diagonal_v_damping, triangle_first, triangle_second, triangle_third, direction_coefficients, triangle_areas, positions, velocities, triangle_conditions, local_forces, local_symmetric_force_position_derivatives, local_force_velocity_derivatives);
    }

    void assemble_hinges(const ::cuda::stream_ref stream, const std::uint32_t hinge_count, const std::uint32_t local_force_offset, const std::uint32_t local_block_offset, const float imperfection_stiffness, const float bending_damping, const std::uint32_t* const first_opposite, const std::uint32_t* const second_opposite, const float* const rest_spans, const float* const area_sums, const float* const stiffnesses, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, float* const curvatures, float* const curvature_first_derivatives, float* const curvature_second_derivatives, float* const responses, float* const response_derivatives, const simulation::VectorView<float> local_forces, float* const local_symmetric_force_position_derivatives, float* const local_force_velocity_derivatives) {
        if (hinge_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(hinge_count), assemble_hinges_kernel, hinge_count, local_force_offset, local_block_offset, imperfection_stiffness, bending_damping, first_opposite, second_opposite, rest_spans, area_sums, stiffnesses, positions, velocities, curvatures, curvature_first_derivatives, curvature_second_derivatives, responses, response_derivatives, local_forces, local_symmetric_force_position_derivatives, local_force_velocity_derivatives);
    }

    void gather(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t block_count, const Vector3<float> gravity, const float* const masses, const simulation::VectorView<const float> external_forces, const std::uint32_t* const force_contribution_offsets, const std::uint32_t* const force_contribution_indices, const simulation::VectorView<const float> local_forces, const std::uint32_t* const block_contribution_offsets, const std::uint32_t* const block_contribution_indices, const float* const local_symmetric_force_position_derivatives, const float* const local_force_velocity_derivatives, const simulation::VectorView<float> forces, float* const symmetric_force_position_derivatives, float* const force_velocity_derivatives) {
        if (particle_count != 0u) ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), gather_forces_kernel, particle_count, gravity, masses, external_forces, force_contribution_offsets, force_contribution_indices, local_forces, forces);
        if (block_count != 0u) ::cuda::launch(stream, ::cuda::distribute<block_size>(block_count), gather_blocks_kernel, block_count, block_contribution_offsets, block_contribution_indices, local_symmetric_force_position_derivatives, local_force_velocity_derivatives, symmetric_force_position_derivatives, force_velocity_derivatives);
    }

    void build_system_and_right_hand_side(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const std::uint32_t* const row_offsets, const std::uint32_t* const column_indices, const float* const masses, const float* const symmetric_force_position_derivatives, const float* const force_velocity_derivatives, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> previous_positions, const simulation::VectorView<const float> previous_velocities, const simulation::VectorView<const float> forces, float* const system, const simulation::VectorView<float> right_hand_side) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), build_system_and_right_hand_side_kernel, particle_count, 1.0F / time_step, row_offsets, column_indices, masses, symmetric_force_position_derivatives, force_velocity_derivatives, positions, velocities, previous_positions, previous_velocities, forces, system, right_hand_side);
    }

    void build_prescribed_displacement(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* const fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<float> prescribed_displacement) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), build_prescribed_displacement_kernel, particle_count, fixed_vertex_mask, fixed_positions, positions, prescribed_displacement);
    }

    void subtract(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const simulation::VectorView<const float> first, const simulation::VectorView<const float> second, const simulation::VectorView<float> result) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), subtract_kernel, particle_count, first, second, result);
    }

    void finalize(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const std::uint32_t* const fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> previous_positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> solution, const simulation::VectorView<const float> prescribed_displacement, const simulation::VectorView<float> bdf2_displacement, const simulation::VectorView<float> next_positions, const simulation::VectorView<float> next_velocities, const simulation::VectorView<float> next_previous_positions, const simulation::VectorView<float> next_previous_velocities) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), finalize_kernel, particle_count, 1.0F / time_step, fixed_vertex_mask, fixed_positions, positions, previous_positions, velocities, solution, prescribed_displacement, bdf2_displacement, next_positions, next_velocities, next_previous_positions, next_previous_velocities);
    }
} // namespace physica::deformables::cloth::solvers::choi_ko::kernels
