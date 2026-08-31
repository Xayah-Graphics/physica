#include "stvk-fem-kernels.h"
#include <cfloat>
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::stvk_fem::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        struct ElementKinematics final {
            Vector3<float> first_column;
            Vector3<float> second_column;
            float strain_00;
            float strain_01;
            float strain_11;
        };

        [[nodiscard]] __device__ ElementKinematics evaluate_kinematics(const Vector3<float>* const positions, const Vector3<float> material_u_gradient, const Vector3<float> material_v_gradient) {
            const Vector3<float> first_column  = material_u_gradient.x * positions[0] + material_u_gradient.y * positions[1] + material_u_gradient.z * positions[2];
            const Vector3<float> second_column = material_v_gradient.x * positions[0] + material_v_gradient.y * positions[1] + material_v_gradient.z * positions[2];
            return {
                .first_column  = first_column,
                .second_column = second_column,
                .strain_00     = 0.5F * (dot(first_column, first_column) - 1.0F),
                .strain_01     = 0.5F * dot(first_column, second_column),
                .strain_11     = 0.5F * (dot(second_column, second_column) - 1.0F),
            };
        }

        [[nodiscard]] __device__ float element_energy(const ElementKinematics kinematics, const float lame_lambda, const float lame_mu, const float weight) {
            const float trace = kinematics.strain_00 + kinematics.strain_11;
            return weight * (lame_mu * (kinematics.strain_00 * kinematics.strain_00 + 2.0F * kinematics.strain_01 * kinematics.strain_01 + kinematics.strain_11 * kinematics.strain_11) + 0.5F * lame_lambda * trace * trace);
        }

        [[nodiscard]] __device__ double potential_term(const std::uint32_t term, const std::uint32_t particle_count, const std::uint32_t triangle_count, const float step_size, const float inverse_time_step_squared, const float lame_lambda, const float lame_mu, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* const triangle_weights, const float* const masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction) {
            if (term < particle_count) {
                Vector3<float> position = load(positions, term);
                if (step_size != 0.0F) position = position + step_size * load(direction, term);
                const Vector3<float> difference = position - load(predicted_positions, term);
                return 0.5 * static_cast<double>(masses[term]) * static_cast<double>(inverse_time_step_squared) * (static_cast<double>(difference.x) * difference.x + static_cast<double>(difference.y) * difference.y + static_cast<double>(difference.z) * difference.z);
            }
            const std::uint32_t triangle = term - particle_count;
            if (triangle >= triangle_count) return 0.0;
            const std::uint32_t vertices[]{triangle_first[triangle], triangle_second[triangle], triangle_third[triangle]};
            Vector3<float> local_positions[]{load(positions, vertices[0]), load(positions, vertices[1]), load(positions, vertices[2])};
            if (step_size != 0.0F) local_positions[0] = local_positions[0] + step_size * load(direction, vertices[0]);
            if (step_size != 0.0F) local_positions[1] = local_positions[1] + step_size * load(direction, vertices[1]);
            if (step_size != 0.0F) local_positions[2] = local_positions[2] + step_size * load(direction, vertices[2]);
            const Vector3<float> material_u_gradient = load(material_u_gradients, triangle);
            const Vector3<float> material_v_gradient = load(material_v_gradients, triangle);
            const double first_x = static_cast<double>(material_u_gradient.x) * local_positions[0].x + static_cast<double>(material_u_gradient.y) * local_positions[1].x + static_cast<double>(material_u_gradient.z) * local_positions[2].x;
            const double first_y = static_cast<double>(material_u_gradient.x) * local_positions[0].y + static_cast<double>(material_u_gradient.y) * local_positions[1].y + static_cast<double>(material_u_gradient.z) * local_positions[2].y;
            const double first_z = static_cast<double>(material_u_gradient.x) * local_positions[0].z + static_cast<double>(material_u_gradient.y) * local_positions[1].z + static_cast<double>(material_u_gradient.z) * local_positions[2].z;
            const double second_x = static_cast<double>(material_v_gradient.x) * local_positions[0].x + static_cast<double>(material_v_gradient.y) * local_positions[1].x + static_cast<double>(material_v_gradient.z) * local_positions[2].x;
            const double second_y = static_cast<double>(material_v_gradient.x) * local_positions[0].y + static_cast<double>(material_v_gradient.y) * local_positions[1].y + static_cast<double>(material_v_gradient.z) * local_positions[2].y;
            const double second_z = static_cast<double>(material_v_gradient.x) * local_positions[0].z + static_cast<double>(material_v_gradient.y) * local_positions[1].z + static_cast<double>(material_v_gradient.z) * local_positions[2].z;
            const double strain_00 = 0.5 * (first_x * first_x + first_y * first_y + first_z * first_z - 1.0);
            const double strain_01 = 0.5 * (first_x * second_x + first_y * second_y + first_z * second_z);
            const double strain_11 = 0.5 * (second_x * second_x + second_y * second_y + second_z * second_z - 1.0);
            const double trace = strain_00 + strain_11;
            return static_cast<double>(triangle_weights[triangle]) * (static_cast<double>(lame_mu) * (strain_00 * strain_00 + 2.0 * strain_01 * strain_01 + strain_11 * strain_11) + 0.5 * static_cast<double>(lame_lambda) * trace * trace);
        }

        __global__ void evaluate_elements_kernel(const std::uint32_t triangle_count, const float lame_lambda, const float lame_mu, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const simulation::VectorView<const float> positions, const simulation::VectorView<float> deformation_gradient_first_columns, const simulation::VectorView<float> deformation_gradient_second_columns, const simulation::VectorView<float> green_strains, float* triangle_energies, const simulation::VectorView<float> triangle_gradients, float* triangle_hessians) {
            const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
            if (triangle >= triangle_count) return;
            const std::uint32_t vertices[]{triangle_first[triangle], triangle_second[triangle], triangle_third[triangle]};
            const Vector3<float> local_positions[]{load(positions, vertices[0]), load(positions, vertices[1]), load(positions, vertices[2])};
            const Vector3<float> material_u_gradient = load(material_u_gradients, triangle);
            const Vector3<float> material_v_gradient = load(material_v_gradients, triangle);
            const float material_u[]{material_u_gradient.x, material_u_gradient.y, material_u_gradient.z};
            const float material_v[]{material_v_gradient.x, material_v_gradient.y, material_v_gradient.z};
            const float weight = triangle_weights[triangle];
            const ElementKinematics kinematics = evaluate_kinematics(local_positions, material_u_gradient, material_v_gradient);
            store(deformation_gradient_first_columns, triangle, kinematics.first_column);
            store(deformation_gradient_second_columns, triangle, kinematics.second_column);
            store(green_strains, triangle, {.x = kinematics.strain_00, .y = kinematics.strain_01, .z = kinematics.strain_11});
            triangle_energies[triangle] = element_energy(kinematics, lame_lambda, lame_mu, weight);

            const float trace = kinematics.strain_00 + kinematics.strain_11;
            const float stress_00 = 2.0F * lame_mu * kinematics.strain_00 + lame_lambda * trace;
            const float stress_01 = 2.0F * lame_mu * kinematics.strain_01;
            const float stress_11 = 2.0F * lame_mu * kinematics.strain_11 + lame_lambda * trace;
            const Vector3<float> first_piola_column  = stress_00 * kinematics.first_column + stress_01 * kinematics.second_column;
            const Vector3<float> second_piola_column = stress_01 * kinematics.first_column + stress_11 * kinematics.second_column;
            Vector3<float> spatial_gradients[3];
            for (std::uint32_t local = 0u; local < 3u; ++local) {
                spatial_gradients[local] = material_u[local] * kinematics.first_column + material_v[local] * kinematics.second_column;
                store(triangle_gradients, 3u * triangle + local, weight * (material_u[local] * first_piola_column + material_v[local] * second_piola_column));
            }

            float deformation_product[9];
            for (std::uint32_t row = 0u; row < 3u; ++row)
                for (std::uint32_t column = 0u; column < 3u; ++column) deformation_product[3u * row + column] = kinematics.first_column[row] * kinematics.first_column[column] + kinematics.second_column[row] * kinematics.second_column[column];
            for (std::uint32_t local_row = 0u; local_row < 3u; ++local_row) {
                for (std::uint32_t local_column = local_row; local_column < 3u; ++local_column) {
                    const float stress_contraction = material_u[local_column] * (stress_00 * material_u[local_row] + stress_01 * material_v[local_row]) + material_v[local_column] * (stress_01 * material_u[local_row] + stress_11 * material_v[local_row]);
                    const float material_dot = material_u[local_row] * material_u[local_column] + material_v[local_row] * material_v[local_column];
                    for (std::uint32_t row = 0u; row < 3u; ++row) {
                        for (std::uint32_t column = 0u; column < 3u; ++column) {
                            const float identity = row == column ? stress_contraction : 0.0F;
                            const float value = weight * (identity + lame_mu * (spatial_gradients[local_column][row] * spatial_gradients[local_row][column] + material_dot * deformation_product[3u * row + column]) + lame_lambda * spatial_gradients[local_row][row] * spatial_gradients[local_column][column]);
                            triangle_hessians[81u * triangle + 27u * local_row + 9u * local_column + 3u * row + column] = value;
                            if (local_row != local_column) triangle_hessians[81u * triangle + 27u * local_column + 9u * local_row + 3u * column + row] = value;
                        }
                    }
                }
            }
        }

        __global__ void assemble_incremental_system_kernel(const std::uint32_t particle_count, const float inverse_time_step_squared, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const std::uint32_t* vertex_triangle_offsets, const std::uint32_t* vertex_triangles, const std::uint32_t* matrix_row_offsets, const std::uint32_t* matrix_column_indices, const std::uint32_t* block_contribution_offsets, const std::uint32_t* block_contributions, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> triangle_gradients, const float* triangle_hessians, const simulation::VectorView<float> gradient, float* hessian) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= particle_count) return;
            Vector3<float> value = masses[row] * inverse_time_step_squared * (load(positions, row) - load(predicted_positions, row));
            for (std::uint32_t incidence = vertex_triangle_offsets[row]; incidence < vertex_triangle_offsets[row + 1u]; ++incidence) {
                const std::uint32_t triangle = vertex_triangles[incidence];
                const std::uint32_t local = triangle_first[triangle] == row ? 0u : triangle_second[triangle] == row ? 1u : 2u;
                value = value + load(triangle_gradients, 3u * triangle + local);
            }
            store(gradient, row, value);

            for (std::uint32_t block = matrix_row_offsets[row]; block < matrix_row_offsets[row + 1u]; ++block) {
                float values[9]{};
                for (std::uint32_t contribution = block_contribution_offsets[block]; contribution < block_contribution_offsets[block + 1u]; ++contribution) {
                    const std::uint32_t local_block = block_contributions[contribution];
                    for (std::uint32_t entry = 0u; entry < 9u; ++entry) values[entry] += triangle_hessians[9u * local_block + entry];
                }
                if (matrix_column_indices[block] == row) {
                    const float inertia = masses[row] * inverse_time_step_squared;
                    values[0] += inertia;
                    values[4] += inertia;
                    values[8] += inertia;
                }
                for (std::uint32_t entry = 0u; entry < 9u; ++entry) hessian[9u * block + entry] = values[entry];
            }
        }

        __global__ void compute_gershgorin_bounds_kernel(const std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const std::uint32_t* fixed_vertex_mask, const float* hessian, float* lower_bounds) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            if (fixed_vertex_mask[row] != 0u) {
                lower_bounds[row] = FLT_MAX;
                return;
            }
            float minimum = FLT_MAX;
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                float diagonal = 0.0F;
                float off_diagonal_sum = 0.0F;
                for (std::uint32_t block = row_offsets[row]; block < row_offsets[row + 1u]; ++block) {
                    const std::uint32_t column = column_indices[block];
                    if (fixed_vertex_mask[column] != 0u) continue;
                    for (std::uint32_t column_component = 0u; column_component < 3u; ++column_component) {
                        const float entry = hessian[9u * block + 3u * component + column_component];
                        if (column == row && column_component == component) diagonal = entry;
                        else off_diagonal_sum += fabsf(entry);
                    }
                }
                minimum = fminf(minimum, diagonal - off_diagonal_sum);
            }
            lower_bounds[row] = minimum;
        }

        __global__ void reduce_minimum_kernel(const std::uint32_t count, const float* values, double* result) {
            __shared__ float partial[block_size];
            float value = FLT_MAX;
            for (std::uint32_t index = threadIdx.x; index < count; index += blockDim.x) value = fminf(value, values[index]);
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] = fminf(partial[threadIdx.x], partial[threadIdx.x + offset]);
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = static_cast<double>(partial[0]);
        }

        __global__ void choose_regularization_kernel(const float positive_margin, const double* minimum_bound, float* shift) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            shift[0] = fmaxf(positive_margin - static_cast<float>(minimum_bound[0]), 0.0F);
        }

        __global__ void build_regularized_hessian_kernel(const std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const std::uint32_t* fixed_vertex_mask, const float* shift, const float* hessian, float* regularized_hessian) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            for (std::uint32_t block = row_offsets[row]; block < row_offsets[row + 1u]; ++block) {
                for (std::uint32_t entry = 0u; entry < 9u; ++entry) regularized_hessian[9u * block + entry] = hessian[9u * block + entry];
                if (fixed_vertex_mask[row] == 0u && column_indices[block] == row) {
                    regularized_hessian[9u * block] += shift[0];
                    regularized_hessian[9u * block + 4u] += shift[0];
                    regularized_hessian[9u * block + 8u] += shift[0];
                }
            }
        }

        __global__ void negate_kernel(const std::uint32_t particle_count, const simulation::VectorView<const float> input, const simulation::VectorView<float> output) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(output, particle, -load(input, particle));
        }

        __global__ void evaluate_directional_derivative_kernel(const std::uint32_t particle_count, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> gradient, const simulation::VectorView<const float> direction, double* result) {
            __shared__ double partial[block_size];
            double value = 0.0;
            for (std::uint32_t particle = threadIdx.x; particle < particle_count; particle += blockDim.x) {
                if (fixed_vertex_mask[particle] != 0u) continue;
                const Vector3<float> first  = load(gradient, particle);
                const Vector3<float> second = load(direction, particle);
                value += static_cast<double>(first.x) * second.x + static_cast<double>(first.y) * second.y + static_cast<double>(first.z) * second.z;
            }
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = partial[0];
        }

        __global__ void evaluate_potential_kernel(const std::uint32_t particle_count, const std::uint32_t triangle_count, const float inverse_time_step_squared, const float lame_lambda, const float lame_mu, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, double* potential) {
            __shared__ double partial[block_size];
            const std::uint32_t term_count = particle_count + triangle_count;
            double value = 0.0;
            for (std::uint32_t term = threadIdx.x; term < term_count; term += blockDim.x) value += potential_term(term, particle_count, triangle_count, 0.0F, inverse_time_step_squared, lame_lambda, lame_mu, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, masses, predicted_positions, positions, positions);
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) potential[0] = partial[0];
        }

        __global__ void evaluate_candidate_potentials_kernel(const std::uint32_t candidate_count, const std::uint32_t particle_count, const std::uint32_t triangle_count, const float inverse_time_step_squared, const float lame_lambda, const float lame_mu, const float* candidate_steps, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, double* potentials) {
            __shared__ double partial[block_size];
            const std::uint32_t candidate = blockIdx.x;
            if (candidate >= candidate_count) return;
            const std::uint32_t term_count = particle_count + triangle_count;
            double value = 0.0;
            for (std::uint32_t term = threadIdx.x; term < term_count; term += blockDim.x) value += potential_term(term, particle_count, triangle_count, candidate_steps[candidate], inverse_time_step_squared, lame_lambda, lame_mu, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, masses, predicted_positions, positions, direction);
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) potentials[candidate] = partial[0];
        }

        __global__ void select_step_size_kernel(const std::uint32_t candidate_count, const float armijo_coefficient, const float* candidate_steps, const double* current_potential, const double* directional_derivative, const double* candidate_potentials, float* accepted_step_size, std::uint32_t* accepted_candidate, double* accepted_potential) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            for (std::uint32_t candidate = 0u; candidate < candidate_count; ++candidate) {
                if (candidate_potentials[candidate] > current_potential[0] + static_cast<double>(armijo_coefficient * candidate_steps[candidate]) * directional_derivative[0]) continue;
                accepted_step_size[0] = candidate_steps[candidate];
                accepted_candidate[0] = candidate;
                accepted_potential[0] = candidate_potentials[candidate];
                return;
            }
            accepted_step_size[0] = 0.0F;
            accepted_candidate[0] = candidate_count;
            accepted_potential[0] = current_potential[0];
        }

        __global__ void update_positions_kernel(const std::uint32_t particle_count, const float* step_size, const simulation::VectorView<const float> direction, const simulation::VectorView<float> positions) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(positions, particle, load(positions, particle) + step_size[0] * load(direction, particle));
        }
    } // namespace

    void evaluate_elements(const ::cuda::stream_ref stream, const std::uint32_t triangle_count, const float lame_lambda, const float lame_mu, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const simulation::VectorView<const float> positions, const simulation::VectorView<float> deformation_gradient_first_columns, const simulation::VectorView<float> deformation_gradient_second_columns, const simulation::VectorView<float> green_strains, float* triangle_energies, const simulation::VectorView<float> triangle_gradients, float* triangle_hessians) {
        if (triangle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(triangle_count), evaluate_elements_kernel, triangle_count, lame_lambda, lame_mu, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, positions, deformation_gradient_first_columns, deformation_gradient_second_columns, green_strains, triangle_energies, triangle_gradients, triangle_hessians);
    }

    void assemble_incremental_system(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float inverse_time_step_squared, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const std::uint32_t* vertex_triangle_offsets, const std::uint32_t* vertex_triangles, const std::uint32_t* matrix_row_offsets, const std::uint32_t* matrix_column_indices, const std::uint32_t* block_contribution_offsets, const std::uint32_t* block_contributions, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> triangle_gradients, const float* triangle_hessians, const simulation::VectorView<float> gradient, float* hessian) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), assemble_incremental_system_kernel, particle_count, inverse_time_step_squared, triangle_first, triangle_second, triangle_third, vertex_triangle_offsets, vertex_triangles, matrix_row_offsets, matrix_column_indices, block_contribution_offsets, block_contributions, masses, predicted_positions, positions, triangle_gradients, triangle_hessians, gradient, hessian);
    }

    void compute_gershgorin_bounds(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const std::uint32_t* fixed_vertex_mask, const float* hessian, float* lower_bounds) {
        if (row_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), compute_gershgorin_bounds_kernel, row_count, row_offsets, column_indices, fixed_vertex_mask, hessian, lower_bounds);
    }

    void reduce_minimum(const ::cuda::stream_ref stream, const std::uint32_t count, const float* values, double* result) {
        if (count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), reduce_minimum_kernel, count, values, result);
    }

    void choose_regularization(const ::cuda::stream_ref stream, const float positive_margin, const double* minimum_bound, float* shift) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(1u), choose_regularization_kernel, positive_margin, minimum_bound, shift);
    }

    void build_regularized_hessian(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* row_offsets, const std::uint32_t* column_indices, const std::uint32_t* fixed_vertex_mask, const float* shift, const float* hessian, float* regularized_hessian) {
        if (row_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), build_regularized_hessian_kernel, row_count, row_offsets, column_indices, fixed_vertex_mask, shift, hessian, regularized_hessian);
    }

    void negate(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const simulation::VectorView<const float> input, const simulation::VectorView<float> output) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), negate_kernel, particle_count, input, output);
    }

    void evaluate_directional_derivative(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* fixed_vertex_mask, const simulation::VectorView<const float> gradient, const simulation::VectorView<const float> direction, double* result) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), evaluate_directional_derivative_kernel, particle_count, fixed_vertex_mask, gradient, direction, result);
    }

    void evaluate_potential(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t triangle_count, const float inverse_time_step_squared, const float lame_lambda, const float lame_mu, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, double* potential) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), evaluate_potential_kernel, particle_count, triangle_count, inverse_time_step_squared, lame_lambda, lame_mu, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, masses, predicted_positions, positions, potential);
    }

    void evaluate_candidate_potentials(const ::cuda::stream_ref stream, const std::uint32_t candidate_count, const std::uint32_t particle_count, const std::uint32_t triangle_count, const float inverse_time_step_squared, const float lame_lambda, const float lame_mu, const float* candidate_steps, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, double* potentials) {
        if (candidate_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(candidate_count * block_size), evaluate_candidate_potentials_kernel, candidate_count, particle_count, triangle_count, inverse_time_step_squared, lame_lambda, lame_mu, candidate_steps, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, masses, predicted_positions, positions, direction, potentials);
    }

    void select_step_size(const ::cuda::stream_ref stream, const std::uint32_t candidate_count, const float armijo_coefficient, const float* candidate_steps, const double* current_potential, const double* directional_derivative, const double* candidate_potentials, float* accepted_step_size, std::uint32_t* accepted_candidate, double* accepted_potential) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(1u), select_step_size_kernel, candidate_count, armijo_coefficient, candidate_steps, current_potential, directional_derivative, candidate_potentials, accepted_step_size, accepted_candidate, accepted_potential);
    }

    void update_positions(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float* step_size, const simulation::VectorView<const float> direction, const simulation::VectorView<float> positions) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), update_positions_kernel, particle_count, step_size, direction, positions);
    }
} // namespace physica::deformables::cloth::solvers::stvk_fem::kernels
