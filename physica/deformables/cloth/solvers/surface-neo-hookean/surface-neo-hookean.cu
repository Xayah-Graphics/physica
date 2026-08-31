#include "surface-neo-hookean-kernels.h"
#include <cfloat>
#include <cuda/launch>
#include <cuda/std/cmath>
#include <cuda/std/limits>

namespace physica::deformables::cloth::solvers::surface_neo_hookean::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

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
            const double metric_00 = first_x * first_x + first_y * first_y + first_z * first_z;
            const double metric_01 = first_x * second_x + first_y * second_y + first_z * second_z;
            const double metric_11 = second_x * second_x + second_y * second_y + second_z * second_z;
            const double determinant = metric_00 * metric_11 - metric_01 * metric_01;
            if (determinant <= 0.0) return ::cuda::std::numeric_limits<double>::infinity();
            const double log_jacobian = 0.5 * ::log(determinant);
            return static_cast<double>(triangle_weights[triangle]) * (0.5 * static_cast<double>(lame_mu) * (metric_00 + metric_11 - 2.0) - static_cast<double>(lame_mu) * log_jacobian + 0.5 * static_cast<double>(lame_lambda) * log_jacobian * log_jacobian);
        }

        __global__ void evaluate_elements_kernel(const std::uint32_t triangle_count, const float lame_lambda, const float lame_mu, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const simulation::VectorView<const float> positions, const simulation::VectorView<float> deformation_gradient_first_columns, const simulation::VectorView<float> deformation_gradient_second_columns, float* surface_jacobians, float* log_surface_jacobians, float* triangle_energies, const simulation::VectorView<float> triangle_gradients, float* triangle_hessians) {
            const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
            if (triangle >= triangle_count) return;
            const std::uint32_t vertices[]{triangle_first[triangle], triangle_second[triangle], triangle_third[triangle]};
            const Vector3<float> local_positions[]{load(positions, vertices[0]), load(positions, vertices[1]), load(positions, vertices[2])};
            const Vector3<float> material_u_gradient = load(material_u_gradients, triangle);
            const Vector3<float> material_v_gradient = load(material_v_gradients, triangle);
            const float material_u[]{material_u_gradient.x, material_u_gradient.y, material_u_gradient.z};
            const float material_v[]{material_v_gradient.x, material_v_gradient.y, material_v_gradient.z};
            const float weight = triangle_weights[triangle];
            const Vector3<float> first_column  = material_u[0] * local_positions[0] + material_u[1] * local_positions[1] + material_u[2] * local_positions[2];
            const Vector3<float> second_column = material_v[0] * local_positions[0] + material_v[1] * local_positions[1] + material_v[2] * local_positions[2];
            const float metric_00 = dot(first_column, first_column);
            const float metric_01 = dot(first_column, second_column);
            const float metric_11 = dot(second_column, second_column);
            const float determinant = metric_00 * metric_11 - metric_01 * metric_01;
            const float jacobian = sqrtf(determinant);
            const float log_jacobian = 0.5F * logf(determinant);
            const float inverse_00 = metric_11 / determinant;
            const float inverse_01 = -metric_01 / determinant;
            const float inverse_11 = metric_00 / determinant;
            const Vector3<float> inverse_columns[]{inverse_00 * first_column + inverse_01 * second_column, inverse_01 * first_column + inverse_11 * second_column};
            const Vector3<float> deformation_columns[]{first_column, second_column};
            const float inverse_metric[]{inverse_00, inverse_01, inverse_01, inverse_11};
            const float coefficient = lame_lambda * log_jacobian - lame_mu;
            const Vector3<float> piola_columns[]{lame_mu * first_column + coefficient * inverse_columns[0], lame_mu * second_column + coefficient * inverse_columns[1]};
            float tangent_projector[9];
            for (std::uint32_t row = 0u; row < 3u; ++row)
                for (std::uint32_t column = 0u; column < 3u; ++column) tangent_projector[3u * row + column] = inverse_columns[0][row] * deformation_columns[0][column] + inverse_columns[1][row] * deformation_columns[1][column];

            store(deformation_gradient_first_columns, triangle, first_column);
            store(deformation_gradient_second_columns, triangle, second_column);
            surface_jacobians[triangle] = jacobian;
            log_surface_jacobians[triangle] = log_jacobian;
            triangle_energies[triangle] = weight * (0.5F * lame_mu * (metric_00 + metric_11 - 2.0F) - lame_mu * log_jacobian + 0.5F * lame_lambda * log_jacobian * log_jacobian);
            for (std::uint32_t local = 0u; local < 3u; ++local) store(triangle_gradients, 3u * triangle + local, weight * (material_u[local] * piola_columns[0] + material_v[local] * piola_columns[1]));

            for (std::uint32_t row_dof = 0u; row_dof < 9u; ++row_dof) {
                const std::uint32_t local_row = row_dof / 3u;
                const std::uint32_t row = row_dof % 3u;
                const float row_material[]{material_u[local_row], material_v[local_row]};
                for (std::uint32_t column_dof = row_dof; column_dof < 9u; ++column_dof) {
                    const std::uint32_t local_column = column_dof / 3u;
                    const std::uint32_t column = column_dof % 3u;
                    const float column_material[]{material_u[local_column], material_v[local_column]};
                    float value{};
                    for (std::uint32_t alpha = 0u; alpha < 2u; ++alpha) {
                        for (std::uint32_t beta = 0u; beta < 2u; ++beta) {
                            const float spatial_identity = row == column ? 1.0F : 0.0F;
                            const float material_identity = alpha == beta ? 1.0F : 0.0F;
                            const float log_hessian = (spatial_identity - tangent_projector[3u * row + column]) * inverse_metric[2u * alpha + beta] - inverse_columns[beta][row] * inverse_columns[alpha][column];
                            const float material_hessian = lame_mu * spatial_identity * material_identity + lame_lambda * inverse_columns[alpha][row] * inverse_columns[beta][column] + coefficient * log_hessian;
                            value += row_material[alpha] * material_hessian * column_material[beta];
                        }
                    }
                    const float weighted = weight * value;
                    triangle_hessians[81u * triangle + 27u * local_row + 9u * local_column + 3u * row + column] = weighted;
                    triangle_hessians[81u * triangle + 27u * local_column + 9u * local_row + 3u * column + row] = weighted;
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

        __global__ void evaluate_domain_steps_kernel(const std::uint32_t triangle_count, const float domain_safety, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, float* triangle_domain_steps) {
            const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
            if (triangle >= triangle_count) return;
            const std::uint32_t vertices[]{triangle_first[triangle], triangle_second[triangle], triangle_third[triangle]};
            const Vector3<float> local_positions[]{load(positions, vertices[0]), load(positions, vertices[1]), load(positions, vertices[2])};
            const Vector3<float> local_direction[]{load(direction, vertices[0]), load(direction, vertices[1]), load(direction, vertices[2])};
            const Vector3<float> material_u_gradient = load(material_u_gradients, triangle);
            const Vector3<float> material_v_gradient = load(material_v_gradients, triangle);
            const double material_u[]{material_u_gradient.x, material_u_gradient.y, material_u_gradient.z};
            const double material_v[]{material_v_gradient.x, material_v_gradient.y, material_v_gradient.z};
            double first[3]{};
            double second[3]{};
            double delta_first[3]{};
            double delta_second[3]{};
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                for (std::uint32_t local = 0u; local < 3u; ++local) {
                    first[component] += material_u[local] * static_cast<double>(local_positions[local][component]);
                    second[component] += material_v[local] * static_cast<double>(local_positions[local][component]);
                    delta_first[component] += material_u[local] * static_cast<double>(local_direction[local][component]);
                    delta_second[component] += material_v[local] * static_cast<double>(local_direction[local][component]);
                }
            }
            const double metric_00 = first[0] * first[0] + first[1] * first[1] + first[2] * first[2];
            const double metric_01 = first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
            const double metric_11 = second[0] * second[0] + second[1] * second[1] + second[2] * second[2];
            const double metric_difference = metric_00 - metric_11;
            const double maximum_squared = 0.5 * (metric_00 + metric_11 + ::sqrt(metric_difference * metric_difference + 4.0 * metric_01 * metric_01));
            const double cross_x = first[1] * second[2] - first[2] * second[1];
            const double cross_y = first[2] * second[0] - first[0] * second[2];
            const double cross_z = first[0] * second[1] - first[1] * second[0];
            const double minimum_singular_value = ::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z) / ::sqrt(maximum_squared);
            const double direction_metric_00 = delta_first[0] * delta_first[0] + delta_first[1] * delta_first[1] + delta_first[2] * delta_first[2];
            const double direction_metric_01 = delta_first[0] * delta_second[0] + delta_first[1] * delta_second[1] + delta_first[2] * delta_second[2];
            const double direction_metric_11 = delta_second[0] * delta_second[0] + delta_second[1] * delta_second[1] + delta_second[2] * delta_second[2];
            const double direction_metric_difference = direction_metric_00 - direction_metric_11;
            const double direction_maximum_squared = 0.5 * (direction_metric_00 + direction_metric_11 + ::sqrt(direction_metric_difference * direction_metric_difference + 4.0 * direction_metric_01 * direction_metric_01));
            triangle_domain_steps[triangle] = direction_maximum_squared == 0.0 ? 1.0F : fminf(1.0F, domain_safety * static_cast<float>(minimum_singular_value / ::sqrt(direction_maximum_squared)));
        }

        __global__ void reduce_domain_step_kernel(const std::uint32_t triangle_count, const float* triangle_domain_steps, float* maximum_domain_step) {
            __shared__ float partial[block_size];
            float value = 1.0F;
            for (std::uint32_t triangle = threadIdx.x; triangle < triangle_count; triangle += blockDim.x) value = fminf(value, triangle_domain_steps[triangle]);
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] = fminf(partial[threadIdx.x], partial[threadIdx.x + offset]);
                __syncthreads();
            }
            if (threadIdx.x == 0u) maximum_domain_step[0] = partial[0];
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

        __global__ void evaluate_candidate_potentials_kernel(const std::uint32_t candidate_count, const std::uint32_t particle_count, const std::uint32_t triangle_count, const float inverse_time_step_squared, const float lame_lambda, const float lame_mu, const float* candidate_contractions, const float* maximum_domain_step, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, double* potentials) {
            __shared__ double partial[block_size];
            const std::uint32_t candidate = blockIdx.x;
            if (candidate >= candidate_count) return;
            const std::uint32_t term_count = particle_count + triangle_count;
            double value = 0.0;
            const float step_size = maximum_domain_step[0] * candidate_contractions[candidate];
            for (std::uint32_t term = threadIdx.x; term < term_count; term += blockDim.x) value += potential_term(term, particle_count, triangle_count, step_size, inverse_time_step_squared, lame_lambda, lame_mu, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, masses, predicted_positions, positions, direction);
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) potentials[candidate] = partial[0];
        }

        __global__ void select_step_size_kernel(const std::uint32_t candidate_count, const float armijo_coefficient, const float* candidate_contractions, const float* maximum_domain_step, const double* current_potential, const double* directional_derivative, const double* candidate_potentials, float* accepted_step_size, std::uint32_t* accepted_candidate, double* accepted_potential) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            for (std::uint32_t candidate = 0u; candidate < candidate_count; ++candidate) {
                const float step_size = maximum_domain_step[0] * candidate_contractions[candidate];
                if (!::cuda::std::isfinite(candidate_potentials[candidate])) continue;
                if (candidate_potentials[candidate] > current_potential[0] + static_cast<double>(armijo_coefficient * step_size) * directional_derivative[0]) continue;
                accepted_step_size[0] = step_size;
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

    void evaluate_elements(const ::cuda::stream_ref stream, const std::uint32_t triangle_count, const float lame_lambda, const float lame_mu, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const simulation::VectorView<const float> positions, const simulation::VectorView<float> deformation_gradient_first_columns, const simulation::VectorView<float> deformation_gradient_second_columns, float* surface_jacobians, float* log_surface_jacobians, float* triangle_energies, const simulation::VectorView<float> triangle_gradients, float* triangle_hessians) {
        if (triangle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(triangle_count), evaluate_elements_kernel, triangle_count, lame_lambda, lame_mu, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, positions, deformation_gradient_first_columns, deformation_gradient_second_columns, surface_jacobians, log_surface_jacobians, triangle_energies, triangle_gradients, triangle_hessians);
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

    void evaluate_domain_steps(const ::cuda::stream_ref stream, const std::uint32_t triangle_count, const float domain_safety, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, float* triangle_domain_steps) {
        if (triangle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(triangle_count), evaluate_domain_steps_kernel, triangle_count, domain_safety, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, positions, direction, triangle_domain_steps);
    }

    void reduce_domain_step(const ::cuda::stream_ref stream, const std::uint32_t triangle_count, const float* triangle_domain_steps, float* maximum_domain_step) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), reduce_domain_step_kernel, triangle_count, triangle_domain_steps, maximum_domain_step);
    }

    void evaluate_potential(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t triangle_count, const float inverse_time_step_squared, const float lame_lambda, const float lame_mu, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, double* potential) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), evaluate_potential_kernel, particle_count, triangle_count, inverse_time_step_squared, lame_lambda, lame_mu, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, masses, predicted_positions, positions, potential);
    }

    void evaluate_candidate_potentials(const ::cuda::stream_ref stream, const std::uint32_t candidate_count, const std::uint32_t particle_count, const std::uint32_t triangle_count, const float inverse_time_step_squared, const float lame_lambda, const float lame_mu, const float* candidate_contractions, const float* maximum_domain_step, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const simulation::VectorView<const float> material_u_gradients, const simulation::VectorView<const float> material_v_gradients, const float* triangle_weights, const float* masses, const simulation::VectorView<const float> predicted_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, double* potentials) {
        if (candidate_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(candidate_count * block_size), evaluate_candidate_potentials_kernel, candidate_count, particle_count, triangle_count, inverse_time_step_squared, lame_lambda, lame_mu, candidate_contractions, maximum_domain_step, triangle_first, triangle_second, triangle_third, material_u_gradients, material_v_gradients, triangle_weights, masses, predicted_positions, positions, direction, potentials);
    }

    void select_step_size(const ::cuda::stream_ref stream, const std::uint32_t candidate_count, const float armijo_coefficient, const float* candidate_contractions, const float* maximum_domain_step, const double* current_potential, const double* directional_derivative, const double* candidate_potentials, float* accepted_step_size, std::uint32_t* accepted_candidate, double* accepted_potential) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(1u), select_step_size_kernel, candidate_count, armijo_coefficient, candidate_contractions, maximum_domain_step, current_potential, directional_derivative, candidate_potentials, accepted_step_size, accepted_candidate, accepted_potential);
    }

    void update_positions(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float* step_size, const simulation::VectorView<const float> direction, const simulation::VectorView<float> positions) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), update_positions_kernel, particle_count, step_size, direction, positions);
    }
} // namespace physica::deformables::cloth::solvers::surface_neo_hookean::kernels
