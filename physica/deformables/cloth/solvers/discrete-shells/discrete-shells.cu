#include "discrete-shells-kernels.h"
#include "second-order.cuh"
#include <cfloat>
#include <cuda/launch>

namespace physica::deformables::cloth::solvers::discrete_shells::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        template <std::uint32_t Dimension>
        [[nodiscard]] __device__ SecondVector3<Dimension> variable(const Vector3<float> value, const std::uint32_t local_vertex) {
            return {
                .x = SecondOrder<Dimension>::variable(value.x, 3u * local_vertex),
                .y = SecondOrder<Dimension>::variable(value.y, 3u * local_vertex + 1u),
                .z = SecondOrder<Dimension>::variable(value.z, 3u * local_vertex + 2u),
            };
        }

        template <std::uint32_t Dimension, std::uint32_t VertexCount>
        __device__ void store_derivatives(const SecondOrder<Dimension>& quantity, const std::uint32_t element, const simulation::VectorView<float> gradients, float* const hessians) {
            for (std::uint32_t local = 0u; local < VertexCount; ++local) store(gradients, VertexCount * element + local, {.x = quantity.gradient[3u * local], .y = quantity.gradient[3u * local + 1u], .z = quantity.gradient[3u * local + 2u]});
            for (std::uint32_t local_row = 0u; local_row < VertexCount; ++local_row) {
                for (std::uint32_t local_column = 0u; local_column < VertexCount; ++local_column) {
                    for (std::uint32_t row = 0u; row < 3u; ++row)
                        for (std::uint32_t column = 0u; column < 3u; ++column) hessians[9u * (VertexCount * VertexCount * element + VertexCount * local_row + local_column) + 3u * row + column] = quantity.hessian(3u * local_row + row, 3u * local_column + column);
                }
            }
        }

        [[nodiscard]] __device__ SecondOrder<12u> dihedral(const SecondVector3<12u>* const vertices) {
            const SecondVector3<12u> edge          = discrete_shells::normalized(vertices[1] - vertices[0]);
            const SecondVector3<12u> first_normal  = discrete_shells::normalized(discrete_shells::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
            const SecondVector3<12u> second_normal = discrete_shells::normalized(discrete_shells::cross(vertices[0] - vertices[1], vertices[3] - vertices[1]));
            return discrete_shells::atan2(discrete_shells::dot(discrete_shells::cross(first_normal, second_normal), edge), discrete_shells::dot(first_normal, second_normal));
        }

        [[nodiscard]] __device__ float dihedral(const Vector3<float>* const vertices) {
            const Vector3<float> edge          = ::physica::normalized(vertices[1] - vertices[0]);
            const Vector3<float> first_normal  = ::physica::normalized(::physica::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
            const Vector3<float> second_normal = ::physica::normalized(::physica::cross(vertices[0] - vertices[1], vertices[3] - vertices[1]));
            return atan2f(::physica::dot(::physica::cross(first_normal, second_normal), edge), ::physica::dot(first_normal, second_normal));
        }

        [[nodiscard]] __device__ double dihedral(const Vector3<double>* const vertices) {
            const Vector3<double> edge          = ::physica::normalized(vertices[1] - vertices[0]);
            const Vector3<double> first_normal  = ::physica::normalized(::physica::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
            const Vector3<double> second_normal = ::physica::normalized(::physica::cross(vertices[0] - vertices[1], vertices[3] - vertices[1]));
            return ::atan2(::physica::dot(::physica::cross(first_normal, second_normal), edge), ::physica::dot(first_normal, second_normal));
        }

        [[nodiscard]] __device__ double angle_delta(const double angle, const double reference) {
            double result = angle - reference;
            if (result > 3.14159265358979323846) result -= 6.28318530717958647692;
            if (result <= -3.14159265358979323846) result += 6.28318530717958647692;
            return result;
        }

        template <std::uint32_t Dimension>
        [[nodiscard]] __device__ SecondOrder<Dimension> angle_delta(const SecondOrder<Dimension>& angle, const float reference) {
            SecondOrder<Dimension> result = angle - SecondOrder<Dimension>::constant(reference);
            if (result.value > 3.14159265358979323846F) result.value -= 6.28318530717958647692F;
            if (result.value <= -3.14159265358979323846F) result.value += 6.28318530717958647692F;
            return result;
        }

        [[nodiscard]] __device__ Vector3<double> load_double(const simulation::VectorView<const float> field, const std::uint32_t index) {
            const Vector3<float> value = load(field, index);
            return {.x = value.x, .y = value.y, .z = value.z};
        }

        [[nodiscard]] __device__ Vector3<double> candidate_position(const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, const std::uint32_t index, const float step) {
            const Vector3<double> position = load_double(positions, index);
            if (step == 0.0F) return position;
            return position + static_cast<double>(step) * load_double(direction, index);
        }

        [[nodiscard]] __device__ std::uint32_t hinge_local_vertex(const std::uint32_t particle, const std::uint32_t hinge, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const first_opposite, const std::uint32_t* const second_opposite) {
            if (edge_first[hinge] == particle) return 0u;
            if (edge_second[hinge] == particle) return 1u;
            if (first_opposite[hinge] == particle) return 2u;
            return 3u;
        }

        [[nodiscard]] __device__ const float* energy_hessian_block(const std::uint32_t contribution, const std::uint32_t edge_count, const std::uint32_t triangle_count, const float* const edge_hessians, const float* const triangle_hessians, const float* const hinge_hessians) {
            const std::uint32_t edge_block_count = 4u * edge_count;
            const std::uint32_t triangle_block_count = 9u * triangle_count;
            if (contribution < edge_block_count) return edge_hessians + 9u * contribution;
            if (contribution < edge_block_count + triangle_block_count) return triangle_hessians + 9u * (contribution - edge_block_count);
            return hinge_hessians + 9u * (contribution - edge_block_count - triangle_block_count);
        }

        [[nodiscard]] __device__ double potential_term(const std::uint32_t term, const std::uint32_t particle_count, const std::uint32_t edge_count, const std::uint32_t triangle_count, const std::uint32_t hinge_count, const float step, const float time_step, const float length_stiffness, const float area_stiffness, const float bending_stiffness, const float bending_damping, const Vector3<float> gravity, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const std::uint32_t* const hinge_edge_first, const std::uint32_t* const hinge_edge_second, const std::uint32_t* const hinge_first_opposite, const std::uint32_t* const hinge_second_opposite, const float* const edge_rest_lengths, const float* const triangle_rest_areas, const float* const hinge_rest_angles, const float* const previous_hinge_angles, const float* const hinge_weights, const float* const masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction) {
            const double mass_coefficient = 4.0 / (static_cast<double>(time_step) * time_step);
            if (term < particle_count) {
                const Vector3<double> position = candidate_position(positions, direction, term, step);
                const Vector3<double> difference = position - load_double(position_predictor, term);
                const Vector3<float> external = load(external_forces, term);
                const Vector3<double> force{.x = static_cast<double>(masses[term]) * gravity.x + external.x, .y = static_cast<double>(masses[term]) * gravity.y + external.y, .z = static_cast<double>(masses[term]) * gravity.z + external.z};
                return 0.5 * static_cast<double>(masses[term]) * mass_coefficient * ::physica::dot(difference, difference) - ::physica::dot(force, position);
            }

            std::uint32_t local_term = term - particle_count;
            if (local_term < edge_count) {
                const Vector3<double> difference = candidate_position(positions, direction, edge_second[local_term], step) - candidate_position(positions, direction, edge_first[local_term], step);
                const double condition = 1.0 - ::physica::length(difference) / edge_rest_lengths[local_term];
                return static_cast<double>(length_stiffness) * edge_rest_lengths[local_term] * condition * condition;
            }

            local_term -= edge_count;
            if (local_term < triangle_count) {
                const Vector3<double> first = candidate_position(positions, direction, triangle_first[local_term], step);
                const Vector3<double> second = candidate_position(positions, direction, triangle_second[local_term], step);
                const Vector3<double> third = candidate_position(positions, direction, triangle_third[local_term], step);
                const double area = 0.5 * ::physica::length(::physica::cross(second - first, third - first));
                const double condition = 1.0 - area / triangle_rest_areas[local_term];
                return static_cast<double>(area_stiffness) * triangle_rest_areas[local_term] * condition * condition;
            }

            local_term -= triangle_count;
            if (local_term >= hinge_count) return 0.0;
            const Vector3<double> vertices[]{
                candidate_position(positions, direction, hinge_edge_first[local_term], step),
                candidate_position(positions, direction, hinge_edge_second[local_term], step),
                candidate_position(positions, direction, hinge_first_opposite[local_term], step),
                candidate_position(positions, direction, hinge_second_opposite[local_term], step),
            };
            const double angle = dihedral(vertices);
            const double elastic_delta = angle_delta(angle, hinge_rest_angles[local_term]);
            const double damping_delta = angle_delta(angle, previous_hinge_angles[local_term]);
            return static_cast<double>(bending_stiffness) * hinge_weights[local_term] * elastic_delta * elastic_delta + 0.5 * static_cast<double>(bending_damping) * hinge_weights[local_term] / time_step * damping_delta * damping_delta;
        }

        __global__ void prepare_newmark_kernel(const std::uint32_t particle_count, const float time_step, const std::uint32_t* const fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> accelerations, const simulation::VectorView<float> position_predictor, const simulation::VectorView<float> velocity_predictor, const simulation::VectorView<float> candidate_positions) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            if (fixed_vertex_mask[particle] != 0u) {
                store(position_predictor, particle, load(fixed_positions, particle));
                store(velocity_predictor, particle, {});
                store(candidate_positions, particle, load(fixed_positions, particle));
                return;
            }
            const Vector3<float> acceleration = load(accelerations, particle);
            const Vector3<float> predicted_position = load(positions, particle) + time_step * load(velocities, particle) + 0.25F * time_step * time_step * acceleration;
            store(position_predictor, particle, predicted_position);
            store(velocity_predictor, particle, load(velocities, particle) + 0.5F * time_step * acceleration);
            store(candidate_positions, particle, predicted_position);
        }

        __global__ void evaluate_hinge_angles_kernel(const std::uint32_t hinge_count, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const first_opposite, const std::uint32_t* const second_opposite, const simulation::VectorView<const float> positions, float* const angles) {
            const std::uint32_t hinge = blockIdx.x * blockDim.x + threadIdx.x;
            if (hinge >= hinge_count) return;
            const Vector3<float> vertices[]{load(positions, edge_first[hinge]), load(positions, edge_second[hinge]), load(positions, first_opposite[hinge]), load(positions, second_opposite[hinge])};
            angles[hinge] = dihedral(vertices);
        }

        __global__ void evaluate_edges_kernel(const std::uint32_t edge_count, const float length_stiffness, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const float* const rest_lengths, const simulation::VectorView<const float> positions, float* const conditions, float* const energies, const simulation::VectorView<float> gradients, float* const hessians) {
            const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
            if (edge >= edge_count) return;
            const SecondVector3<6u> first = variable<6u>(load(positions, edge_first[edge]), 0u);
            const SecondVector3<6u> second = variable<6u>(load(positions, edge_second[edge]), 1u);
            const SecondOrder<6u> condition = SecondOrder<6u>::constant(1.0F) - discrete_shells::length(second - first) / SecondOrder<6u>::constant(rest_lengths[edge]);
            const SecondOrder<6u> energy = SecondOrder<6u>::constant(length_stiffness * rest_lengths[edge]) * condition * condition;
            conditions[edge] = condition.value;
            energies[edge] = energy.value;
            store_derivatives<6u, 2u>(energy, edge, gradients, hessians);
        }

        __global__ void evaluate_triangles_kernel(const std::uint32_t triangle_count, const float area_stiffness, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const float* const rest_areas, const simulation::VectorView<const float> positions, float* const conditions, float* const energies, const simulation::VectorView<float> gradients, float* const hessians) {
            const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
            if (triangle >= triangle_count) return;
            const SecondVector3<9u> first = variable<9u>(load(positions, triangle_first[triangle]), 0u);
            const SecondVector3<9u> second = variable<9u>(load(positions, triangle_second[triangle]), 1u);
            const SecondVector3<9u> third = variable<9u>(load(positions, triangle_third[triangle]), 2u);
            const SecondOrder<9u> area = SecondOrder<9u>::constant(0.5F) * discrete_shells::length(discrete_shells::cross(second - first, third - first));
            const SecondOrder<9u> condition = SecondOrder<9u>::constant(1.0F) - area / SecondOrder<9u>::constant(rest_areas[triangle]);
            const SecondOrder<9u> energy = SecondOrder<9u>::constant(area_stiffness * rest_areas[triangle]) * condition * condition;
            conditions[triangle] = condition.value;
            energies[triangle] = energy.value;
            store_derivatives<9u, 3u>(energy, triangle, gradients, hessians);
        }

        __global__ void evaluate_hinges_kernel(const std::uint32_t hinge_count, const float time_step, const float bending_stiffness, const float bending_damping, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const first_opposite, const std::uint32_t* const second_opposite, const float* const rest_angles, const float* const previous_angles, const float* const weights, const simulation::VectorView<const float> positions, float* const angles, float* const angle_deltas, float* const angle_rates, float* const energies, float* const damping_potentials, const simulation::VectorView<float> angle_gradients, float* const angle_hessians, const simulation::VectorView<float> energy_gradients, float* const energy_hessians, const simulation::VectorView<float> damping_residuals, float* const damping_jacobians) {
            const std::uint32_t hinge = blockIdx.x * blockDim.x + threadIdx.x;
            if (hinge >= hinge_count) return;
            const SecondVector3<12u> vertices[]{
                variable<12u>(load(positions, edge_first[hinge]), 0u),
                variable<12u>(load(positions, edge_second[hinge]), 1u),
                variable<12u>(load(positions, first_opposite[hinge]), 2u),
                variable<12u>(load(positions, second_opposite[hinge]), 3u),
            };
            const SecondOrder<12u> angle = dihedral(vertices);
            const SecondOrder<12u> elastic_delta = angle_delta(angle, rest_angles[hinge]);
            const SecondOrder<12u> damping_delta = angle_delta(angle, previous_angles[hinge]);
            const SecondOrder<12u> energy = SecondOrder<12u>::constant(bending_stiffness * weights[hinge]) * elastic_delta * elastic_delta;
            const SecondOrder<12u> damping_potential = SecondOrder<12u>::constant(0.5F * bending_damping * weights[hinge] / time_step) * damping_delta * damping_delta;
            angles[hinge] = angle.value;
            angle_deltas[hinge] = elastic_delta.value;
            angle_rates[hinge] = damping_delta.value / time_step;
            energies[hinge] = energy.value;
            damping_potentials[hinge] = damping_potential.value;
            store_derivatives<12u, 4u>(angle, hinge, angle_gradients, angle_hessians);
            store_derivatives<12u, 4u>(energy, hinge, energy_gradients, energy_hessians);
            store_derivatives<12u, 4u>(damping_potential, hinge, damping_residuals, damping_jacobians);
        }

        __global__ void assemble_system_kernel(const std::uint32_t particle_count, const std::uint32_t edge_count, const std::uint32_t triangle_count, const float mass_coefficient, const Vector3<float> gravity, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const std::uint32_t* const hinge_edge_first, const std::uint32_t* const hinge_edge_second, const std::uint32_t* const hinge_first_opposite, const std::uint32_t* const hinge_second_opposite, const std::uint32_t* const vertex_edge_offsets, const std::uint32_t* const vertex_edges, const std::uint32_t* const vertex_triangle_offsets, const std::uint32_t* const vertex_triangles, const std::uint32_t* const vertex_hinge_offsets, const std::uint32_t* const vertex_hinges, const std::uint32_t* const matrix_row_offsets, const std::uint32_t* const matrix_column_indices, const std::uint32_t* const energy_contribution_offsets, const std::uint32_t* const energy_contributions, const std::uint32_t* const damping_contribution_offsets, const std::uint32_t* const damping_contributions, const float* const masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> edge_gradients, const float* const edge_hessians, const simulation::VectorView<const float> triangle_gradients, const float* const triangle_hessians, const simulation::VectorView<const float> hinge_gradients, const float* const hinge_hessians, const simulation::VectorView<const float> damping_residuals, const float* const damping_jacobians, const simulation::VectorView<float> energy_gradient, const simulation::VectorView<float> damping_residual, const simulation::VectorView<float> residual, float* const energy_hessian, float* const damping_jacobian, float* const unregularized_system) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= particle_count) return;
            Vector3<float> elastic{};
            Vector3<float> damping{};
            for (std::uint32_t incidence = vertex_edge_offsets[row]; incidence < vertex_edge_offsets[row + 1u]; ++incidence) {
                const std::uint32_t edge = vertex_edges[incidence];
                elastic = elastic + load(edge_gradients, 2u * edge + (edge_first[edge] == row ? 0u : 1u));
            }
            for (std::uint32_t incidence = vertex_triangle_offsets[row]; incidence < vertex_triangle_offsets[row + 1u]; ++incidence) {
                const std::uint32_t triangle = vertex_triangles[incidence];
                const std::uint32_t local = triangle_first[triangle] == row ? 0u : triangle_second[triangle] == row ? 1u : 2u;
                elastic = elastic + load(triangle_gradients, 3u * triangle + local);
            }
            for (std::uint32_t incidence = vertex_hinge_offsets[row]; incidence < vertex_hinge_offsets[row + 1u]; ++incidence) {
                const std::uint32_t hinge = vertex_hinges[incidence];
                const std::uint32_t local = hinge_local_vertex(row, hinge, hinge_edge_first, hinge_edge_second, hinge_first_opposite, hinge_second_opposite);
                elastic = elastic + load(hinge_gradients, 4u * hinge + local);
                damping = damping + load(damping_residuals, 4u * hinge + local);
            }
            store(energy_gradient, row, elastic);
            store(damping_residual, row, damping);
            store(residual, row, masses[row] * mass_coefficient * (load(positions, row) - load(position_predictor, row)) + elastic + damping - masses[row] * gravity - load(external_forces, row));

            for (std::uint32_t block = matrix_row_offsets[row]; block < matrix_row_offsets[row + 1u]; ++block) {
                float elastic_block[9]{};
                float damping_block[9]{};
                for (std::uint32_t index = energy_contribution_offsets[block]; index < energy_contribution_offsets[block + 1u]; ++index) {
                    const float* const local = energy_hessian_block(energy_contributions[index], edge_count, triangle_count, edge_hessians, triangle_hessians, hinge_hessians);
                    for (std::uint32_t entry = 0u; entry < 9u; ++entry) elastic_block[entry] += local[entry];
                }
                for (std::uint32_t index = damping_contribution_offsets[block]; index < damping_contribution_offsets[block + 1u]; ++index) {
                    const float* const local = damping_jacobians + 9u * damping_contributions[index];
                    for (std::uint32_t entry = 0u; entry < 9u; ++entry) damping_block[entry] += local[entry];
                }
                for (std::uint32_t entry = 0u; entry < 9u; ++entry) {
                    energy_hessian[9u * block + entry] = elastic_block[entry];
                    damping_jacobian[9u * block + entry] = damping_block[entry];
                    unregularized_system[9u * block + entry] = elastic_block[entry] + damping_block[entry];
                }
                if (matrix_column_indices[block] == row) {
                    const float inertia = masses[row] * mass_coefficient;
                    unregularized_system[9u * block] += inertia;
                    unregularized_system[9u * block + 4u] += inertia;
                    unregularized_system[9u * block + 8u] += inertia;
                }
            }
        }

        __global__ void compute_gershgorin_bounds_kernel(const std::uint32_t row_count, const std::uint32_t* const row_offsets, const std::uint32_t* const column_indices, const std::uint32_t* const fixed_vertex_mask, const float* const matrix, float* const lower_bounds) {
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
                        const float value = matrix[9u * block + 3u * component + column_component];
                        if (column == row && column_component == component) diagonal = value;
                        else off_diagonal_sum += fabsf(value);
                    }
                }
                minimum = fminf(minimum, diagonal - off_diagonal_sum);
            }
            lower_bounds[row] = minimum;
        }

        __global__ void reduce_minimum_kernel(const std::uint32_t count, const float* const values, double* const result) {
            __shared__ float partial[block_size];
            float value = FLT_MAX;
            for (std::uint32_t index = threadIdx.x; index < count; index += blockDim.x) value = fminf(value, values[index]);
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] = fminf(partial[threadIdx.x], partial[threadIdx.x + offset]);
                __syncthreads();
            }
            if (threadIdx.x == 0u) result[0] = partial[0];
        }

        __global__ void choose_regularization_kernel(const float positive_margin, const double* const minimum_bound, float* const shift) {
            if (blockIdx.x != 0u || threadIdx.x != 0u) return;
            shift[0] = fmaxf(positive_margin - static_cast<float>(minimum_bound[0]), 0.0F);
        }

        __global__ void build_regularized_system_kernel(const std::uint32_t row_count, const std::uint32_t* const row_offsets, const std::uint32_t* const column_indices, const std::uint32_t* const fixed_vertex_mask, const float* const shift, const float* const unregularized_system, float* const system) {
            const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= row_count) return;
            for (std::uint32_t block = row_offsets[row]; block < row_offsets[row + 1u]; ++block) {
                for (std::uint32_t entry = 0u; entry < 9u; ++entry) system[9u * block + entry] = unregularized_system[9u * block + entry];
                if (fixed_vertex_mask[row] == 0u && column_indices[block] == row) {
                    system[9u * block] += shift[0];
                    system[9u * block + 4u] += shift[0];
                    system[9u * block + 8u] += shift[0];
                }
            }
        }

        __global__ void negate_kernel(const std::uint32_t particle_count, const simulation::VectorView<const float> input, const simulation::VectorView<float> output) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(output, particle, -load(input, particle));
        }

        __global__ void evaluate_directional_derivative_kernel(const std::uint32_t particle_count, const std::uint32_t* const fixed_vertex_mask, const simulation::VectorView<const float> gradient, const simulation::VectorView<const float> direction, double* const result) {
            __shared__ double partial[block_size];
            double value = 0.0;
            for (std::uint32_t particle = threadIdx.x; particle < particle_count; particle += blockDim.x) {
                if (fixed_vertex_mask[particle] != 0u) continue;
                const Vector3<float> first = load(gradient, particle);
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

        __global__ void evaluate_potential_kernel(const std::uint32_t particle_count, const std::uint32_t edge_count, const std::uint32_t triangle_count, const std::uint32_t hinge_count, const float time_step, const float length_stiffness, const float area_stiffness, const float bending_stiffness, const float bending_damping, const Vector3<float> gravity, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const std::uint32_t* const hinge_edge_first, const std::uint32_t* const hinge_edge_second, const std::uint32_t* const hinge_first_opposite, const std::uint32_t* const hinge_second_opposite, const float* const edge_rest_lengths, const float* const triangle_rest_areas, const float* const hinge_rest_angles, const float* const previous_hinge_angles, const float* const hinge_weights, const float* const masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> positions, double* const potential) {
            __shared__ double partial[block_size];
            const std::uint32_t term_count = particle_count + edge_count + triangle_count + hinge_count;
            double value = 0.0;
            for (std::uint32_t term = threadIdx.x; term < term_count; term += blockDim.x) value += potential_term(term, particle_count, edge_count, triangle_count, hinge_count, 0.0F, time_step, length_stiffness, area_stiffness, bending_stiffness, bending_damping, gravity, edge_first, edge_second, triangle_first, triangle_second, triangle_third, hinge_edge_first, hinge_edge_second, hinge_first_opposite, hinge_second_opposite, edge_rest_lengths, triangle_rest_areas, hinge_rest_angles, previous_hinge_angles, hinge_weights, masses, external_forces, position_predictor, positions, positions);
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) potential[0] = partial[0];
        }

        __global__ void evaluate_candidate_potentials_kernel(const std::uint32_t candidate_count, const std::uint32_t particle_count, const std::uint32_t edge_count, const std::uint32_t triangle_count, const std::uint32_t hinge_count, const float time_step, const float length_stiffness, const float area_stiffness, const float bending_stiffness, const float bending_damping, const Vector3<float> gravity, const float* const candidate_steps, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const std::uint32_t* const hinge_edge_first, const std::uint32_t* const hinge_edge_second, const std::uint32_t* const hinge_first_opposite, const std::uint32_t* const hinge_second_opposite, const float* const edge_rest_lengths, const float* const triangle_rest_areas, const float* const hinge_rest_angles, const float* const previous_hinge_angles, const float* const hinge_weights, const float* const masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, double* const potentials) {
            __shared__ double partial[block_size];
            const std::uint32_t candidate = blockIdx.x;
            if (candidate >= candidate_count) return;
            const std::uint32_t term_count = particle_count + edge_count + triangle_count + hinge_count;
            double value = 0.0;
            for (std::uint32_t term = threadIdx.x; term < term_count; term += blockDim.x) value += potential_term(term, particle_count, edge_count, triangle_count, hinge_count, candidate_steps[candidate], time_step, length_stiffness, area_stiffness, bending_stiffness, bending_damping, gravity, edge_first, edge_second, triangle_first, triangle_second, triangle_third, hinge_edge_first, hinge_edge_second, hinge_first_opposite, hinge_second_opposite, edge_rest_lengths, triangle_rest_areas, hinge_rest_angles, previous_hinge_angles, hinge_weights, masses, external_forces, position_predictor, positions, direction);
            partial[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
                __syncthreads();
            }
            if (threadIdx.x == 0u) potentials[candidate] = partial[0];
        }

        __global__ void select_step_size_kernel(const std::uint32_t candidate_count, const float armijo_coefficient, const float* const candidate_steps, const double* const current_potential, const double* const directional_derivative, const double* const candidate_potentials, float* const accepted_step_size, std::uint32_t* const accepted_candidate, double* const accepted_potential) {
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

        __global__ void update_positions_kernel(const std::uint32_t particle_count, const float* const step_size, const simulation::VectorView<const float> direction, const simulation::VectorView<float> positions) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(positions, particle, load(positions, particle) + step_size[0] * load(direction, particle));
        }

        __global__ void reconstruct_newmark_kernel(const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> velocity_predictor, const simulation::VectorView<const float> positions, const simulation::VectorView<float> velocities, const simulation::VectorView<float> accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> acceleration = 4.0F / (time_step * time_step) * (load(positions, particle) - load(position_predictor, particle));
            store(accelerations, particle, acceleration);
            store(velocities, particle, load(velocity_predictor, particle) + 0.5F * time_step * acceleration);
        }
    } // namespace

    void prepare_newmark(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const std::uint32_t* const fixed_vertex_mask, const simulation::VectorView<const float> fixed_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> accelerations, const simulation::VectorView<float> position_predictor, const simulation::VectorView<float> velocity_predictor, const simulation::VectorView<float> candidate_positions) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), prepare_newmark_kernel, particle_count, time_step, fixed_vertex_mask, fixed_positions, positions, velocities, accelerations, position_predictor, velocity_predictor, candidate_positions);
    }

    void evaluate_hinge_angles(const ::cuda::stream_ref stream, const std::uint32_t hinge_count, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const first_opposite, const std::uint32_t* const second_opposite, const simulation::VectorView<const float> positions, float* const angles) {
        if (hinge_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(hinge_count), evaluate_hinge_angles_kernel, hinge_count, edge_first, edge_second, first_opposite, second_opposite, positions, angles);
    }

    void evaluate_edges(const ::cuda::stream_ref stream, const std::uint32_t edge_count, const float length_stiffness, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const float* const rest_lengths, const simulation::VectorView<const float> positions, float* const conditions, float* const energies, const simulation::VectorView<float> gradients, float* const hessians) {
        if (edge_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(edge_count), evaluate_edges_kernel, edge_count, length_stiffness, edge_first, edge_second, rest_lengths, positions, conditions, energies, gradients, hessians);
    }

    void evaluate_triangles(const ::cuda::stream_ref stream, const std::uint32_t triangle_count, const float area_stiffness, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const float* const rest_areas, const simulation::VectorView<const float> positions, float* const conditions, float* const energies, const simulation::VectorView<float> gradients, float* const hessians) {
        if (triangle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(triangle_count), evaluate_triangles_kernel, triangle_count, area_stiffness, triangle_first, triangle_second, triangle_third, rest_areas, positions, conditions, energies, gradients, hessians);
    }

    void evaluate_hinges(const ::cuda::stream_ref stream, const std::uint32_t hinge_count, const float time_step, const float bending_stiffness, const float bending_damping, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const first_opposite, const std::uint32_t* const second_opposite, const float* const rest_angles, const float* const previous_angles, const float* const weights, const simulation::VectorView<const float> positions, float* const angles, float* const angle_deltas, float* const angle_rates, float* const energies, float* const damping_potentials, const simulation::VectorView<float> angle_gradients, float* const angle_hessians, const simulation::VectorView<float> energy_gradients, float* const energy_hessians, const simulation::VectorView<float> damping_residuals, float* const damping_jacobians) {
        if (hinge_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(hinge_count), evaluate_hinges_kernel, hinge_count, time_step, bending_stiffness, bending_damping, edge_first, edge_second, first_opposite, second_opposite, rest_angles, previous_angles, weights, positions, angles, angle_deltas, angle_rates, energies, damping_potentials, angle_gradients, angle_hessians, energy_gradients, energy_hessians, damping_residuals, damping_jacobians);
    }

    void assemble_system(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t edge_count, const std::uint32_t triangle_count, const float mass_coefficient, const Vector3<float> gravity, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const std::uint32_t* const hinge_edge_first, const std::uint32_t* const hinge_edge_second, const std::uint32_t* const hinge_first_opposite, const std::uint32_t* const hinge_second_opposite, const std::uint32_t* const vertex_edge_offsets, const std::uint32_t* const vertex_edges, const std::uint32_t* const vertex_triangle_offsets, const std::uint32_t* const vertex_triangles, const std::uint32_t* const vertex_hinge_offsets, const std::uint32_t* const vertex_hinges, const std::uint32_t* const matrix_row_offsets, const std::uint32_t* const matrix_column_indices, const std::uint32_t* const energy_contribution_offsets, const std::uint32_t* const energy_contributions, const std::uint32_t* const damping_contribution_offsets, const std::uint32_t* const damping_contributions, const float* const masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> edge_gradients, const float* const edge_hessians, const simulation::VectorView<const float> triangle_gradients, const float* const triangle_hessians, const simulation::VectorView<const float> hinge_gradients, const float* const hinge_hessians, const simulation::VectorView<const float> damping_residuals, const float* const damping_jacobians, const simulation::VectorView<float> energy_gradient, const simulation::VectorView<float> damping_residual, const simulation::VectorView<float> residual, float* const energy_hessian, float* const damping_jacobian, float* const unregularized_system) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), assemble_system_kernel, particle_count, edge_count, triangle_count, mass_coefficient, gravity, edge_first, edge_second, triangle_first, triangle_second, triangle_third, hinge_edge_first, hinge_edge_second, hinge_first_opposite, hinge_second_opposite, vertex_edge_offsets, vertex_edges, vertex_triangle_offsets, vertex_triangles, vertex_hinge_offsets, vertex_hinges, matrix_row_offsets, matrix_column_indices, energy_contribution_offsets, energy_contributions, damping_contribution_offsets, damping_contributions, masses, external_forces, position_predictor, positions, edge_gradients, edge_hessians, triangle_gradients, triangle_hessians, hinge_gradients, hinge_hessians, damping_residuals, damping_jacobians, energy_gradient, damping_residual, residual, energy_hessian, damping_jacobian, unregularized_system);
    }

    void compute_gershgorin_bounds(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* const row_offsets, const std::uint32_t* const column_indices, const std::uint32_t* const fixed_vertex_mask, const float* const matrix, float* const lower_bounds) {
        if (row_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), compute_gershgorin_bounds_kernel, row_count, row_offsets, column_indices, fixed_vertex_mask, matrix, lower_bounds);
    }

    void reduce_minimum(const ::cuda::stream_ref stream, const std::uint32_t count, const float* const values, double* const result) {
        if (count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), reduce_minimum_kernel, count, values, result);
    }

    void choose_regularization(const ::cuda::stream_ref stream, const float positive_margin, const double* const minimum_bound, float* const shift) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(1u), choose_regularization_kernel, positive_margin, minimum_bound, shift);
    }

    void build_regularized_system(const ::cuda::stream_ref stream, const std::uint32_t row_count, const std::uint32_t* const row_offsets, const std::uint32_t* const column_indices, const std::uint32_t* const fixed_vertex_mask, const float* const shift, const float* const unregularized_system, float* const system) {
        if (row_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(row_count), build_regularized_system_kernel, row_count, row_offsets, column_indices, fixed_vertex_mask, shift, unregularized_system, system);
    }

    void negate(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const simulation::VectorView<const float> input, const simulation::VectorView<float> output) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), negate_kernel, particle_count, input, output);
    }

    void evaluate_directional_derivative(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* const fixed_vertex_mask, const simulation::VectorView<const float> gradient, const simulation::VectorView<const float> direction, double* const result) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), evaluate_directional_derivative_kernel, particle_count, fixed_vertex_mask, gradient, direction, result);
    }

    void evaluate_potential(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t edge_count, const std::uint32_t triangle_count, const std::uint32_t hinge_count, const float time_step, const float length_stiffness, const float area_stiffness, const float bending_stiffness, const float bending_damping, const Vector3<float> gravity, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const std::uint32_t* const hinge_edge_first, const std::uint32_t* const hinge_edge_second, const std::uint32_t* const hinge_first_opposite, const std::uint32_t* const hinge_second_opposite, const float* const edge_rest_lengths, const float* const triangle_rest_areas, const float* const hinge_rest_angles, const float* const previous_hinge_angles, const float* const hinge_weights, const float* const masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> positions, double* const potential) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(block_size), evaluate_potential_kernel, particle_count, edge_count, triangle_count, hinge_count, time_step, length_stiffness, area_stiffness, bending_stiffness, bending_damping, gravity, edge_first, edge_second, triangle_first, triangle_second, triangle_third, hinge_edge_first, hinge_edge_second, hinge_first_opposite, hinge_second_opposite, edge_rest_lengths, triangle_rest_areas, hinge_rest_angles, previous_hinge_angles, hinge_weights, masses, external_forces, position_predictor, positions, potential);
    }

    void evaluate_candidate_potentials(const ::cuda::stream_ref stream, const std::uint32_t candidate_count, const std::uint32_t particle_count, const std::uint32_t edge_count, const std::uint32_t triangle_count, const std::uint32_t hinge_count, const float time_step, const float length_stiffness, const float area_stiffness, const float bending_stiffness, const float bending_damping, const Vector3<float> gravity, const float* const candidate_steps, const std::uint32_t* const edge_first, const std::uint32_t* const edge_second, const std::uint32_t* const triangle_first, const std::uint32_t* const triangle_second, const std::uint32_t* const triangle_third, const std::uint32_t* const hinge_edge_first, const std::uint32_t* const hinge_edge_second, const std::uint32_t* const hinge_first_opposite, const std::uint32_t* const hinge_second_opposite, const float* const edge_rest_lengths, const float* const triangle_rest_areas, const float* const hinge_rest_angles, const float* const previous_hinge_angles, const float* const hinge_weights, const float* const masses, const simulation::VectorView<const float> external_forces, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> direction, double* const potentials) {
        if (candidate_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(candidate_count * block_size), evaluate_candidate_potentials_kernel, candidate_count, particle_count, edge_count, triangle_count, hinge_count, time_step, length_stiffness, area_stiffness, bending_stiffness, bending_damping, gravity, candidate_steps, edge_first, edge_second, triangle_first, triangle_second, triangle_third, hinge_edge_first, hinge_edge_second, hinge_first_opposite, hinge_second_opposite, edge_rest_lengths, triangle_rest_areas, hinge_rest_angles, previous_hinge_angles, hinge_weights, masses, external_forces, position_predictor, positions, direction, potentials);
    }

    void select_step_size(const ::cuda::stream_ref stream, const std::uint32_t candidate_count, const float armijo_coefficient, const float* const candidate_steps, const double* const current_potential, const double* const directional_derivative, const double* const candidate_potentials, float* const accepted_step_size, std::uint32_t* const accepted_candidate, double* const accepted_potential) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(1u), select_step_size_kernel, candidate_count, armijo_coefficient, candidate_steps, current_potential, directional_derivative, candidate_potentials, accepted_step_size, accepted_candidate, accepted_potential);
    }

    void update_positions(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float* const step_size, const simulation::VectorView<const float> direction, const simulation::VectorView<float> positions) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), update_positions_kernel, particle_count, step_size, direction, positions);
    }

    void reconstruct_newmark(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const simulation::VectorView<const float> position_predictor, const simulation::VectorView<const float> velocity_predictor, const simulation::VectorView<const float> positions, const simulation::VectorView<float> velocities, const simulation::VectorView<float> accelerations) {
        if (particle_count == 0u) return;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), reconstruct_newmark_kernel, particle_count, time_step, position_predictor, velocity_predictor, positions, velocities, accelerations);
    }
} // namespace physica::deformables::cloth::solvers::discrete_shells::kernels
