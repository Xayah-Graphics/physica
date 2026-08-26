#include "../../detail/cuda/device.cuh"
#include "pbf-kernels.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace physica::fluids::liquid::cuda_detail::pbf {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        __host__ std::uint32_t blocks(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }
        __device__ float poly6_radial_derivative(const float radius, const float support_radius) {
            const float squared_radius = support_radius * support_radius;
            if (radius >= support_radius) return 0.0F;
            const float difference  = squared_radius - radius * radius;
            const float coefficient = 315.0F / (64.0F * pi * powf(support_radius, 9.0F));
            return -6.0F * coefficient * radius * difference * difference;
        }

        __device__ Float3 spiky_gradient(const Float3 displacement, const float support_radius) {
            const float distance = length(displacement);
            if (distance == 0.0F || distance >= support_radius) return {};
            const float difference  = support_radius - distance;
            const float coefficient = -45.0F / (pi * powf(support_radius, 6.0F));
            return scale(displacement, coefficient * difference * difference / distance);
        }

        __device__ Float3 spiky_gradient_tangent(const Float3 displacement, const Float3 displacement_tangent, const float support_radius) {
            const float distance = length(displacement);
            if (distance == 0.0F || distance >= support_radius) return {};
            const Float3 direction         = scale(displacement, 1.0F / distance);
            const float radial_tangent     = dot(direction, displacement_tangent);
            const Float3 direction_tangent = scale(add(displacement_tangent, scale(direction, -radial_tangent)), 1.0F / distance);
            const float difference         = support_radius - distance;
            const float coefficient        = -45.0F / (pi * powf(support_radius, 6.0F));
            return scale(add(scale(direction, -2.0F * difference * radial_tangent), scale(direction_tangent, difference * difference)), coefficient);
        }

        __device__ Double3 spiky_hessian_product(const Float3 displacement, const Double3 vector, const float support_radius) {
            const double distance = sqrt(static_cast<double>(dot(displacement, displacement)));
            if (distance == 0.0 || distance >= support_radius) return {};
            const Double3 direction         = scale(displacement, 1.0 / distance);
            const double projection         = dot(direction, vector);
            const Double3 direction_tangent = scale(subtract(vector, scale(direction, projection)), 1.0 / distance);
            const double difference         = support_radius - distance;
            const double coefficient        = -45.0 / (static_cast<double>(pi) * pow(static_cast<double>(support_radius), 6.0));
            return scale(add(scale(direction, -2.0 * difference * projection), scale(direction_tangent, difference * difference)), coefficient);
        }

        __device__ void artificial_pressure(const Float3 displacement, const float support_radius, const float strength, const float exponent, const float radius, float& value, Float3& gradient, float& strength_derivative, float& exponent_derivative, float& radius_derivative) {
            const float reference = poly6({radius, 0.0F, 0.0F}, support_radius);
            const float kernel    = poly6(displacement, support_radius);
            if (kernel == 0.0F) {
                value = strength_derivative = exponent_derivative = radius_derivative = 0.0F;
                gradient                                                              = {};
                return;
            }
            const float ratio   = kernel / reference;
            const float power   = powf(ratio, exponent);
            value               = -strength * power;
            strength_derivative = -power;
            exponent_derivative = value * logf(ratio);
            radius_derivative   = -value * exponent * poly6_radial_derivative(radius, support_radius) / reference;
            gradient            = scale(poly6_gradient(displacement, support_radius), -strength * exponent * powf(ratio, exponent - 1.0F) / reference);
        }

        __global__ void predict_forward_kernel(const std::uint32_t particle_count, const float time_step, const Float3 gravity, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> controls, const VectorView<float> predicted_positions) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(add(gravity, load(controls, particle)), time_step));
            store(predicted_positions, particle, add(load(positions, particle), scale(predicted_velocity, time_step)));
        }

        __global__ void predict_jvp_kernel(const std::uint32_t particle_count, const float time_step, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ConstVectorView<float> control_tangent, const VectorView<float> predicted_position_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(predicted_position_tangent, particle, add(load(position_tangent, particle), scale(add(load(velocity_tangent, particle), scale(load(control_tangent, particle), time_step)), time_step)));
        }

        __global__ void predict_vjp_kernel(const std::uint32_t particle_count, const float time_step, const ConstVectorView<double> predicted_position_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> control_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Double3 adjoint = load(predicted_position_adjoint, particle);
            accumulate(position_adjoint, particle, adjoint);
            accumulate(velocity_adjoint, particle, scale(adjoint, time_step));
            accumulate(control_adjoint, particle, scale(adjoint, time_step * time_step));
        }

        __global__ void lambda_forward_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView particles, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* relaxation, const VectorView<float> gradient_sums, float* denominators, float* lambdas) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const float rest_density       = particles.rest_densities[particle];
            Float3 self_gradient{};
            float denominator = relaxation[particle];
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 gradient   = spiky_gradient(subtract(position, load(positions, neighbor)), support_radius);
                            const float coefficient = particles.masses[neighbor] / rest_density;
                            self_gradient           = add(self_gradient, scale(gradient, coefficient));
                            denominator += coefficient * coefficient * dot(gradient, gradient);
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            self_gradient                = add(self_gradient, scale(spiky_gradient(subtract(position, boundary_position(boundary, neighbor)), support_radius), boundary.volumes[neighbor]));
                        }
                    }
            denominator += dot(self_gradient, self_gradient);
            store(gradient_sums, particle, self_gradient);
            denominators[particle] = denominator;
            lambdas[particle]      = -(densities[particle] / rest_density - 1.0F) / denominator;
        }

        __global__ void lambda_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* density_tangent, const float* relaxation, const float* relaxation_tangent, const VectorView<float> gradient_sum_tangent, float* denominator_tangent, float* lambda_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const Float3 position_dot      = load(position_tangent, particle);
            const float rest_density       = particles.rest_densities[particle];
            const float rest_density_dot   = particle_tangent.rest_densities[particle];
            Float3 self_gradient{};
            Float3 self_gradient_dot{};
            float denominator     = relaxation[particle];
            float denominator_dot = relaxation_tangent[particle];
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement          = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot      = subtract(position_dot, load(position_tangent, neighbor));
                            const Float3 gradient              = spiky_gradient(displacement, support_radius);
                            const Float3 gradient_dot          = spiky_gradient_tangent(displacement, displacement_dot, support_radius);
                            const float coefficient            = particles.masses[neighbor] / rest_density;
                            const float coefficient_dot        = particle_tangent.masses[neighbor] / rest_density - particles.masses[neighbor] * rest_density_dot / (rest_density * rest_density);
                            const Float3 neighbor_gradient     = scale(gradient, coefficient);
                            const Float3 neighbor_gradient_dot = add(scale(gradient, coefficient_dot), scale(gradient_dot, coefficient));
                            self_gradient                      = add(self_gradient, neighbor_gradient);
                            self_gradient_dot                  = add(self_gradient_dot, neighbor_gradient_dot);
                            denominator += dot(neighbor_gradient, neighbor_gradient);
                            denominator_dot += 2.0F * dot(neighbor_gradient, neighbor_gradient_dot);
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            self_gradient                = add(self_gradient, scale(spiky_gradient(displacement, support_radius), boundary.volumes[neighbor]));
                            self_gradient_dot            = add(self_gradient_dot, scale(spiky_gradient_tangent(displacement, position_dot, support_radius), boundary.volumes[neighbor]));
                        }
                    }
            denominator += dot(self_gradient, self_gradient);
            denominator_dot += 2.0F * dot(self_gradient, self_gradient_dot);
            const float constraint     = densities[particle] / rest_density - 1.0F;
            const float constraint_dot = density_tangent[particle] / rest_density - densities[particle] * rest_density_dot / (rest_density * rest_density);
            store(gradient_sum_tangent, particle, self_gradient_dot);
            denominator_tangent[particle] = denominator_dot;
            lambda_tangent[particle]      = -constraint_dot / denominator + constraint * denominator_dot / (denominator * denominator);
        }

        __global__ void lambda_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView particles, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const ConstVectorView<float> gradient_sums, const float* denominators, const double* lambda_adjoint, const VectorView<double> position_adjoint, double* density_adjoint, const ParticleParameterAdjointView particle_adjoint, double* relaxation_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position            = load(topology_positions, particle);
            const Float3 position                     = load(positions, particle);
            const float rest_density                  = particles.rest_densities[particle];
            const double local_lambda_adjoint         = lambda_adjoint[particle];
            const double local_constraint             = static_cast<double>(densities[particle]) / rest_density - 1.0;
            const double local_denominator_adjoint    = local_lambda_adjoint * local_constraint / (static_cast<double>(denominators[particle]) * denominators[particle]);
            const Double3 local_self_gradient_adjoint = scale(load(gradient_sums, particle), 2.0 * local_denominator_adjoint);
            Double3 position_contribution{};
            double mass_contribution         = 0.0;
            double rest_density_contribution = local_lambda_adjoint * densities[particle] / (static_cast<double>(denominators[particle]) * rest_density * rest_density);
            Float3 particle_self_gradient{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement               = subtract(position, load(positions, neighbor));
                            const Float3 gradient                   = spiky_gradient(displacement, support_radius);
                            const float coefficient                 = particles.masses[neighbor] / rest_density;
                            const Float3 neighbor_gradient          = scale(gradient, -coefficient);
                            const Double3 neighbor_gradient_adjoint = scale(neighbor_gradient, 2.0 * local_denominator_adjoint);
                            const Double3 gradient_adjoint          = scale(subtract(local_self_gradient_adjoint, neighbor_gradient_adjoint), coefficient);
                            position_contribution                   = add(position_contribution, spiky_hessian_product(displacement, gradient_adjoint, support_radius));
                            particle_self_gradient                  = add(particle_self_gradient, scale(gradient, coefficient));

                            const float neighbor_rest_density               = particles.rest_densities[neighbor];
                            const double neighbor_constraint                = static_cast<double>(densities[neighbor]) / neighbor_rest_density - 1.0;
                            const double neighbor_denominator_adjoint       = lambda_adjoint[neighbor] * neighbor_constraint / (static_cast<double>(denominators[neighbor]) * denominators[neighbor]);
                            const Double3 neighbor_self_gradient_adjoint    = scale(load(gradient_sums, neighbor), 2.0 * neighbor_denominator_adjoint);
                            const Float3 reverse_displacement               = scale(displacement, -1.0F);
                            const Float3 reverse_gradient                   = spiky_gradient(reverse_displacement, support_radius);
                            const float reverse_coefficient                 = particles.masses[particle] / neighbor_rest_density;
                            const Float3 reverse_neighbor_gradient          = scale(reverse_gradient, -reverse_coefficient);
                            const Double3 reverse_neighbor_gradient_adjoint = scale(reverse_neighbor_gradient, 2.0 * neighbor_denominator_adjoint);
                            const Double3 reverse_gradient_adjoint          = scale(subtract(neighbor_self_gradient_adjoint, reverse_neighbor_gradient_adjoint), reverse_coefficient);
                            position_contribution                           = add(position_contribution, scale(spiky_hessian_product(reverse_displacement, reverse_gradient_adjoint, support_radius), -1.0));
                            mass_contribution += dot(subtract(neighbor_self_gradient_adjoint, reverse_neighbor_gradient_adjoint), reverse_gradient) / neighbor_rest_density;
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            position_contribution        = add(position_contribution, spiky_hessian_product(displacement, scale(local_self_gradient_adjoint, boundary.volumes[neighbor]), support_radius));
                        }
                    }
            rest_density_contribution -= dot(local_self_gradient_adjoint, particle_self_gradient) / rest_density;
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 gradient                   = spiky_gradient(subtract(position, load(positions, neighbor)), support_radius);
                            const Float3 neighbor_gradient          = scale(gradient, -particles.masses[neighbor] / rest_density);
                            const Double3 neighbor_gradient_adjoint = scale(neighbor_gradient, 2.0 * local_denominator_adjoint);
                            rest_density_contribution -= dot(neighbor_gradient_adjoint, neighbor_gradient) / rest_density;
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            density_adjoint[particle] -= local_lambda_adjoint / (static_cast<double>(denominators[particle]) * rest_density);
            particle_adjoint.masses[particle] += mass_contribution;
            particle_adjoint.rest_densities[particle] += rest_density_contribution;
            relaxation_adjoint[particle] += local_denominator_adjoint;
        }

        __global__ void correction_forward_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView particles, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* lambdas, const float* strength, const float* exponent, const float* radius, const VectorView<float> corrections) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const float rest_density       = particles.rest_densities[particle];
            Float3 correction{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            float artificial, strength_derivative, exponent_derivative, radius_derivative;
                            Float3 artificial_gradient;
                            artificial_pressure(displacement, support_radius, strength[particle], exponent[particle], radius[particle], artificial, artificial_gradient, strength_derivative, exponent_derivative, radius_derivative);
                            const float coefficient = particles.masses[neighbor] / rest_density;
                            correction              = add(correction, scale(spiky_gradient(displacement, support_radius), coefficient * (lambdas[particle] + lambdas[neighbor] + artificial)));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            float artificial, strength_derivative, exponent_derivative, radius_derivative;
                            Float3 artificial_gradient;
                            artificial_pressure(displacement, support_radius, strength[particle], exponent[particle], radius[particle], artificial, artificial_gradient, strength_derivative, exponent_derivative, radius_derivative);
                            correction = add(correction, scale(spiky_gradient(displacement, support_radius), boundary.volumes[neighbor] * (lambdas[particle] + artificial)));
                        }
                    }
            store(corrections, particle, correction);
        }

        __global__ void correction_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* lambdas, const float* lambda_tangent, const float* strength, const float* strength_tangent, const float* exponent, const float* exponent_tangent, const float* radius, const float* radius_tangent, const VectorView<float> correction_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const Float3 position_dot      = load(position_tangent, particle);
            const float rest_density       = particles.rest_densities[particle];
            const float rest_density_dot   = particle_tangent.rest_densities[particle];
            Float3 result{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement     = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            float artificial, strength_derivative, exponent_derivative, radius_derivative;
                            Float3 artificial_gradient;
                            artificial_pressure(displacement, support_radius, strength[particle], exponent[particle], radius[particle], artificial, artificial_gradient, strength_derivative, exponent_derivative, radius_derivative);
                            const float artificial_dot  = dot(artificial_gradient, displacement_dot) + strength_derivative * strength_tangent[particle] + exponent_derivative * exponent_tangent[particle] + radius_derivative * radius_tangent[particle];
                            const float coefficient     = particles.masses[neighbor] / rest_density;
                            const float coefficient_dot = particle_tangent.masses[neighbor] / rest_density - particles.masses[neighbor] * rest_density_dot / (rest_density * rest_density);
                            const float multiplier      = lambdas[particle] + lambdas[neighbor] + artificial;
                            const float multiplier_dot  = lambda_tangent[particle] + lambda_tangent[neighbor] + artificial_dot;
                            result                      = add(result, add(scale(spiky_gradient(displacement, support_radius), coefficient_dot * multiplier + coefficient * multiplier_dot), scale(spiky_gradient_tangent(displacement, displacement_dot, support_radius), coefficient * multiplier)));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            float artificial, strength_derivative, exponent_derivative, radius_derivative;
                            Float3 artificial_gradient;
                            artificial_pressure(displacement, support_radius, strength[particle], exponent[particle], radius[particle], artificial, artificial_gradient, strength_derivative, exponent_derivative, radius_derivative);
                            const float artificial_dot = dot(artificial_gradient, position_dot) + strength_derivative * strength_tangent[particle] + exponent_derivative * exponent_tangent[particle] + radius_derivative * radius_tangent[particle];
                            const float multiplier     = lambdas[particle] + artificial;
                            result                     = add(result, scale(add(scale(spiky_gradient(displacement, support_radius), lambda_tangent[particle] + artificial_dot), scale(spiky_gradient_tangent(displacement, position_dot, support_radius), multiplier)), boundary.volumes[neighbor]));
                        }
                    }
            store(correction_tangent, particle, result);
        }

        __global__ void correction_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView particles, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* lambdas, const float* strength, const float* exponent, const float* radius, const ConstVectorView<double> correction_adjoint, const VectorView<double> position_adjoint, double* lambda_adjoint, const ParticleParameterAdjointView particle_adjoint, double* strength_adjoint, double* exponent_adjoint, double* radius_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position     = load(topology_positions, particle);
            const Float3 position              = load(positions, particle);
            const float rest_density           = particles.rest_densities[particle];
            const Double3 local_output_adjoint = load(correction_adjoint, particle);
            Double3 position_contribution{};
            double lambda_contribution       = 0.0;
            double mass_contribution         = 0.0;
            double rest_density_contribution = 0.0;
            double strength_contribution     = 0.0;
            double exponent_contribution     = 0.0;
            double radius_contribution       = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 gradient     = spiky_gradient(displacement, support_radius);
                            float artificial, strength_derivative, exponent_derivative, radius_derivative;
                            Float3 artificial_gradient;
                            artificial_pressure(displacement, support_radius, strength[particle], exponent[particle], radius[particle], artificial, artificial_gradient, strength_derivative, exponent_derivative, radius_derivative);
                            const float coefficient     = particles.masses[neighbor] / rest_density;
                            const float multiplier      = lambdas[particle] + lambdas[neighbor] + artificial;
                            const double scalar_adjoint = coefficient * dot(local_output_adjoint, gradient);
                            position_contribution       = add(position_contribution, spiky_hessian_product(displacement, scale(local_output_adjoint, coefficient * multiplier), support_radius));
                            position_contribution       = add(position_contribution, scale(artificial_gradient, scalar_adjoint));
                            lambda_contribution += scalar_adjoint;
                            rest_density_contribution -= dot(local_output_adjoint, scale(gradient, coefficient * multiplier)) / rest_density;
                            strength_contribution += scalar_adjoint * strength_derivative;
                            exponent_contribution += scalar_adjoint * exponent_derivative;
                            radius_contribution += scalar_adjoint * radius_derivative;

                            const Float3 reverse_displacement = scale(displacement, -1.0F);
                            const Float3 reverse_gradient     = spiky_gradient(reverse_displacement, support_radius);
                            float reverse_artificial, reverse_strength_derivative, reverse_exponent_derivative, reverse_radius_derivative;
                            Float3 reverse_artificial_gradient;
                            artificial_pressure(reverse_displacement, support_radius, strength[neighbor], exponent[neighbor], radius[neighbor], reverse_artificial, reverse_artificial_gradient, reverse_strength_derivative, reverse_exponent_derivative, reverse_radius_derivative);
                            const float reverse_rest_density     = particles.rest_densities[neighbor];
                            const float reverse_coefficient      = particles.masses[particle] / reverse_rest_density;
                            const float reverse_multiplier       = lambdas[neighbor] + lambdas[particle] + reverse_artificial;
                            const Double3 reverse_output_adjoint = load(correction_adjoint, neighbor);
                            const double reverse_scalar_adjoint  = reverse_coefficient * dot(reverse_output_adjoint, reverse_gradient);
                            position_contribution                = add(position_contribution, scale(spiky_hessian_product(reverse_displacement, scale(reverse_output_adjoint, reverse_coefficient * reverse_multiplier), support_radius), -1.0));
                            position_contribution                = add(position_contribution, scale(reverse_artificial_gradient, -reverse_scalar_adjoint));
                            lambda_contribution += reverse_scalar_adjoint;
                            mass_contribution += dot(reverse_output_adjoint, scale(reverse_gradient, reverse_multiplier / reverse_rest_density));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            const Float3 gradient        = spiky_gradient(displacement, support_radius);
                            float artificial, strength_derivative, exponent_derivative, radius_derivative;
                            Float3 artificial_gradient;
                            artificial_pressure(displacement, support_radius, strength[particle], exponent[particle], radius[particle], artificial, artificial_gradient, strength_derivative, exponent_derivative, radius_derivative);
                            const float volume          = boundary.volumes[neighbor];
                            const float multiplier      = lambdas[particle] + artificial;
                            const double scalar_adjoint = volume * dot(local_output_adjoint, gradient);
                            position_contribution       = add(position_contribution, spiky_hessian_product(displacement, scale(local_output_adjoint, volume * multiplier), support_radius));
                            position_contribution       = add(position_contribution, scale(artificial_gradient, scalar_adjoint));
                            lambda_contribution += scalar_adjoint;
                            strength_contribution += scalar_adjoint * strength_derivative;
                            exponent_contribution += scalar_adjoint * exponent_derivative;
                            radius_contribution += scalar_adjoint * radius_derivative;
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            lambda_adjoint[particle] += lambda_contribution;
            particle_adjoint.masses[particle] += mass_contribution;
            particle_adjoint.rest_densities[particle] += rest_density_contribution;
            strength_adjoint[particle] += strength_contribution;
            exponent_adjoint[particle] += exponent_contribution;
            radius_adjoint[particle] += radius_contribution;
        }

        __global__ void project_forward_kernel(const std::uint32_t particle_count, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> corrections, std::uint32_t* collision_masks, const VectorView<float> projected_positions) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 candidate = add(load(positions, particle), load(corrections, particle));
            std::uint32_t mask     = 0u;
            if (candidate.x < domain.minimum_x || candidate.x > domain.maximum_x) mask |= 1u;
            if (candidate.y < domain.minimum_y || candidate.y > domain.maximum_y) mask |= 2u;
            if (candidate.z < domain.minimum_z || candidate.z > domain.maximum_z) mask |= 4u;
            collision_masks[particle] = mask;
            store(projected_positions, particle,
                {
                    fminf(domain.maximum_x, fmaxf(domain.minimum_x, candidate.x)),
                    fminf(domain.maximum_y, fmaxf(domain.minimum_y, candidate.y)),
                    fminf(domain.maximum_z, fmaxf(domain.minimum_z, candidate.z)),
                });
        }

        __global__ void project_jvp_kernel(const std::uint32_t particle_count, const std::uint32_t* collision_masks, const ConstVectorView<float> position_tangent, const ConstVectorView<float> correction_tangent, const VectorView<float> projected_position_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Float3 result            = add(load(position_tangent, particle), load(correction_tangent, particle));
            const std::uint32_t mask = collision_masks[particle];
            if ((mask & 1u) != 0u) result.x = 0.0F;
            if ((mask & 2u) != 0u) result.y = 0.0F;
            if ((mask & 4u) != 0u) result.z = 0.0F;
            store(projected_position_tangent, particle, result);
        }

        __global__ void project_vjp_kernel(const std::uint32_t particle_count, const std::uint32_t* collision_masks, const ConstVectorView<double> projected_position_adjoint, const VectorView<double> position_adjoint, const VectorView<double> correction_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Double3 result           = load(projected_position_adjoint, particle);
            const std::uint32_t mask = collision_masks[particle];
            if ((mask & 1u) != 0u) result.x = 0.0;
            if ((mask & 2u) != 0u) result.y = 0.0;
            if ((mask & 4u) != 0u) result.z = 0.0;
            accumulate(position_adjoint, particle, result);
            accumulate(correction_adjoint, particle, result);
        }

        __global__ void reconstruct_forward_kernel(const std::uint32_t particle_count, const float inverse_time_step, const ConstVectorView<float> positions, const ConstVectorView<float> corrected_positions, const VectorView<float> velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(velocities, particle, scale(subtract(load(corrected_positions, particle), load(positions, particle)), inverse_time_step));
        }

        __global__ void reconstruct_jvp_kernel(const std::uint32_t particle_count, const float inverse_time_step, const ConstVectorView<float> position_tangent, const ConstVectorView<float> corrected_position_tangent, const VectorView<float> velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(velocity_tangent, particle, scale(subtract(load(corrected_position_tangent, particle), load(position_tangent, particle)), inverse_time_step));
        }

        __global__ void reconstruct_vjp_kernel(const std::uint32_t particle_count, const float inverse_time_step, const ConstVectorView<double> velocity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> corrected_position_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Double3 contribution = scale(load(velocity_adjoint, particle), inverse_time_step);
            accumulate(position_adjoint, particle, scale(contribution, -1.0));
            accumulate(corrected_position_adjoint, particle, contribution);
        }

        __global__ void vorticity_forward_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const NeighborhoodView neighborhood, const VectorView<float> vorticities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const Float3 velocity          = load(velocities, particle);
            Float3 vorticity{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            vorticity = add(vorticity, cross(subtract(load(velocities, neighbor), velocity), spiky_gradient(subtract(position, load(positions, neighbor)), support_radius)));
                        }
                    }
            store(vorticities, particle, vorticity);
        }

        __global__ void vorticity_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const NeighborhoodView neighborhood, const VectorView<float> vorticity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const Float3 velocity          = load(velocities, particle);
            const Float3 position_dot      = load(position_tangent, particle);
            const Float3 velocity_dot      = load(velocity_tangent, particle);
            Float3 result{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement     = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const Float3 difference       = subtract(load(velocities, neighbor), velocity);
                            const Float3 difference_dot   = subtract(load(velocity_tangent, neighbor), velocity_dot);
                            result                        = add(result, add(cross(difference_dot, spiky_gradient(displacement, support_radius)), cross(difference, spiky_gradient_tangent(displacement, displacement_dot, support_radius))));
                        }
                    }
            store(vorticity_tangent, particle, result);
        }

        __global__ void vorticity_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const NeighborhoodView neighborhood, const ConstVectorView<double> vorticity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position     = load(topology_positions, particle);
            const Float3 position              = load(positions, particle);
            const Float3 velocity              = load(velocities, particle);
            const Double3 local_output_adjoint = load(vorticity_adjoint, particle);
            Double3 position_contribution{};
            Double3 velocity_contribution{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 gradient     = spiky_gradient(displacement, support_radius);
                            const Float3 difference   = subtract(load(velocities, neighbor), velocity);
                            velocity_contribution     = add(velocity_contribution, cross(local_output_adjoint, gradient));
                            position_contribution     = add(position_contribution, spiky_hessian_product(displacement, cross(local_output_adjoint, difference), support_radius));

                            const Double3 reverse_output_adjoint = load(vorticity_adjoint, neighbor);
                            const Float3 reverse_displacement    = scale(displacement, -1.0F);
                            const Float3 reverse_gradient        = spiky_gradient(reverse_displacement, support_radius);
                            const Float3 reverse_difference      = subtract(velocity, load(velocities, neighbor));
                            velocity_contribution                = add(velocity_contribution, cross(reverse_gradient, reverse_output_adjoint));
                            position_contribution                = add(position_contribution, scale(spiky_hessian_product(reverse_displacement, cross(reverse_output_adjoint, reverse_difference), support_radius), -1.0));
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            accumulate(velocity_adjoint, particle, velocity_contribution);
        }

        __global__ void magnitude_forward_kernel(const std::uint32_t particle_count, const ConstVectorView<float> vorticities, float* magnitudes) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            magnitudes[particle] = length(load(vorticities, particle));
        }

        __global__ void normal_forward_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const NeighborhoodView neighborhood, const float* magnitudes, const VectorView<float> normals, float* normalizers) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            Float3 gradient{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            gradient = add(gradient, scale(spiky_gradient(subtract(position, load(positions, neighbor)), support_radius), magnitudes[neighbor] - magnitudes[particle]));
                        }
                    }
            const float normalizer = length(gradient);
            normalizers[particle]  = normalizer;
            store(normals, particle, normalizer > 0.0F ? scale(gradient, 1.0F / normalizer) : Float3{});
        }

        __global__ void magnitude_jvp_kernel(const std::uint32_t particle_count, const ConstVectorView<float> vorticities, const ConstVectorView<float> vorticity_tangent, const float* magnitudes, float* magnitude_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            magnitude_tangent[particle] = magnitudes[particle] > 0.0F ? dot(load(vorticities, particle), load(vorticity_tangent, particle)) / magnitudes[particle] : 0.0F;
        }

        __global__ void normal_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const NeighborhoodView neighborhood, const float* magnitudes, const float* magnitude_tangent, const ConstVectorView<float> normals, const float* normalizers, const VectorView<float> normal_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const Float3 position_dot      = load(position_tangent, particle);
            Float3 gradient_dot{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement     = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const float difference        = magnitudes[neighbor] - magnitudes[particle];
                            const float difference_dot    = magnitude_tangent[neighbor] - magnitude_tangent[particle];
                            gradient_dot                  = add(gradient_dot, add(scale(spiky_gradient(displacement, support_radius), difference_dot), scale(spiky_gradient_tangent(displacement, displacement_dot, support_radius), difference)));
                        }
                    }
            if (normalizers[particle] == 0.0F) {
                store(normal_tangent, particle, {});
                return;
            }
            const Float3 normal = load(normals, particle);
            store(normal_tangent, particle, scale(add(gradient_dot, scale(normal, -dot(normal, gradient_dot))), 1.0F / normalizers[particle]));
        }

        __device__ Double3 normalized_input_adjoint(const Float3 normal, const float normalizer, const Double3 normal_adjoint) {
            if (normalizer == 0.0F) return {};
            return scale(add(normal_adjoint, scale(normal, -dot(normal_adjoint, normal))), 1.0 / normalizer);
        }

        __global__ void normal_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> vorticities, const NeighborhoodView neighborhood, const float* magnitudes, const ConstVectorView<float> normals, const float* normalizers, const ConstVectorView<double> normal_adjoint, const VectorView<double> position_adjoint, const VectorView<double> vorticity_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position       = load(topology_positions, particle);
            const Float3 position                = load(positions, particle);
            const Double3 local_gradient_adjoint = normalized_input_adjoint(load(normals, particle), normalizers[particle], load(normal_adjoint, particle));
            Double3 position_contribution{};
            double magnitude_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 gradient     = spiky_gradient(displacement, support_radius);
                            const float difference    = magnitudes[neighbor] - magnitudes[particle];
                            position_contribution     = add(position_contribution, spiky_hessian_product(displacement, scale(local_gradient_adjoint, difference), support_radius));
                            magnitude_contribution -= dot(local_gradient_adjoint, gradient);

                            const Float3 reverse_displacement      = scale(displacement, -1.0F);
                            const Float3 reverse_gradient          = spiky_gradient(reverse_displacement, support_radius);
                            const Double3 reverse_gradient_adjoint = normalized_input_adjoint(load(normals, neighbor), normalizers[neighbor], load(normal_adjoint, neighbor));
                            const float reverse_difference         = magnitudes[particle] - magnitudes[neighbor];
                            position_contribution                  = add(position_contribution, scale(spiky_hessian_product(reverse_displacement, scale(reverse_gradient_adjoint, reverse_difference), support_radius), -1.0));
                            magnitude_contribution += dot(reverse_gradient_adjoint, reverse_gradient);
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            if (magnitudes[particle] > 0.0F) accumulate(vorticity_adjoint, particle, scale(load(vorticities, particle), magnitude_contribution / magnitudes[particle]));
        }

        __global__ void confinement_forward_kernel(const std::uint32_t particle_count, const float time_step, const ConstVectorView<float> velocities, const ConstVectorView<float> vorticities, const ConstVectorView<float> normals, const float* confinement, const VectorView<float> confined_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(confined_velocities, particle, add(load(velocities, particle), scale(cross(load(normals, particle), load(vorticities, particle)), time_step * confinement[particle])));
        }

        __global__ void confinement_jvp_kernel(const std::uint32_t particle_count, const float time_step, const ConstVectorView<float> vorticities, const ConstVectorView<float> normals, const float* confinement, const ConstVectorView<float> velocity_tangent, const ConstVectorView<float> vorticity_tangent, const ConstVectorView<float> normal_tangent, const float* confinement_tangent, const VectorView<float> confined_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 force     = cross(load(normals, particle), load(vorticities, particle));
            const Float3 force_dot = add(cross(load(normal_tangent, particle), load(vorticities, particle)), cross(load(normals, particle), load(vorticity_tangent, particle)));
            store(confined_velocity_tangent, particle, add(load(velocity_tangent, particle), scale(add(scale(force, confinement_tangent[particle]), scale(force_dot, confinement[particle])), time_step)));
        }

        __global__ void confinement_vjp_kernel(const std::uint32_t particle_count, const float time_step, const ConstVectorView<float> vorticities, const ConstVectorView<float> normals, const float* confinement, const ConstVectorView<double> confined_velocity_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> vorticity_adjoint, const VectorView<double> normal_adjoint, double* confinement_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Double3 output_adjoint = load(confined_velocity_adjoint, particle);
            const Float3 vorticity       = load(vorticities, particle);
            const Float3 normal          = load(normals, particle);
            accumulate(velocity_adjoint, particle, output_adjoint);
            accumulate(vorticity_adjoint, particle, scale(cross(output_adjoint, normal), time_step * confinement[particle]));
            accumulate(normal_adjoint, particle, scale(cross(vorticity, output_adjoint), time_step * confinement[particle]));
            confinement_adjoint[particle] += time_step * dot(output_adjoint, cross(normal, vorticity));
        }

        __global__ void xsph_forward_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const NeighborhoodView neighborhood, const float* viscosity, const VectorView<float> output_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const Float3 velocity          = load(velocities, particle);
            Float3 smoothing{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            smoothing = add(smoothing, scale(subtract(load(velocities, neighbor), velocity), poly6(subtract(position, load(positions, neighbor)), support_radius)));
                        }
                    }
            store(output_velocities, particle, add(velocity, scale(smoothing, viscosity[particle])));
        }

        __global__ void xsph_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const NeighborhoodView neighborhood, const float* viscosity, const float* viscosity_tangent, const VectorView<float> output_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position = load(topology_positions, particle);
            const Float3 position          = load(positions, particle);
            const Float3 velocity          = load(velocities, particle);
            const Float3 position_dot      = load(position_tangent, particle);
            const Float3 velocity_dot      = load(velocity_tangent, particle);
            Float3 smoothing{};
            Float3 smoothing_dot{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement     = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const Float3 difference       = subtract(load(velocities, neighbor), velocity);
                            const Float3 difference_dot   = subtract(load(velocity_tangent, neighbor), velocity_dot);
                            const float weight            = poly6(displacement, support_radius);
                            const float weight_dot        = dot(poly6_gradient(displacement, support_radius), displacement_dot);
                            smoothing                     = add(smoothing, scale(difference, weight));
                            smoothing_dot                 = add(smoothing_dot, add(scale(difference_dot, weight), scale(difference, weight_dot)));
                        }
                    }
            store(output_velocity_tangent, particle, add(velocity_dot, add(scale(smoothing, viscosity_tangent[particle]), scale(smoothing_dot, viscosity[particle]))));
        }

        __global__ void xsph_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const NeighborhoodView neighborhood, const float* viscosity, const ConstVectorView<double> output_velocity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, double* viscosity_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 topology_position     = load(topology_positions, particle);
            const Float3 position              = load(positions, particle);
            const Float3 velocity              = load(velocities, particle);
            const Double3 local_output_adjoint = load(output_velocity_adjoint, particle);
            Double3 position_contribution{};
            Double3 velocity_contribution = local_output_adjoint;
            double viscosity_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, topology_position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 difference   = subtract(load(velocities, neighbor), velocity);
                            const float weight        = poly6(displacement, support_radius);
                            velocity_contribution     = add(velocity_contribution, scale(local_output_adjoint, -viscosity[particle] * weight));
                            position_contribution     = add(position_contribution, scale(poly6_gradient(displacement, support_radius), viscosity[particle] * dot(local_output_adjoint, difference)));
                            viscosity_contribution += weight * dot(local_output_adjoint, difference);

                            const Float3 reverse_displacement    = scale(displacement, -1.0F);
                            const Double3 reverse_output_adjoint = load(output_velocity_adjoint, neighbor);
                            const Float3 reverse_difference      = subtract(velocity, load(velocities, neighbor));
                            const float reverse_weight           = poly6(reverse_displacement, support_radius);
                            velocity_contribution                = add(velocity_contribution, scale(reverse_output_adjoint, viscosity[neighbor] * reverse_weight));
                            position_contribution                = add(position_contribution, scale(poly6_gradient(reverse_displacement, support_radius), -viscosity[neighbor] * dot(reverse_output_adjoint, reverse_difference)));
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            accumulate(velocity_adjoint, particle, velocity_contribution);
            viscosity_adjoint[particle] += viscosity_contribution;
        }

    } // namespace

    void launch_predict_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const float gravity_x, const float gravity_y, const float gravity_z, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> controls, const VectorView<float> predicted_positions) {
        predict_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, {gravity_x, gravity_y, gravity_z}, positions, velocities, controls, predicted_positions);
    }

    void launch_predict_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ConstVectorView<float> control_tangent, const VectorView<float> predicted_position_tangent) {
        predict_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, position_tangent, velocity_tangent, control_tangent, predicted_position_tangent);
    }

    void launch_predict_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const ConstVectorView<double> predicted_position_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> control_adjoint) {
        predict_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, predicted_position_adjoint, position_adjoint, velocity_adjoint, control_adjoint);
    }

    void launch_lambda_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView particles, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* relaxation, const VectorView<float> gradient_sums, float* denominators, float* lambdas) {
        lambda_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, particles, neighborhood, boundary, densities, relaxation, gradient_sums, denominators, lambdas);
    }

    void launch_lambda_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* density_tangent, const float* relaxation, const float* relaxation_tangent, const VectorView<float> gradient_sum_tangent, float* denominator_tangent, float* lambda_tangent) {
        lambda_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, position_tangent, particles, particle_tangent, neighborhood, boundary, densities, density_tangent, relaxation, relaxation_tangent, gradient_sum_tangent, denominator_tangent, lambda_tangent);
    }

    void launch_lambda_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView particles, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const ConstVectorView<float> gradient_sums, const float* denominators, const double* lambda_adjoint, const VectorView<double> position_adjoint, double* density_adjoint, const ParticleParameterAdjointView particle_adjoint, double* relaxation_adjoint) {
        lambda_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, particles, neighborhood, boundary, densities, gradient_sums, denominators, lambda_adjoint, position_adjoint, density_adjoint, particle_adjoint, relaxation_adjoint);
    }

    void launch_correction_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView particles, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* lambdas, const float* artificial_pressure_strength, const float* artificial_pressure_exponent, const float* artificial_pressure_radius, const VectorView<float> corrections) {
        correction_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, particles, neighborhood, boundary, lambdas, artificial_pressure_strength, artificial_pressure_exponent, artificial_pressure_radius, corrections);
    }

    void launch_correction_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* lambdas, const float* lambda_tangent, const float* artificial_pressure_strength, const float* artificial_pressure_strength_tangent, const float* artificial_pressure_exponent, const float* artificial_pressure_exponent_tangent, const float* artificial_pressure_radius, const float* artificial_pressure_radius_tangent, const VectorView<float> correction_tangent) {
        correction_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, position_tangent, particles, particle_tangent, neighborhood, boundary, lambdas, lambda_tangent, artificial_pressure_strength, artificial_pressure_strength_tangent, artificial_pressure_exponent, artificial_pressure_exponent_tangent, artificial_pressure_radius, artificial_pressure_radius_tangent, correction_tangent);
    }

    void launch_correction_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView particles, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* lambdas, const float* artificial_pressure_strength, const float* artificial_pressure_exponent, const float* artificial_pressure_radius, const ConstVectorView<double> correction_adjoint, const VectorView<double> position_adjoint, double* lambda_adjoint, const ParticleParameterAdjointView particle_adjoint, double* artificial_pressure_strength_adjoint, double* artificial_pressure_exponent_adjoint, double* artificial_pressure_radius_adjoint) {
        correction_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, particles, neighborhood, boundary, lambdas, artificial_pressure_strength, artificial_pressure_exponent, artificial_pressure_radius, correction_adjoint, position_adjoint, lambda_adjoint, particle_adjoint, artificial_pressure_strength_adjoint, artificial_pressure_exponent_adjoint, artificial_pressure_radius_adjoint);
    }

    void launch_project_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> corrections, std::uint32_t* collision_masks, const VectorView<float> projected_positions) {
        project_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, domain, positions, corrections, collision_masks, projected_positions);
    }

    void launch_project_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* collision_masks, const ConstVectorView<float> position_tangent, const ConstVectorView<float> correction_tangent, const VectorView<float> projected_position_tangent) {
        project_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, collision_masks, position_tangent, correction_tangent, projected_position_tangent);
    }

    void launch_project_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t* collision_masks, const ConstVectorView<double> projected_position_adjoint, const VectorView<double> position_adjoint, const VectorView<double> correction_adjoint) {
        project_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, collision_masks, projected_position_adjoint, position_adjoint, correction_adjoint);
    }

    void launch_reconstruct_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float inverse_time_step, const ConstVectorView<float> positions, const ConstVectorView<float> corrected_positions, const VectorView<float> velocities) {
        reconstruct_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, inverse_time_step, positions, corrected_positions, velocities);
    }

    void launch_reconstruct_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float inverse_time_step, const ConstVectorView<float> position_tangent, const ConstVectorView<float> corrected_position_tangent, const VectorView<float> velocity_tangent) {
        reconstruct_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, inverse_time_step, position_tangent, corrected_position_tangent, velocity_tangent);
    }

    void launch_reconstruct_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float inverse_time_step, const ConstVectorView<double> velocity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> corrected_position_adjoint) {
        reconstruct_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, inverse_time_step, velocity_adjoint, position_adjoint, corrected_position_adjoint);
    }

    void launch_vorticity_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const NeighborhoodView neighborhood, const VectorView<float> vorticities) {
        vorticity_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, velocities, neighborhood, vorticities);
    }

    void launch_vorticity_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const NeighborhoodView neighborhood, const VectorView<float> vorticity_tangent) {
        vorticity_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, velocities, position_tangent, velocity_tangent, neighborhood, vorticity_tangent);
    }

    void launch_vorticity_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const NeighborhoodView neighborhood, const ConstVectorView<double> vorticity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint) {
        vorticity_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, velocities, neighborhood, vorticity_adjoint, position_adjoint, velocity_adjoint);
    }

    void launch_normal_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> vorticities, const NeighborhoodView neighborhood, float* magnitudes, const VectorView<float> normals, float* normalizers) {
        magnitude_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, vorticities, magnitudes);
        normal_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, neighborhood, magnitudes, normals, normalizers);
    }

    void launch_normal_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> vorticities, const ConstVectorView<float> position_tangent, const ConstVectorView<float> vorticity_tangent, const NeighborhoodView neighborhood, const float* magnitudes, const ConstVectorView<float> normals, const float* normalizers, float* magnitude_tangent, const VectorView<float> normal_tangent) {
        magnitude_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, vorticities, vorticity_tangent, magnitudes, magnitude_tangent);
        normal_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, position_tangent, neighborhood, magnitudes, magnitude_tangent, normals, normalizers, normal_tangent);
    }

    void launch_normal_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> vorticities, const NeighborhoodView neighborhood, const float* magnitudes, const ConstVectorView<float> normals, const float* normalizers, const ConstVectorView<double> normal_adjoint, const VectorView<double> position_adjoint, const VectorView<double> vorticity_adjoint) {
        normal_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, vorticities, neighborhood, magnitudes, normals, normalizers, normal_adjoint, position_adjoint, vorticity_adjoint);
    }

    void launch_confinement_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const ConstVectorView<float> velocities, const ConstVectorView<float> vorticities, const ConstVectorView<float> normals, const float* confinement, const VectorView<float> confined_velocities) {
        confinement_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, velocities, vorticities, normals, confinement, confined_velocities);
    }

    void launch_confinement_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const ConstVectorView<float> vorticities, const ConstVectorView<float> normals, const float* confinement, const ConstVectorView<float> velocity_tangent, const ConstVectorView<float> vorticity_tangent, const ConstVectorView<float> normal_tangent, const float* confinement_tangent, const VectorView<float> confined_velocity_tangent) {
        confinement_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, vorticities, normals, confinement, velocity_tangent, vorticity_tangent, normal_tangent, confinement_tangent, confined_velocity_tangent);
    }

    void launch_confinement_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const ConstVectorView<float> vorticities, const ConstVectorView<float> normals, const float* confinement, const ConstVectorView<double> confined_velocity_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> vorticity_adjoint, const VectorView<double> normal_adjoint, double* confinement_adjoint) {
        confinement_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, vorticities, normals, confinement, confined_velocity_adjoint, velocity_adjoint, vorticity_adjoint, normal_adjoint, confinement_adjoint);
    }

    void launch_xsph_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const NeighborhoodView neighborhood, const float* viscosity, const VectorView<float> output_velocities) {
        xsph_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, velocities, neighborhood, viscosity, output_velocities);
    }

    void launch_xsph_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const NeighborhoodView neighborhood, const float* viscosity, const float* viscosity_tangent, const VectorView<float> output_velocity_tangent) {
        xsph_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, velocities, position_tangent, velocity_tangent, neighborhood, viscosity, viscosity_tangent, output_velocity_tangent);
    }

    void launch_xsph_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const NeighborhoodView neighborhood, const float* viscosity, const ConstVectorView<double> output_velocity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, double* viscosity_adjoint) {
        xsph_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, topology_positions, positions, velocities, neighborhood, viscosity, output_velocity_adjoint, position_adjoint, velocity_adjoint, viscosity_adjoint);
    }

} // namespace physica::fluids::liquid::cuda_detail::pbf
