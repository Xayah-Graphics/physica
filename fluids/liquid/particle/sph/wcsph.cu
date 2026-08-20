#include "wcsph.h"
#include "../density/device.cuh"
#include "../neighborhood/device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace physica::fluids::liquid::particle::cuda_detail::wcsph {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        struct ViscosityReverse {
            Double3 displacement;
            Double3 velocity_difference;
            double neighbor_mass;
            double first_density;
            double second_density;
            double first_viscosity;
            double second_viscosity;
            double first_speed_of_sound;
            double second_speed_of_sound;
        };

        struct SurfaceReverse {
            Double3 displacement;
            double first_mass;
            double second_mass;
            double tension;
        };

        __host__ std::uint32_t blocks(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }


        __device__ ViscosityReverse viscosity_reverse(const Float3 displacement, const Float3 velocity_difference, const float first_density, const float second_density, const float first_viscosity, const float second_viscosity, const float first_speed_of_sound, const float second_speed_of_sound, const float neighbor_mass, const float support_radius, const Double3 acceleration_adjoint) {
            ViscosityReverse result{};
            const double compression = dot(velocity_difference, displacement);
            if (compression >= 0.0) return result;
            const double denominator = dot(displacement, displacement) + 0.01 * support_radius * support_radius;
            const double viscosity = 0.5 * (first_viscosity + second_viscosity);
            const double speed_of_sound = 0.5 * (first_speed_of_sound + second_speed_of_sound);
            const double density_sum = first_density + second_density;
            const double nu = 2.0 * viscosity * support_radius * speed_of_sound / density_sum;
            const double pi_value = -nu * compression / denominator;
            const Float3 gradient = cubic_gradient(displacement, support_radius);
            result.neighbor_mass = -pi_value * dot(acceleration_adjoint, gradient);
            const double pi_adjoint = -neighbor_mass * dot(acceleration_adjoint, gradient);
            const Double3 gradient_adjoint = scale(acceleration_adjoint, -neighbor_mass * pi_value);
            result.displacement = cubic_hessian_product(displacement, gradient_adjoint, support_radius);
            const double nu_adjoint = -compression * pi_adjoint / denominator;
            const double compression_adjoint = -nu * pi_adjoint / denominator;
            const double denominator_adjoint = nu * compression * pi_adjoint / (denominator * denominator);
            result.velocity_difference = scale(displacement, compression_adjoint);
            result.displacement = add(result.displacement, add(scale(velocity_difference, compression_adjoint), scale(displacement, 2.0 * denominator_adjoint)));
            const double viscosity_adjoint = 2.0 * support_radius * speed_of_sound * nu_adjoint / density_sum;
            const double speed_adjoint = 2.0 * viscosity * support_radius * nu_adjoint / density_sum;
            const double density_adjoint = -nu * nu_adjoint / density_sum;
            result.first_viscosity = 0.5 * viscosity_adjoint;
            result.second_viscosity = 0.5 * viscosity_adjoint;
            result.first_speed_of_sound = 0.5 * speed_adjoint;
            result.second_speed_of_sound = 0.5 * speed_adjoint;
            result.first_density = density_adjoint;
            result.second_density = density_adjoint;
            return result;
        }

        __device__ SurfaceReverse surface_reverse(const Float3 displacement, const float first_mass, const float second_mass, const float tension, const float support_radius, const float diameter, const Double3 acceleration_adjoint) {
            SurfaceReverse result{};
            const float distance = length(displacement);
            if (distance >= support_radius) return result;
            const float weight = distance > diameter ? cubic(displacement, support_radius) : cubic({diameter, 0.0F, 0.0F}, support_radius);
            const double factor = -static_cast<double>(tension) * second_mass * weight / first_mass;
            const double factor_adjoint = dot(acceleration_adjoint, displacement);
            result.displacement = scale(acceleration_adjoint, factor);
            if (distance > diameter) result.displacement = add(result.displacement, scale(cubic_gradient(displacement, support_radius), -static_cast<double>(tension) * second_mass * factor_adjoint / first_mass));
            result.tension = -second_mass * weight * factor_adjoint / first_mass;
            result.second_mass = -tension * weight * factor_adjoint / first_mass;
            result.first_mass = tension * second_mass * weight * factor_adjoint / (first_mass * first_mass);
            return result;
        }

        __global__ void external_forward_kernel(const std::uint32_t particle_count, const Float3 gravity, const ConstVectorView<float> controls, const VectorView<float> accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            store(accelerations, particle, add(gravity, load(controls, particle)));
        }

        __global__ void eos_forward_kernel(const std::uint32_t particle_count, const float* densities, const ParticleParameterView particles, const float* speed_of_sound, const float* tait_exponent, float* pressures) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float rest_density = particles.rest_densities[particle];
            const float speed = speed_of_sound[particle];
            const float exponent = tait_exponent[particle];
            const float ratio = densities[particle] / rest_density;
            pressures[particle] = rest_density * speed * speed * (powf(ratio, exponent) - 1.0F) / exponent;
        }

        __global__ void eos_jvp_kernel(const std::uint32_t particle_count, const float* densities, const float* density_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const float* speed_of_sound, const float* speed_of_sound_tangent, const float* tait_exponent, const float* tait_exponent_tangent, float* pressure_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float density = densities[particle];
            const float rest_density = particles.rest_densities[particle];
            const float speed = speed_of_sound[particle];
            const float exponent = tait_exponent[particle];
            const float ratio = density / rest_density;
            const float powered = powf(ratio, exponent);
            const float difference = powered - 1.0F;
            const float density_derivative = speed * speed * powf(ratio, exponent - 1.0F);
            const float rest_density_derivative = speed * speed * difference / exponent - speed * speed * powered;
            const float speed_derivative = 2.0F * rest_density * speed * difference / exponent;
            const float exponent_derivative = rest_density * speed * speed * (exponent * powered * logf(ratio) - difference) / (exponent * exponent);
            pressure_tangent[particle] = density_derivative * density_tangent[particle] + rest_density_derivative * particle_tangent.rest_densities[particle] + speed_derivative * speed_of_sound_tangent[particle] + exponent_derivative * tait_exponent_tangent[particle];
        }

        __global__ void eos_vjp_kernel(const std::uint32_t particle_count, const float* densities, const ParticleParameterView particles, const float* speed_of_sound, const float* tait_exponent, const double* pressure_adjoint, double* density_adjoint, const ParticleParameterAdjointView particle_adjoint, double* speed_of_sound_adjoint, double* tait_exponent_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const double density = densities[particle];
            const double rest_density = particles.rest_densities[particle];
            const double speed = speed_of_sound[particle];
            const double exponent = tait_exponent[particle];
            const double ratio = density / rest_density;
            const double powered = pow(ratio, exponent);
            const double difference = powered - 1.0;
            const double adjoint = pressure_adjoint[particle];
            density_adjoint[particle] += adjoint * speed * speed * pow(ratio, exponent - 1.0);
            particle_adjoint.rest_densities[particle] += adjoint * (speed * speed * difference / exponent - speed * speed * powered);
            speed_of_sound_adjoint[particle] += adjoint * 2.0 * rest_density * speed * difference / exponent;
            tait_exponent_adjoint[particle] += adjoint * rest_density * speed * speed * (exponent * powered * log(ratio) - difference) / (exponent * exponent);
        }

        __global__ void viscosity_forward_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ParticleParameterView particles, const float* speed_of_sound, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const VectorView<float> accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 velocity = load(velocities, particle);
            Float3 acceleration{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 velocity_difference = subtract(velocity, load(velocities, neighbor));
                            const float compression = dot(velocity_difference, displacement);
                            if (compression >= 0.0F) continue;
                            const float alpha = 0.5F * (particles.viscosities[particle] + particles.viscosities[neighbor]);
                            const float speed = 0.5F * (speed_of_sound[particle] + speed_of_sound[neighbor]);
                            const float nu = 2.0F * alpha * support_radius * speed / (densities[particle] + densities[neighbor]);
                            const float pi_value = -nu * compression / (dot(displacement, displacement) + 0.01F * support_radius * support_radius);
                            acceleration = add(acceleration, scale(cubic_gradient(displacement, support_radius), -particles.masses[neighbor] * pi_value));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, boundary_position(boundary, neighbor));
                            const Float3 velocity_difference = subtract(velocity, boundary_velocity(boundary, neighbor));
                            const float compression = dot(velocity_difference, displacement);
                            if (compression >= 0.0F) continue;
                            const float nu = particles.viscosities[particle] * support_radius * speed_of_sound[particle] / densities[particle];
                            const float pi_value = -nu * compression / (dot(displacement, displacement) + 0.01F * support_radius * support_radius);
                            acceleration = add(acceleration, scale(cubic_gradient(displacement, support_radius), -particles.rest_densities[particle] * boundary.volumes[neighbor] * pi_value));
                        }
                    }
            store(accelerations, particle, acceleration);
        }

        __global__ void viscosity_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const float* speed_of_sound, const float* speed_of_sound_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* density_tangent, const VectorView<float> acceleration_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 velocity = load(velocities, particle);
            const Float3 position_dot = load(position_tangent, particle);
            const Float3 velocity_dot = load(velocity_tangent, particle);
            Float3 result{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 velocity_difference = subtract(velocity, load(velocities, neighbor));
                            const float compression = dot(velocity_difference, displacement);
                            if (compression >= 0.0F) continue;
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const Float3 velocity_difference_dot = subtract(velocity_dot, load(velocity_tangent, neighbor));
                            const float compression_dot = dot(velocity_difference_dot, displacement) + dot(velocity_difference, displacement_dot);
                            const float denominator = dot(displacement, displacement) + 0.01F * support_radius * support_radius;
                            const float denominator_dot = 2.0F * dot(displacement, displacement_dot);
                            const float alpha = 0.5F * (particles.viscosities[particle] + particles.viscosities[neighbor]);
                            const float alpha_dot = 0.5F * (particle_tangent.viscosities[particle] + particle_tangent.viscosities[neighbor]);
                            const float speed = 0.5F * (speed_of_sound[particle] + speed_of_sound[neighbor]);
                            const float speed_dot = 0.5F * (speed_of_sound_tangent[particle] + speed_of_sound_tangent[neighbor]);
                            const float density_sum = densities[particle] + densities[neighbor];
                            const float density_sum_dot = density_tangent[particle] + density_tangent[neighbor];
                            const float nu = 2.0F * alpha * support_radius * speed / density_sum;
                            const float nu_dot = 2.0F * support_radius * (alpha_dot * speed + alpha * speed_dot) / density_sum - nu * density_sum_dot / density_sum;
                            const float pi_value = -nu * compression / denominator;
                            const float pi_dot = -(nu_dot * compression + nu * compression_dot) / denominator + nu * compression * denominator_dot / (denominator * denominator);
                            const float mass = particles.masses[neighbor];
                            const float mass_dot = particle_tangent.masses[neighbor];
                            const Float3 gradient = cubic_gradient(displacement, support_radius);
                            const Float3 gradient_dot = cubic_gradient_tangent(displacement, displacement_dot, support_radius);
                            result = add(result, add(scale(gradient, -(mass_dot * pi_value + mass * pi_dot)), scale(gradient_dot, -mass * pi_value)));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, boundary_position(boundary, neighbor));
                            const Float3 velocity_difference = subtract(velocity, boundary_velocity(boundary, neighbor));
                            const float compression = dot(velocity_difference, displacement);
                            if (compression >= 0.0F) continue;
                            const float compression_dot = dot(velocity_dot, displacement) + dot(velocity_difference, position_dot);
                            const float denominator = dot(displacement, displacement) + 0.01F * support_radius * support_radius;
                            const float denominator_dot = 2.0F * dot(displacement, position_dot);
                            const float nu = particles.viscosities[particle] * support_radius * speed_of_sound[particle] / densities[particle];
                            const float nu_dot = support_radius * (particle_tangent.viscosities[particle] * speed_of_sound[particle] + particles.viscosities[particle] * speed_of_sound_tangent[particle]) / densities[particle] - nu * density_tangent[particle] / densities[particle];
                            const float pi_value = -nu * compression / denominator;
                            const float pi_dot = -(nu_dot * compression + nu * compression_dot) / denominator + nu * compression * denominator_dot / (denominator * denominator);
                            const float mass = particles.rest_densities[particle] * boundary.volumes[neighbor];
                            const float mass_dot = particle_tangent.rest_densities[particle] * boundary.volumes[neighbor];
                            const Float3 gradient = cubic_gradient(displacement, support_radius);
                            const Float3 gradient_dot = cubic_gradient_tangent(displacement, position_dot, support_radius);
                            result = add(result, add(scale(gradient, -(mass_dot * pi_value + mass * pi_dot)), scale(gradient_dot, -mass * pi_value)));
                        }
                    }
            store(acceleration_tangent, particle, result);
        }

        __global__ void viscosity_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ParticleParameterView particles, const float* speed_of_sound, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const ConstVectorView<double> acceleration_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, double* density_adjoint, const ParticleParameterAdjointView particle_adjoint, double* speed_of_sound_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 velocity = load(velocities, particle);
            Double3 position_contribution{};
            Double3 velocity_contribution{};
            double mass_contribution = 0.0;
            double rest_density_contribution = 0.0;
            double density_contribution = 0.0;
            double viscosity_contribution = 0.0;
            double speed_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 velocity_difference = subtract(velocity, load(velocities, neighbor));
                            const ViscosityReverse local = viscosity_reverse(displacement, velocity_difference, densities[particle], densities[neighbor], particles.viscosities[particle], particles.viscosities[neighbor], speed_of_sound[particle], speed_of_sound[neighbor], particles.masses[neighbor], support_radius, load(acceleration_adjoint, particle));
                            position_contribution = add(position_contribution, local.displacement);
                            velocity_contribution = add(velocity_contribution, local.velocity_difference);
                            density_contribution += local.first_density;
                            viscosity_contribution += local.first_viscosity;
                            speed_contribution += local.first_speed_of_sound;

                            const Float3 reverse_displacement = scale(displacement, -1.0F);
                            const Float3 reverse_velocity_difference = scale(velocity_difference, -1.0F);
                            const ViscosityReverse cross = viscosity_reverse(reverse_displacement, reverse_velocity_difference, densities[neighbor], densities[particle], particles.viscosities[neighbor], particles.viscosities[particle], speed_of_sound[neighbor], speed_of_sound[particle], particles.masses[particle], support_radius, load(acceleration_adjoint, neighbor));
                            position_contribution = add(position_contribution, scale(cross.displacement, -1.0));
                            velocity_contribution = add(velocity_contribution, scale(cross.velocity_difference, -1.0));
                            mass_contribution += cross.neighbor_mass;
                            density_contribution += cross.second_density;
                            viscosity_contribution += cross.second_viscosity;
                            speed_contribution += cross.second_speed_of_sound;
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, boundary_position(boundary, neighbor));
                            const Float3 velocity_difference = subtract(velocity, boundary_velocity(boundary, neighbor));
                            const float boundary_mass = particles.rest_densities[particle] * boundary.volumes[neighbor];
                            const ViscosityReverse local = viscosity_reverse(displacement, velocity_difference, densities[particle], densities[particle], particles.viscosities[particle], particles.viscosities[particle], speed_of_sound[particle], speed_of_sound[particle], boundary_mass, support_radius, load(acceleration_adjoint, particle));
                            position_contribution = add(position_contribution, local.displacement);
                            velocity_contribution = add(velocity_contribution, local.velocity_difference);
                            rest_density_contribution += local.neighbor_mass * boundary.volumes[neighbor];
                            density_contribution += local.first_density + local.second_density;
                            viscosity_contribution += local.first_viscosity + local.second_viscosity;
                            speed_contribution += local.first_speed_of_sound + local.second_speed_of_sound;
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            accumulate(velocity_adjoint, particle, velocity_contribution);
            density_adjoint[particle] += density_contribution;
            particle_adjoint.masses[particle] += mass_contribution;
            particle_adjoint.rest_densities[particle] += rest_density_contribution;
            particle_adjoint.viscosities[particle] += viscosity_contribution;
            speed_of_sound_adjoint[particle] += speed_contribution;
        }

        __global__ void surface_forward_kernel(const std::uint32_t particle_count, const float support_radius, const float diameter, const ConstVectorView<float> positions, const ParticleParameterView particles, const float* boundary_surface_tension, const NeighborhoodView neighborhood, const BoundaryView boundary, const VectorView<float> accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            Float3 acceleration{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const float distance = length(displacement);
                            if (distance >= support_radius) continue;
                            const float weight = distance > diameter ? cubic(displacement, support_radius) : cubic({diameter, 0.0F, 0.0F}, support_radius);
                            const float factor = -particles.surface_tensions[particle] * particles.masses[neighbor] * weight / particles.masses[particle];
                            acceleration = add(acceleration, scale(displacement, factor));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, boundary_position(boundary, neighbor));
                            const float distance = length(displacement);
                            if (distance >= support_radius) continue;
                            const float weight = distance > diameter ? cubic(displacement, support_radius) : cubic({diameter, 0.0F, 0.0F}, support_radius);
                            const float factor = -boundary_surface_tension[particle] * particles.rest_densities[particle] * boundary.volumes[neighbor] * weight / particles.masses[particle];
                            acceleration = add(acceleration, scale(displacement, factor));
                        }
                    }
            store(accelerations, particle, acceleration);
        }

        __global__ void surface_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const float diameter, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const float* boundary_surface_tension, const float* boundary_surface_tension_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const VectorView<float> acceleration_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 position_dot = load(position_tangent, particle);
            Float3 result{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const float distance = length(displacement);
                            if (distance >= support_radius) continue;
                            const float weight = distance > diameter ? cubic(displacement, support_radius) : cubic({diameter, 0.0F, 0.0F}, support_radius);
                            const float weight_dot = distance > diameter ? dot(cubic_gradient(displacement, support_radius), displacement_dot) : 0.0F;
                            const float first_mass = particles.masses[particle];
                            const float second_mass = particles.masses[neighbor];
                            const float tension = particles.surface_tensions[particle];
                            const float factor = -tension * second_mass * weight / first_mass;
                            const float factor_dot = -particle_tangent.surface_tensions[particle] * second_mass * weight / first_mass - tension * particle_tangent.masses[neighbor] * weight / first_mass - tension * second_mass * weight_dot / first_mass + tension * second_mass * weight * particle_tangent.masses[particle] / (first_mass * first_mass);
                            result = add(result, add(scale(displacement, factor_dot), scale(displacement_dot, factor)));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, boundary_position(boundary, neighbor));
                            const float distance = length(displacement);
                            if (distance >= support_radius) continue;
                            const float weight = distance > diameter ? cubic(displacement, support_radius) : cubic({diameter, 0.0F, 0.0F}, support_radius);
                            const float weight_dot = distance > diameter ? dot(cubic_gradient(displacement, support_radius), position_dot) : 0.0F;
                            const float first_mass = particles.masses[particle];
                            const float boundary_mass = particles.rest_densities[particle] * boundary.volumes[neighbor];
                            const float boundary_mass_dot = particle_tangent.rest_densities[particle] * boundary.volumes[neighbor];
                            const float tension = boundary_surface_tension[particle];
                            const float factor = -tension * boundary_mass * weight / first_mass;
                            const float factor_dot = -boundary_surface_tension_tangent[particle] * boundary_mass * weight / first_mass - tension * boundary_mass_dot * weight / first_mass - tension * boundary_mass * weight_dot / first_mass + tension * boundary_mass * weight * particle_tangent.masses[particle] / (first_mass * first_mass);
                            result = add(result, add(scale(displacement, factor_dot), scale(position_dot, factor)));
                        }
                    }
            store(acceleration_tangent, particle, result);
        }

        __global__ void surface_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const float diameter, const ConstVectorView<float> positions, const ParticleParameterView particles, const float* boundary_surface_tension, const NeighborhoodView neighborhood, const BoundaryView boundary, const ConstVectorView<double> acceleration_adjoint, const VectorView<double> position_adjoint, const ParticleParameterAdjointView particle_adjoint, double* boundary_surface_tension_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            Double3 position_contribution{};
            double mass_contribution = 0.0;
            double rest_density_contribution = 0.0;
            double tension_contribution = 0.0;
            double boundary_tension_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const SurfaceReverse local = surface_reverse(displacement, particles.masses[particle], particles.masses[neighbor], particles.surface_tensions[particle], support_radius, diameter, load(acceleration_adjoint, particle));
                            position_contribution = add(position_contribution, local.displacement);
                            mass_contribution += local.first_mass;
                            tension_contribution += local.tension;

                            const SurfaceReverse cross = surface_reverse(scale(displacement, -1.0F), particles.masses[neighbor], particles.masses[particle], particles.surface_tensions[neighbor], support_radius, diameter, load(acceleration_adjoint, neighbor));
                            position_contribution = add(position_contribution, scale(cross.displacement, -1.0));
                            mass_contribution += cross.second_mass;
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, boundary_position(boundary, neighbor));
                            const float boundary_mass = particles.rest_densities[particle] * boundary.volumes[neighbor];
                            const SurfaceReverse local = surface_reverse(displacement, particles.masses[particle], boundary_mass, boundary_surface_tension[particle], support_radius, diameter, load(acceleration_adjoint, particle));
                            position_contribution = add(position_contribution, local.displacement);
                            mass_contribution += local.first_mass;
                            rest_density_contribution += local.second_mass * boundary.volumes[neighbor];
                            boundary_tension_contribution += local.tension;
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            particle_adjoint.masses[particle] += mass_contribution;
            particle_adjoint.rest_densities[particle] += rest_density_contribution;
            particle_adjoint.surface_tensions[particle] += tension_contribution;
            boundary_surface_tension_adjoint[particle] += boundary_tension_contribution;
        }

    } // namespace

    void launch_external_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float gravity_x, const float gravity_y, const float gravity_z, const ConstVectorView<float> controls, const VectorView<float> accelerations) {
        external_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, {gravity_x, gravity_y, gravity_z}, controls, accelerations);
    }

    void launch_eos_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float* densities, const ParticleParameterView particles, const float* speed_of_sound, const float* tait_exponent, float* pressures) {
        eos_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, densities, particles, speed_of_sound, tait_exponent, pressures);
    }

    void launch_eos_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float* densities, const float* density_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const float* speed_of_sound, const float* speed_of_sound_tangent, const float* tait_exponent, const float* tait_exponent_tangent, float* pressure_tangent) {
        eos_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, densities, density_tangent, particles, particle_tangent, speed_of_sound, speed_of_sound_tangent, tait_exponent, tait_exponent_tangent, pressure_tangent);
    }

    void launch_eos_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float* densities, const ParticleParameterView particles, const float* speed_of_sound, const float* tait_exponent, const double* pressure_adjoint, double* density_adjoint, const ParticleParameterAdjointView particle_adjoint, double* speed_of_sound_adjoint, double* tait_exponent_adjoint) {
        eos_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, densities, particles, speed_of_sound, tait_exponent, pressure_adjoint, density_adjoint, particle_adjoint, speed_of_sound_adjoint, tait_exponent_adjoint);
    }

    void launch_artificial_viscosity_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ParticleParameterView particles, const float* speed_of_sound, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const VectorView<float> accelerations) {
        viscosity_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, velocities, particles, speed_of_sound, neighborhood, boundary, densities, accelerations);
    }

    void launch_artificial_viscosity_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const float* speed_of_sound, const float* speed_of_sound_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* density_tangent, const VectorView<float> acceleration_tangent) {
        viscosity_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, velocities, position_tangent, velocity_tangent, particles, particle_tangent, speed_of_sound, speed_of_sound_tangent, neighborhood, boundary, densities, density_tangent, acceleration_tangent);
    }

    void launch_artificial_viscosity_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ParticleParameterView particles, const float* speed_of_sound, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const ConstVectorView<double> acceleration_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, double* density_adjoint, const ParticleParameterAdjointView particle_adjoint, double* speed_of_sound_adjoint) {
        viscosity_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, velocities, particles, speed_of_sound, neighborhood, boundary, densities, acceleration_adjoint, position_adjoint, velocity_adjoint, density_adjoint, particle_adjoint, speed_of_sound_adjoint);
    }

    void launch_surface_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const float particle_radius, const ConstVectorView<float> positions, const ParticleParameterView particles, const float* boundary_surface_tension, const NeighborhoodView neighborhood, const BoundaryView boundary, const VectorView<float> accelerations) {
        surface_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, 2.0F * particle_radius, positions, particles, boundary_surface_tension, neighborhood, boundary, accelerations);
    }

    void launch_surface_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const float particle_radius, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView particles, const ParticleParameterTangentView particle_tangent, const float* boundary_surface_tension, const float* boundary_surface_tension_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const VectorView<float> acceleration_tangent) {
        surface_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, 2.0F * particle_radius, positions, position_tangent, particles, particle_tangent, boundary_surface_tension, boundary_surface_tension_tangent, neighborhood, boundary, acceleration_tangent);
    }

    void launch_surface_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const float particle_radius, const ConstVectorView<float> positions, const ParticleParameterView particles, const float* boundary_surface_tension, const NeighborhoodView neighborhood, const BoundaryView boundary, const ConstVectorView<double> acceleration_adjoint, const VectorView<double> position_adjoint, const ParticleParameterAdjointView particle_adjoint, double* boundary_surface_tension_adjoint) {
        surface_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, 2.0F * particle_radius, positions, particles, boundary_surface_tension, neighborhood, boundary, acceleration_adjoint, position_adjoint, particle_adjoint, boundary_surface_tension_adjoint);
    }

} // namespace physica::fluids::liquid::particle::cuda_detail::wcsph
