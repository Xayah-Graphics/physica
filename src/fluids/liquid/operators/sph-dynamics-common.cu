#include "../detail/cuda/device.cuh"

#include "sph-dynamics-kernels.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace physica::fluids::liquid::cuda_detail::sph {
    namespace {
        constexpr std::uint32_t block_size = 256u;
        __host__ std::uint32_t blocks(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }
        __global__ void non_pressure_forward_kernel(const std::uint32_t particle_count, const float support_radius, const Float3 gravity, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> controls, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const VectorView<float> accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 velocity = load(velocities, particle);
            Float3 acceleration   = add(gravity, load(controls, particle));
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const float viscosity     = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float weight        = viscosity * parameters.masses[neighbor] * viscosity_laplacian(displacement, support_radius) / densities[neighbor];
                            acceleration              = add(acceleration, scale(subtract(load(velocities, neighbor), velocity), weight));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            const float weight           = parameters.viscosities[particle] * boundary.volumes[neighbor] * viscosity_laplacian(displacement, support_radius);
                            acceleration                 = add(acceleration, scale(subtract(boundary_velocity(boundary, neighbor), velocity), weight));
                        }
                    }
            store(accelerations, particle, acceleration);
        }

        __global__ void non_pressure_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> control_tangent, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ParticleParameterView parameters, const ParticleParameterTangentView parameter_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* density_tangent, const VectorView<float> acceleration_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position     = load(positions, particle);
            const Float3 velocity     = load(velocities, particle);
            const Float3 position_dot = load(position_tangent, particle);
            const Float3 velocity_dot = load(velocity_tangent, particle);
            Float3 result             = load(control_tangent, particle);
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement     = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const float laplacian         = viscosity_laplacian(displacement, support_radius);
                            const float laplacian_dot     = viscosity_laplacian_tangent(displacement, displacement_dot, support_radius);
                            const float viscosity         = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float viscosity_dot     = 0.5F * (parameter_tangent.viscosities[particle] + parameter_tangent.viscosities[neighbor]);
                            const float mass              = parameters.masses[neighbor];
                            const float density           = densities[neighbor];
                            const float weight            = viscosity * mass * laplacian / density;
                            const float weight_dot        = viscosity_dot * mass * laplacian / density + viscosity * parameter_tangent.masses[neighbor] * laplacian / density + viscosity * mass * laplacian_dot / density - viscosity * mass * laplacian * density_tangent[neighbor] / (density * density);
                            const Float3 difference       = subtract(load(velocities, neighbor), velocity);
                            const Float3 difference_dot   = subtract(load(velocity_tangent, neighbor), velocity_dot);
                            result                        = add(result, add(scale(difference, weight_dot), scale(difference_dot, weight)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            const float laplacian        = viscosity_laplacian(displacement, support_radius);
                            const float weight           = parameters.viscosities[particle] * boundary.volumes[neighbor] * laplacian;
                            const float weight_dot       = boundary.volumes[neighbor] * (parameter_tangent.viscosities[particle] * laplacian + parameters.viscosities[particle] * viscosity_laplacian_tangent(displacement, position_dot, support_radius));
                            const Float3 difference      = subtract(boundary_velocity(boundary, neighbor), velocity);
                            result                       = add(result, add(scale(difference, weight_dot), scale(velocity_dot, -weight)));
                        }
                    }
            store(acceleration_tangent, particle, result);
        }

        __global__ void non_pressure_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const ConstVectorView<double> acceleration_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> control_adjoint, double* density_adjoint, const ParticleParameterAdjointView parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position       = load(positions, particle);
            const Float3 velocity       = load(velocities, particle);
            const Double3 local_adjoint = load(acceleration_adjoint, particle);
            Double3 position_contribution{};
            Double3 velocity_contribution{};
            double density_contribution   = 0.0;
            double mass_contribution      = 0.0;
            double viscosity_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement      = subtract(position, load(positions, neighbor));
                            const Float3 difference        = subtract(load(velocities, neighbor), velocity);
                            const float laplacian          = viscosity_laplacian(displacement, support_radius);
                            const float viscosity          = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float mass               = parameters.masses[neighbor];
                            const float density            = densities[neighbor];
                            const float weight             = viscosity * mass * laplacian / density;
                            const double weight_adjoint    = dot(local_adjoint, difference);
                            velocity_contribution          = add(velocity_contribution, scale(local_adjoint, -weight));
                            const double viscosity_adjoint = weight_adjoint * mass * laplacian / density;
                            viscosity_contribution += 0.5 * viscosity_adjoint;
                            position_contribution = add(position_contribution, viscosity_laplacian_gradient(displacement, support_radius, weight_adjoint * viscosity * mass / density));

                            const Double3 cross_adjoint       = load(acceleration_adjoint, neighbor);
                            const float cross_weight          = viscosity * parameters.masses[particle] * laplacian / densities[particle];
                            const double cross_weight_adjoint = dot(cross_adjoint, scale(difference, -1.0F));
                            velocity_contribution             = add(velocity_contribution, scale(cross_adjoint, cross_weight));
                            mass_contribution += cross_weight_adjoint * viscosity * laplacian / densities[particle];
                            density_contribution -= cross_weight_adjoint * viscosity * parameters.masses[particle] * laplacian / (densities[particle] * densities[particle]);
                            viscosity_contribution += 0.5 * cross_weight_adjoint * parameters.masses[particle] * laplacian / densities[particle];
                            position_contribution = add(position_contribution, viscosity_laplacian_gradient(displacement, support_radius, cross_weight_adjoint * viscosity * parameters.masses[particle] / densities[particle]));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            const Float3 difference      = subtract(boundary_velocity(boundary, neighbor), velocity);
                            const float laplacian        = viscosity_laplacian(displacement, support_radius);
                            const float weight           = parameters.viscosities[particle] * boundary.volumes[neighbor] * laplacian;
                            const double weight_adjoint  = dot(local_adjoint, difference);
                            velocity_contribution        = add(velocity_contribution, scale(local_adjoint, -weight));
                            viscosity_contribution += weight_adjoint * boundary.volumes[neighbor] * laplacian;
                            position_contribution = add(position_contribution, viscosity_laplacian_gradient(displacement, support_radius, weight_adjoint * parameters.viscosities[particle] * boundary.volumes[neighbor]));
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            accumulate(velocity_adjoint, particle, velocity_contribution);
            accumulate(control_adjoint, particle, local_adjoint);
            density_adjoint[particle] += density_contribution;
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.viscosities[particle] += viscosity_contribution;
        }

        __global__ void pressure_forward_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* pressures, const VectorView<float> accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            Float3 acceleration{};
            const float first_term = pressures[particle] / (densities[particle] * densities[particle]);
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const float second_term = pressures[neighbor] / (densities[neighbor] * densities[neighbor]);
                            acceleration            = add(acceleration, scale(cubic_gradient(subtract(position, load(positions, neighbor)), support_radius), -parameters.masses[neighbor] * (first_term + second_term)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            acceleration                 = add(acceleration, scale(cubic_gradient(subtract(position, boundary_position(boundary, neighbor)), support_radius), -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term));
                        }
                    }
            store(accelerations, particle, acceleration);
        }

        __global__ void pressure_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView parameters, const ParticleParameterTangentView parameter_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* density_tangent, const float* pressures, const float* pressure_tangent, const VectorView<float> acceleration_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position      = load(positions, particle);
            const Float3 position_dot  = load(position_tangent, particle);
            const float density        = densities[particle];
            const float first_term     = pressures[particle] / (density * density);
            const float first_term_dot = pressure_tangent[particle] / (density * density) - 2.0F * pressures[particle] * density_tangent[particle] / (density * density * density);
            Float3 result{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const float neighbor_density  = densities[neighbor];
                            const float second_term       = pressures[neighbor] / (neighbor_density * neighbor_density);
                            const float second_term_dot   = pressure_tangent[neighbor] / (neighbor_density * neighbor_density) - 2.0F * pressures[neighbor] * density_tangent[neighbor] / (neighbor_density * neighbor_density * neighbor_density);
                            const Float3 displacement     = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const float factor            = -parameters.masses[neighbor] * (first_term + second_term);
                            const float factor_dot        = -parameter_tangent.masses[neighbor] * (first_term + second_term) - parameters.masses[neighbor] * (first_term_dot + second_term_dot);
                            result                        = add(result, add(scale(cubic_gradient(displacement, support_radius), factor_dot), scale(cubic_gradient_tangent(displacement, displacement_dot, support_radius), factor)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            const float factor           = -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term;
                            const float factor_dot       = -boundary.volumes[neighbor] * (parameter_tangent.rest_densities[particle] * first_term + parameters.rest_densities[particle] * first_term_dot);
                            result                       = add(result, add(scale(cubic_gradient(displacement, support_radius), factor_dot), scale(cubic_gradient_tangent(displacement, position_dot, support_radius), factor)));
                        }
                    }
            store(acceleration_tangent, particle, result);
        }

        __global__ void pressure_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* pressures, const ConstVectorView<double> acceleration_adjoint, const VectorView<double> position_adjoint, double* density_adjoint, double* pressure_adjoint, const ParticleParameterAdjointView parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position       = load(positions, particle);
            const Double3 local_adjoint = load(acceleration_adjoint, particle);
            const double first_term     = static_cast<double>(pressures[particle]) / (densities[particle] * densities[particle]);
            Double3 position_contribution{};
            double first_term_adjoint        = 0.0;
            double mass_contribution         = 0.0;
            double rest_density_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement         = subtract(position, load(positions, neighbor));
                            const Float3 gradient             = cubic_gradient(displacement, support_radius);
                            const double second_term          = static_cast<double>(pressures[neighbor]) / (densities[neighbor] * densities[neighbor]);
                            const double sum                  = first_term + second_term;
                            const double local_factor         = -parameters.masses[neighbor] * sum;
                            const double local_factor_adjoint = dot(local_adjoint, gradient);
                            first_term_adjoint -= parameters.masses[neighbor] * local_factor_adjoint;
                            position_contribution = add(position_contribution, cubic_hessian_product(displacement, scale(local_adjoint, local_factor), support_radius));

                            const Double3 cross_adjoint   = load(acceleration_adjoint, neighbor);
                            const double cross_projection = dot(cross_adjoint, gradient);
                            mass_contribution += sum * cross_projection;
                            first_term_adjoint += parameters.masses[particle] * cross_projection;
                            position_contribution = add(position_contribution, cubic_hessian_product(displacement, scale(cross_adjoint, parameters.masses[particle] * sum), support_radius));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement    = subtract(position, boundary_position(boundary, neighbor));
                            const Float3 gradient        = cubic_gradient(displacement, support_radius);
                            const double factor          = -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term;
                            const double factor_adjoint  = dot(local_adjoint, gradient);
                            rest_density_contribution -= factor_adjoint * boundary.volumes[neighbor] * first_term;
                            first_term_adjoint -= factor_adjoint * parameters.rest_densities[particle] * boundary.volumes[neighbor];
                            position_contribution = add(position_contribution, cubic_hessian_product(displacement, scale(local_adjoint, factor), support_radius));
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.rest_densities[particle] += rest_density_contribution;
            pressure_adjoint[particle] += first_term_adjoint / (densities[particle] * densities[particle]);
            density_adjoint[particle] -= 2.0 * first_term_adjoint * pressures[particle] / (densities[particle] * densities[particle] * densities[particle]);
        }

        __device__ void collision_mask(const Box domain, const Float3 predicted_position, bool& collision_x, bool& collision_y, bool& collision_z) {
            collision_x = predicted_position.x < domain.minimum_x || predicted_position.x > domain.maximum_x;
            collision_y = predicted_position.y < domain.minimum_y || predicted_position.y > domain.maximum_y;
            collision_z = predicted_position.z < domain.minimum_z || predicted_position.z > domain.maximum_z;
        }

        __global__ void integrate_forward_kernel(const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> accelerations, const VectorView<float> next_positions, const VectorView<float> next_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Float3 velocity = add(load(velocities, particle), scale(load(accelerations, particle), time_step));
            Float3 position = add(load(positions, particle), scale(velocity, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, position, collision_x, collision_y, collision_z);
            position.x = fminf(domain.maximum_x, fmaxf(domain.minimum_x, position.x));
            position.y = fminf(domain.maximum_y, fmaxf(domain.minimum_y, position.y));
            position.z = fminf(domain.maximum_z, fmaxf(domain.minimum_z, position.z));
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity = {domain.velocity_x, domain.velocity_y, domain.velocity_z};
            else {
                if (collision_x) velocity.x = domain.velocity_x;
                if (collision_y) velocity.y = domain.velocity_y;
                if (collision_z) velocity.z = domain.velocity_z;
            }
            store(next_positions, particle, position);
            store(next_velocities, particle, velocity);
        }

        __global__ void integrate_jvp_kernel(const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> accelerations, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ConstVectorView<float> acceleration_tangent, const VectorView<float> next_position_tangent, const VectorView<float> next_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(load(accelerations, particle), time_step));
            const Float3 predicted_position = add(load(positions, particle), scale(predicted_velocity, time_step));
            Float3 velocity_dot             = add(load(velocity_tangent, particle), scale(load(acceleration_tangent, particle), time_step));
            Float3 position_dot             = add(load(position_tangent, particle), scale(velocity_dot, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, predicted_position, collision_x, collision_y, collision_z);
            if (collision_x) position_dot.x = 0.0F;
            if (collision_y) position_dot.y = 0.0F;
            if (collision_z) position_dot.z = 0.0F;
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_dot = {};
            else {
                if (collision_x) velocity_dot.x = 0.0F;
                if (collision_y) velocity_dot.y = 0.0F;
                if (collision_z) velocity_dot.z = 0.0F;
            }
            store(next_position_tangent, particle, position_dot);
            store(next_velocity_tangent, particle, velocity_dot);
        }

        __global__ void integrate_vjp_kernel(const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> accelerations, const ConstVectorView<double> next_position_adjoint, const ConstVectorView<double> next_velocity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> acceleration_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(load(accelerations, particle), time_step));
            const Float3 predicted_position = add(load(positions, particle), scale(predicted_velocity, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, predicted_position, collision_x, collision_y, collision_z);
            double position_bar_x = collision_x ? 0.0 : next_position_adjoint.x[particle];
            double position_bar_y = collision_y ? 0.0 : next_position_adjoint.y[particle];
            double position_bar_z = collision_z ? 0.0 : next_position_adjoint.z[particle];
            double velocity_bar_x = next_velocity_adjoint.x[particle] + time_step * position_bar_x;
            double velocity_bar_y = next_velocity_adjoint.y[particle] + time_step * position_bar_y;
            double velocity_bar_z = next_velocity_adjoint.z[particle] + time_step * position_bar_z;
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_bar_x = velocity_bar_y = velocity_bar_z = 0.0;
            else {
                if (collision_x) velocity_bar_x = 0.0;
                if (collision_y) velocity_bar_y = 0.0;
                if (collision_z) velocity_bar_z = 0.0;
            }
            accumulate(position_adjoint, particle, {position_bar_x, position_bar_y, position_bar_z});
            accumulate(velocity_adjoint, particle, {velocity_bar_x, velocity_bar_y, velocity_bar_z});
            accumulate(acceleration_adjoint, particle, {time_step * velocity_bar_x, time_step * velocity_bar_y, time_step * velocity_bar_z});
        }

        __global__ void predict_forward_kernel(const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> non_pressure_accelerations, const ConstVectorView<float> pressure_accelerations, const VectorView<float> predicted_positions, const VectorView<float> predicted_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Float3 velocity = add(load(velocities, particle), scale(add(load(non_pressure_accelerations, particle), load(pressure_accelerations, particle)), time_step));
            Float3 position = add(load(positions, particle), scale(velocity, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, position, collision_x, collision_y, collision_z);
            position.x = fminf(domain.maximum_x, fmaxf(domain.minimum_x, position.x));
            position.y = fminf(domain.maximum_y, fmaxf(domain.minimum_y, position.y));
            position.z = fminf(domain.maximum_z, fmaxf(domain.minimum_z, position.z));
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity = {domain.velocity_x, domain.velocity_y, domain.velocity_z};
            else {
                if (collision_x) velocity.x = domain.velocity_x;
                if (collision_y) velocity.y = domain.velocity_y;
                if (collision_z) velocity.z = domain.velocity_z;
            }
            store(predicted_positions, particle, position);
            store(predicted_velocities, particle, velocity);
        }

        __global__ void predict_jvp_kernel(const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> non_pressure_accelerations, const ConstVectorView<float> pressure_accelerations, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ConstVectorView<float> non_pressure_acceleration_tangent, const ConstVectorView<float> pressure_acceleration_tangent, const VectorView<float> predicted_position_tangent, const VectorView<float> predicted_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(add(load(non_pressure_accelerations, particle), load(pressure_accelerations, particle)), time_step));
            const Float3 predicted_position = add(load(positions, particle), scale(predicted_velocity, time_step));
            Float3 velocity_tangent_value   = add(load(velocity_tangent, particle), scale(add(load(non_pressure_acceleration_tangent, particle), load(pressure_acceleration_tangent, particle)), time_step));
            Float3 position_tangent_value   = add(load(position_tangent, particle), scale(velocity_tangent_value, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, predicted_position, collision_x, collision_y, collision_z);
            if (collision_x) position_tangent_value.x = 0.0F;
            if (collision_y) position_tangent_value.y = 0.0F;
            if (collision_z) position_tangent_value.z = 0.0F;
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_tangent_value = {};
            else {
                if (collision_x) velocity_tangent_value.x = 0.0F;
                if (collision_y) velocity_tangent_value.y = 0.0F;
                if (collision_z) velocity_tangent_value.z = 0.0F;
            }
            store(predicted_position_tangent, particle, position_tangent_value);
            store(predicted_velocity_tangent, particle, velocity_tangent_value);
        }

        __global__ void predict_vjp_kernel(const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> non_pressure_accelerations, const ConstVectorView<float> pressure_accelerations, const ConstVectorView<double> predicted_position_adjoint, const ConstVectorView<double> predicted_velocity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> non_pressure_acceleration_adjoint, const VectorView<double> pressure_acceleration_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(add(load(non_pressure_accelerations, particle), load(pressure_accelerations, particle)), time_step));
            const Float3 predicted_position = add(load(positions, particle), scale(predicted_velocity, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, predicted_position, collision_x, collision_y, collision_z);
            double position_x = collision_x ? 0.0 : predicted_position_adjoint.x[particle];
            double position_y = collision_y ? 0.0 : predicted_position_adjoint.y[particle];
            double position_z = collision_z ? 0.0 : predicted_position_adjoint.z[particle];
            double velocity_x = predicted_velocity_adjoint.x[particle] + time_step * position_x;
            double velocity_y = predicted_velocity_adjoint.y[particle] + time_step * position_y;
            double velocity_z = predicted_velocity_adjoint.z[particle] + time_step * position_z;
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_x = velocity_y = velocity_z = 0.0;
            else {
                if (collision_x) velocity_x = 0.0;
                if (collision_y) velocity_y = 0.0;
                if (collision_z) velocity_z = 0.0;
            }
            position_adjoint.x[particle] += position_x;
            position_adjoint.y[particle] += position_y;
            position_adjoint.z[particle] += position_z;
            velocity_adjoint.x[particle] += velocity_x;
            velocity_adjoint.y[particle] += velocity_y;
            velocity_adjoint.z[particle] += velocity_z;
            const double acceleration_x = time_step * velocity_x;
            const double acceleration_y = time_step * velocity_y;
            const double acceleration_z = time_step * velocity_z;
            non_pressure_acceleration_adjoint.x[particle] += acceleration_x;
            non_pressure_acceleration_adjoint.y[particle] += acceleration_y;
            non_pressure_acceleration_adjoint.z[particle] += acceleration_z;
            pressure_acceleration_adjoint.x[particle] += acceleration_x;
            pressure_acceleration_adjoint.y[particle] += acceleration_y;
            pressure_acceleration_adjoint.z[particle] += acceleration_z;
        }

        __global__ void add_kernel(const std::uint32_t particle_count, const ConstVectorView<float> first, const ConstVectorView<float> second, const VectorView<float> output) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            output.x[particle] = first.x[particle] + second.x[particle];
            output.y[particle] = first.y[particle] + second.y[particle];
            output.z[particle] = first.z[particle] + second.z[particle];
        }

        __global__ void add_adjoint_kernel(const std::uint32_t particle_count, const ConstVectorView<double> output_adjoint, const VectorView<double> first_adjoint, const VectorView<double> second_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const double x = output_adjoint.x[particle];
            const double y = output_adjoint.y[particle];
            const double z = output_adjoint.z[particle];
            first_adjoint.x[particle] += x;
            first_adjoint.y[particle] += y;
            first_adjoint.z[particle] += z;
            second_adjoint.x[particle] += x;
            second_adjoint.y[particle] += y;
            second_adjoint.z[particle] += z;
        }

    } // namespace

    void non_pressure_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const float gravity_x, const float gravity_y, const float gravity_z, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> external_accelerations, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const VectorView<float> accelerations) {
        non_pressure_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, {gravity_x, gravity_y, gravity_z}, positions, velocities, external_accelerations, parameters, neighborhood, boundary, densities, accelerations);
    }

    void non_pressure_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> external_acceleration_tangent, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ParticleParameterView parameters, const ParticleParameterTangentView parameter_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* density_tangent, const VectorView<float> acceleration_tangent) {
        non_pressure_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, velocities, external_acceleration_tangent, position_tangent, velocity_tangent, parameters, parameter_tangent, neighborhood, boundary, densities, density_tangent, acceleration_tangent);
    }

    void non_pressure_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const ConstVectorView<double> acceleration_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> control_adjoint, double* density_adjoint, const ParticleParameterAdjointView parameter_adjoint) {
        non_pressure_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, velocities, parameters, neighborhood, boundary, densities, acceleration_adjoint, position_adjoint, velocity_adjoint, control_adjoint, density_adjoint, parameter_adjoint);
    }

    void integrate_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> accelerations, const VectorView<float> next_positions, const VectorView<float> next_velocities) {
        integrate_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, domain, positions, velocities, accelerations, next_positions, next_velocities);
    }

    void integrate_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> accelerations, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ConstVectorView<float> acceleration_tangent, const VectorView<float> next_position_tangent, const VectorView<float> next_velocity_tangent) {
        integrate_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, domain, positions, velocities, accelerations, position_tangent, velocity_tangent, acceleration_tangent, next_position_tangent, next_velocity_tangent);
    }

    void integrate_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> accelerations, const ConstVectorView<double> next_position_adjoint, const ConstVectorView<double> next_velocity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> acceleration_adjoint) {
        integrate_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, domain, positions, velocities, accelerations, next_position_adjoint, next_velocity_adjoint, position_adjoint, velocity_adjoint, acceleration_adjoint);
    }

    void predict_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> non_pressure_accelerations, const ConstVectorView<float> pressure_accelerations, const VectorView<float> predicted_positions, const VectorView<float> predicted_velocities) {
        predict_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, domain, positions, velocities, non_pressure_accelerations, pressure_accelerations, predicted_positions, predicted_velocities);
    }

    void predict_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> non_pressure_accelerations, const ConstVectorView<float> pressure_accelerations, const ConstVectorView<float> position_tangent, const ConstVectorView<float> velocity_tangent, const ConstVectorView<float> non_pressure_acceleration_tangent, const ConstVectorView<float> pressure_acceleration_tangent, const VectorView<float> predicted_position_tangent, const VectorView<float> predicted_velocity_tangent) {
        predict_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, domain, positions, velocities, non_pressure_accelerations, pressure_accelerations, position_tangent, velocity_tangent, non_pressure_acceleration_tangent, pressure_acceleration_tangent, predicted_position_tangent, predicted_velocity_tangent);
    }

    void predict_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const Box domain, const ConstVectorView<float> positions, const ConstVectorView<float> velocities, const ConstVectorView<float> non_pressure_accelerations, const ConstVectorView<float> pressure_accelerations, const ConstVectorView<double> predicted_position_adjoint, const ConstVectorView<double> predicted_velocity_adjoint, const VectorView<double> position_adjoint, const VectorView<double> velocity_adjoint, const VectorView<double> non_pressure_acceleration_adjoint, const VectorView<double> pressure_acceleration_adjoint) {
        predict_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, time_step, domain, positions, velocities, non_pressure_accelerations, pressure_accelerations, predicted_position_adjoint, predicted_velocity_adjoint, position_adjoint, velocity_adjoint, non_pressure_acceleration_adjoint, pressure_acceleration_adjoint);
    }

    void pressure_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* pressures, const VectorView<float> accelerations) {
        pressure_forward_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, parameters, neighborhood, boundary, densities, pressures, accelerations);
    }

    void pressure_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView parameters, const ParticleParameterTangentView parameter_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* density_tangent, const float* pressures, const float* pressure_tangent, const VectorView<float> acceleration_tangent) {
        pressure_jvp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, position_tangent, parameters, parameter_tangent, neighborhood, boundary, densities, density_tangent, pressures, pressure_tangent, acceleration_tangent);
    }

    void pressure_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const ConstVectorView<float> positions, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const float* densities, const float* pressures, const ConstVectorView<double> acceleration_adjoint, const VectorView<double> position_adjoint, double* density_adjoint, double* pressure_adjoint, const ParticleParameterAdjointView parameter_adjoint) {
        pressure_vjp_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, parameters, neighborhood, boundary, densities, pressures, acceleration_adjoint, position_adjoint, density_adjoint, pressure_adjoint, parameter_adjoint);
    }

    void add(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const ConstVectorView<float> first, const ConstVectorView<float> second, const VectorView<float> output) {
        add_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, first, second, output);
    }

    void add_adjoint(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const ConstVectorView<double> output_adjoint, const VectorView<double> first_adjoint, const VectorView<double> second_adjoint) {
        add_adjoint_kernel<<<blocks(particle_count), block_size, 0, stream.get()>>>(particle_count, output_adjoint, first_adjoint, second_adjoint);
    }

} // namespace physica::fluids::liquid::cuda_detail::sph
