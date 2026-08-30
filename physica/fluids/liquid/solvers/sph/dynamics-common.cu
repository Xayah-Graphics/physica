#include "dynamics-kernels.h"
#include <cstdint>
#include <cuda/cmath>
#include <cuda/std/cmath>
#include <cuda_runtime.h>

namespace physica::fluids::liquid::solvers::sph::kernels::common {
    namespace {
        constexpr std::uint32_t block_size = 256u;
        __global__ void non_pressure_forward_kernel(const std::uint32_t particle_count, const float support_radius, const Vector3<float> gravity, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> controls, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const simulation::VectorView<float> accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = load(positions, particle);
            const Vector3<float> velocity = load(velocities, particle);
            Vector3<float> acceleration   = (gravity + load(controls, particle));
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Vector3<float> displacement = (position - load(positions, neighbor));
                            const float viscosity     = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float weight        = viscosity * parameters.masses[neighbor] * device::viscosity_laplacian(displacement, support_radius) / densities[neighbor];
                            acceleration              = (acceleration + ((load(velocities, neighbor) - velocity) * weight));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Vector3<float> displacement    = (position - device::boundary_position(boundary, neighbor));
                            const float weight           = parameters.viscosities[particle] * boundary.volumes[neighbor] * device::viscosity_laplacian(displacement, support_radius);
                            acceleration                 = (acceleration + ((device::boundary_velocity(boundary, neighbor) - velocity) * weight));
                        }
                    }
            store(accelerations, particle, acceleration);
        }

        __global__ void non_pressure_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> control_tangent, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const device::ParticleParameterView parameters, const device::ParticleParameterTangentView parameter_tangent, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const float* density_tangent, const simulation::VectorView<float> acceleration_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position     = load(positions, particle);
            const Vector3<float> velocity     = load(velocities, particle);
            const Vector3<float> position_dot = load(position_tangent, particle);
            const Vector3<float> velocity_dot = load(velocity_tangent, particle);
            Vector3<float> result             = load(control_tangent, particle);
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Vector3<float> displacement     = (position - load(positions, neighbor));
                            const Vector3<float> displacement_dot = (position_dot - load(position_tangent, neighbor));
                            const float laplacian         = device::viscosity_laplacian(displacement, support_radius);
                            const float laplacian_dot     = device::viscosity_laplacian_tangent(displacement, displacement_dot, support_radius);
                            const float viscosity         = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float viscosity_dot     = 0.5F * (parameter_tangent.viscosities[particle] + parameter_tangent.viscosities[neighbor]);
                            const float mass              = parameters.masses[neighbor];
                            const float density           = densities[neighbor];
                            const float weight            = viscosity * mass * laplacian / density;
                            const float weight_dot        = viscosity_dot * mass * laplacian / density + viscosity * parameter_tangent.masses[neighbor] * laplacian / density + viscosity * mass * laplacian_dot / density - viscosity * mass * laplacian * density_tangent[neighbor] / (density * density);
                            const Vector3<float> difference       = (load(velocities, neighbor) - velocity);
                            const Vector3<float> difference_dot   = (load(velocity_tangent, neighbor) - velocity_dot);
                            result                        = (result + ((difference * weight_dot) + (difference_dot * weight)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Vector3<float> displacement    = (position - device::boundary_position(boundary, neighbor));
                            const float laplacian        = device::viscosity_laplacian(displacement, support_radius);
                            const float weight           = parameters.viscosities[particle] * boundary.volumes[neighbor] * laplacian;
                            const float weight_dot       = boundary.volumes[neighbor] * (parameter_tangent.viscosities[particle] * laplacian + parameters.viscosities[particle] * device::viscosity_laplacian_tangent(displacement, position_dot, support_radius));
                            const Vector3<float> difference      = (device::boundary_velocity(boundary, neighbor) - velocity);
                            result                       = (result + ((difference * weight_dot) + (velocity_dot * -weight)));
                        }
                    }
            store(acceleration_tangent, particle, result);
        }

        __global__ void non_pressure_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const simulation::VectorView<const double> acceleration_adjoint, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint, const simulation::VectorView<double> control_adjoint, double* density_adjoint, const device::ParticleParameterAdjointView parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position       = load(positions, particle);
            const Vector3<float> velocity       = load(velocities, particle);
            const Vector3<double> local_adjoint = load(acceleration_adjoint, particle);
            Vector3<double> position_contribution{};
            Vector3<double> velocity_contribution{};
            double density_contribution   = 0.0;
            double mass_contribution      = 0.0;
            double viscosity_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Vector3<float> displacement      = (position - load(positions, neighbor));
                            const Vector3<float> difference        = (load(velocities, neighbor) - velocity);
                            const float laplacian          = device::viscosity_laplacian(displacement, support_radius);
                            const float viscosity          = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float mass               = parameters.masses[neighbor];
                            const float density            = densities[neighbor];
                            const float weight             = viscosity * mass * laplacian / density;
                            const double weight_adjoint    = dot(local_adjoint, difference);
                            velocity_contribution          = (velocity_contribution + (local_adjoint * -weight));
                            const double viscosity_adjoint = weight_adjoint * mass * laplacian / density;
                            viscosity_contribution += 0.5 * viscosity_adjoint;
                            position_contribution = (position_contribution + device::viscosity_laplacian_gradient(displacement, support_radius, weight_adjoint * viscosity * mass / density));

                            const Vector3<double> cross_adjoint       = load(acceleration_adjoint, neighbor);
                            const float cross_weight          = viscosity * parameters.masses[particle] * laplacian / densities[particle];
                            const double cross_weight_adjoint = dot(cross_adjoint, (difference * -1.0F));
                            velocity_contribution             = (velocity_contribution + (cross_adjoint * cross_weight));
                            mass_contribution += cross_weight_adjoint * viscosity * laplacian / densities[particle];
                            density_contribution -= cross_weight_adjoint * viscosity * parameters.masses[particle] * laplacian / (densities[particle] * densities[particle]);
                            viscosity_contribution += 0.5 * cross_weight_adjoint * parameters.masses[particle] * laplacian / densities[particle];
                            position_contribution = (position_contribution + device::viscosity_laplacian_gradient(displacement, support_radius, cross_weight_adjoint * viscosity * parameters.masses[particle] / densities[particle]));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Vector3<float> displacement    = (position - device::boundary_position(boundary, neighbor));
                            const Vector3<float> difference      = (device::boundary_velocity(boundary, neighbor) - velocity);
                            const float laplacian        = device::viscosity_laplacian(displacement, support_radius);
                            const float weight           = parameters.viscosities[particle] * boundary.volumes[neighbor] * laplacian;
                            const double weight_adjoint  = dot(local_adjoint, difference);
                            velocity_contribution        = (velocity_contribution + (local_adjoint * -weight));
                            viscosity_contribution += weight_adjoint * boundary.volumes[neighbor] * laplacian;
                            position_contribution = (position_contribution + device::viscosity_laplacian_gradient(displacement, support_radius, weight_adjoint * parameters.viscosities[particle] * boundary.volumes[neighbor]));
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            accumulate(velocity_adjoint, particle, velocity_contribution);
            accumulate(control_adjoint, particle, local_adjoint);
            density_adjoint[particle] += density_contribution;
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.viscosities[particle] += viscosity_contribution;
        }

        __global__ void pressure_forward_kernel(const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const float* pressures, const simulation::VectorView<float> accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = load(positions, particle);
            Vector3<float> acceleration{};
            const float first_term = pressures[particle] / (densities[particle] * densities[particle]);
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const float second_term = pressures[neighbor] / (densities[neighbor] * densities[neighbor]);
                            acceleration            = (acceleration + (device::cubic_gradient((position - load(positions, neighbor)), support_radius) * -parameters.masses[neighbor] * (first_term + second_term)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            acceleration                 = (acceleration + (device::cubic_gradient((position - device::boundary_position(boundary, neighbor)), support_radius) * -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term));
                        }
                    }
            store(accelerations, particle, acceleration);
        }

        __global__ void pressure_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> position_tangent, const device::ParticleParameterView parameters, const device::ParticleParameterTangentView parameter_tangent, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const float* density_tangent, const float* pressures, const float* pressure_tangent, const simulation::VectorView<float> acceleration_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position      = load(positions, particle);
            const Vector3<float> position_dot  = load(position_tangent, particle);
            const float density        = densities[particle];
            const float first_term     = pressures[particle] / (density * density);
            const float first_term_dot = pressure_tangent[particle] / (density * density) - 2.0F * pressures[particle] * density_tangent[particle] / (density * density * density);
            Vector3<float> result{};
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const float neighbor_density  = densities[neighbor];
                            const float second_term       = pressures[neighbor] / (neighbor_density * neighbor_density);
                            const float second_term_dot   = pressure_tangent[neighbor] / (neighbor_density * neighbor_density) - 2.0F * pressures[neighbor] * density_tangent[neighbor] / (neighbor_density * neighbor_density * neighbor_density);
                            const Vector3<float> displacement     = (position - load(positions, neighbor));
                            const Vector3<float> displacement_dot = (position_dot - load(position_tangent, neighbor));
                            const float factor            = -parameters.masses[neighbor] * (first_term + second_term);
                            const float factor_dot        = -parameter_tangent.masses[neighbor] * (first_term + second_term) - parameters.masses[neighbor] * (first_term_dot + second_term_dot);
                            result                        = (result + ((device::cubic_gradient(displacement, support_radius) * factor_dot) + (device::cubic_gradient_tangent(displacement, displacement_dot, support_radius) * factor)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Vector3<float> displacement    = (position - device::boundary_position(boundary, neighbor));
                            const float factor           = -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term;
                            const float factor_dot       = -boundary.volumes[neighbor] * (parameter_tangent.rest_densities[particle] * first_term + parameters.rest_densities[particle] * first_term_dot);
                            result                       = (result + ((device::cubic_gradient(displacement, support_radius) * factor_dot) + (device::cubic_gradient_tangent(displacement, position_dot, support_radius) * factor)));
                        }
                    }
            store(acceleration_tangent, particle, result);
        }

        __global__ void pressure_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const float* pressures, const simulation::VectorView<const double> acceleration_adjoint, const simulation::VectorView<double> position_adjoint, double* density_adjoint, double* pressure_adjoint, const device::ParticleParameterAdjointView parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position       = load(positions, particle);
            const Vector3<double> local_adjoint = load(acceleration_adjoint, particle);
            const double first_term     = static_cast<double>(pressures[particle]) / (densities[particle] * densities[particle]);
            Vector3<double> position_contribution{};
            double first_term_adjoint        = 0.0;
            double mass_contribution         = 0.0;
            double rest_density_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last  = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Vector3<float> displacement         = (position - load(positions, neighbor));
                            const Vector3<float> gradient             = device::cubic_gradient(displacement, support_radius);
                            const double second_term          = static_cast<double>(pressures[neighbor]) / (densities[neighbor] * densities[neighbor]);
                            const double sum                  = first_term + second_term;
                            const double local_factor         = -parameters.masses[neighbor] * sum;
                            const double local_factor_adjoint = dot(local_adjoint, gradient);
                            first_term_adjoint -= parameters.masses[neighbor] * local_factor_adjoint;
                            position_contribution = (position_contribution + device::cubic_hessian_product(displacement, (local_adjoint * local_factor), support_radius));

                            const Vector3<double> cross_adjoint   = load(acceleration_adjoint, neighbor);
                            const double cross_projection = dot(cross_adjoint, gradient);
                            mass_contribution += sum * cross_projection;
                            first_term_adjoint += parameters.masses[particle] * cross_projection;
                            position_contribution = (position_contribution + device::cubic_hessian_product(displacement, (cross_adjoint * parameters.masses[particle] * sum), support_radius));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last  = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Vector3<float> displacement    = (position - device::boundary_position(boundary, neighbor));
                            const Vector3<float> gradient        = device::cubic_gradient(displacement, support_radius);
                            const double factor          = -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term;
                            const double factor_adjoint  = dot(local_adjoint, gradient);
                            rest_density_contribution -= factor_adjoint * boundary.volumes[neighbor] * first_term;
                            first_term_adjoint -= factor_adjoint * parameters.rest_densities[particle] * boundary.volumes[neighbor];
                            position_contribution = (position_contribution + device::cubic_hessian_product(displacement, (local_adjoint * factor), support_radius));
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.rest_densities[particle] += rest_density_contribution;
            pressure_adjoint[particle] += first_term_adjoint / (densities[particle] * densities[particle]);
            density_adjoint[particle] -= 2.0 * first_term_adjoint * pressures[particle] / (densities[particle] * densities[particle] * densities[particle]);
        }

        __device__ void collision_mask(const device::CollisionBox collision_box, const Vector3<float> predicted_position, bool& collision_x, bool& collision_y, bool& collision_z) {
            collision_x = predicted_position.x < collision_box.bounds.minimum.x || predicted_position.x > collision_box.bounds.maximum.x;
            collision_y = predicted_position.y < collision_box.bounds.minimum.y || predicted_position.y > collision_box.bounds.maximum.y;
            collision_z = predicted_position.z < collision_box.bounds.minimum.z || predicted_position.z > collision_box.bounds.maximum.z;
        }

        __global__ void integrate_forward_kernel(const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> accelerations, const simulation::VectorView<float> next_positions, const simulation::VectorView<float> next_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Vector3<float> velocity = (load(velocities, particle) + (load(accelerations, particle) * time_step));
            Vector3<float> position = (load(positions, particle) + (velocity * time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(collision_box, position, collision_x, collision_y, collision_z);
            position.x = ::cuda::std::min(collision_box.bounds.maximum.x, ::cuda::std::max(collision_box.bounds.minimum.x, position.x));
            position.y = ::cuda::std::min(collision_box.bounds.maximum.y, ::cuda::std::max(collision_box.bounds.minimum.y, position.y));
            position.z = ::cuda::std::min(collision_box.bounds.maximum.z, ::cuda::std::max(collision_box.bounds.minimum.z, position.z));
            if (collision_box.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity = {collision_box.velocity.x, collision_box.velocity.y, collision_box.velocity.z};
            else {
                if (collision_x) velocity.x = collision_box.velocity.x;
                if (collision_y) velocity.y = collision_box.velocity.y;
                if (collision_z) velocity.z = collision_box.velocity.z;
            }
            store(next_positions, particle, position);
            store(next_velocities, particle, velocity);
        }

        __global__ void integrate_jvp_kernel(const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> accelerations, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const simulation::VectorView<const float> acceleration_tangent, const simulation::VectorView<float> next_position_tangent, const simulation::VectorView<float> next_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> predicted_velocity = (load(velocities, particle) + (load(accelerations, particle) * time_step));
            const Vector3<float> predicted_position = (load(positions, particle) + (predicted_velocity * time_step));
            Vector3<float> velocity_dot             = (load(velocity_tangent, particle) + (load(acceleration_tangent, particle) * time_step));
            Vector3<float> position_dot             = (load(position_tangent, particle) + (velocity_dot * time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(collision_box, predicted_position, collision_x, collision_y, collision_z);
            if (collision_x) position_dot.x = 0.0F;
            if (collision_y) position_dot.y = 0.0F;
            if (collision_z) position_dot.z = 0.0F;
            if (collision_box.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_dot = {};
            else {
                if (collision_x) velocity_dot.x = 0.0F;
                if (collision_y) velocity_dot.y = 0.0F;
                if (collision_z) velocity_dot.z = 0.0F;
            }
            store(next_position_tangent, particle, position_dot);
            store(next_velocity_tangent, particle, velocity_dot);
        }

        __global__ void integrate_vjp_kernel(const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> accelerations, const simulation::VectorView<const double> next_position_adjoint, const simulation::VectorView<const double> next_velocity_adjoint, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint, const simulation::VectorView<double> acceleration_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> predicted_velocity = (load(velocities, particle) + (load(accelerations, particle) * time_step));
            const Vector3<float> predicted_position = (load(positions, particle) + (predicted_velocity * time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(collision_box, predicted_position, collision_x, collision_y, collision_z);
            double position_bar_x = collision_x ? 0.0 : next_position_adjoint.x[particle];
            double position_bar_y = collision_y ? 0.0 : next_position_adjoint.y[particle];
            double position_bar_z = collision_z ? 0.0 : next_position_adjoint.z[particle];
            double velocity_bar_x = next_velocity_adjoint.x[particle] + time_step * position_bar_x;
            double velocity_bar_y = next_velocity_adjoint.y[particle] + time_step * position_bar_y;
            double velocity_bar_z = next_velocity_adjoint.z[particle] + time_step * position_bar_z;
            if (collision_box.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_bar_x = velocity_bar_y = velocity_bar_z = 0.0;
            else {
                if (collision_x) velocity_bar_x = 0.0;
                if (collision_y) velocity_bar_y = 0.0;
                if (collision_z) velocity_bar_z = 0.0;
            }
            accumulate(position_adjoint, particle, {position_bar_x, position_bar_y, position_bar_z});
            accumulate(velocity_adjoint, particle, {velocity_bar_x, velocity_bar_y, velocity_bar_z});
            accumulate(acceleration_adjoint, particle, {time_step * velocity_bar_x, time_step * velocity_bar_y, time_step * velocity_bar_z});
        }

        __global__ void predict_forward_kernel(const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> non_pressure_accelerations, const simulation::VectorView<const float> pressure_accelerations, const simulation::VectorView<float> predicted_positions, const simulation::VectorView<float> predicted_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Vector3<float> velocity = (load(velocities, particle) + ((load(non_pressure_accelerations, particle) + load(pressure_accelerations, particle)) * time_step));
            Vector3<float> position = (load(positions, particle) + (velocity * time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(collision_box, position, collision_x, collision_y, collision_z);
            position.x = ::cuda::std::min(collision_box.bounds.maximum.x, ::cuda::std::max(collision_box.bounds.minimum.x, position.x));
            position.y = ::cuda::std::min(collision_box.bounds.maximum.y, ::cuda::std::max(collision_box.bounds.minimum.y, position.y));
            position.z = ::cuda::std::min(collision_box.bounds.maximum.z, ::cuda::std::max(collision_box.bounds.minimum.z, position.z));
            if (collision_box.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity = {collision_box.velocity.x, collision_box.velocity.y, collision_box.velocity.z};
            else {
                if (collision_x) velocity.x = collision_box.velocity.x;
                if (collision_y) velocity.y = collision_box.velocity.y;
                if (collision_z) velocity.z = collision_box.velocity.z;
            }
            store(predicted_positions, particle, position);
            store(predicted_velocities, particle, velocity);
        }

        __global__ void predict_jvp_kernel(const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> non_pressure_accelerations, const simulation::VectorView<const float> pressure_accelerations, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const simulation::VectorView<const float> non_pressure_acceleration_tangent, const simulation::VectorView<const float> pressure_acceleration_tangent, const simulation::VectorView<float> predicted_position_tangent, const simulation::VectorView<float> predicted_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> predicted_velocity = (load(velocities, particle) + ((load(non_pressure_accelerations, particle) + load(pressure_accelerations, particle)) * time_step));
            const Vector3<float> predicted_position = (load(positions, particle) + (predicted_velocity * time_step));
            Vector3<float> velocity_tangent_value   = (load(velocity_tangent, particle) + ((load(non_pressure_acceleration_tangent, particle) + load(pressure_acceleration_tangent, particle)) * time_step));
            Vector3<float> position_tangent_value   = (load(position_tangent, particle) + (velocity_tangent_value * time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(collision_box, predicted_position, collision_x, collision_y, collision_z);
            if (collision_x) position_tangent_value.x = 0.0F;
            if (collision_y) position_tangent_value.y = 0.0F;
            if (collision_z) position_tangent_value.z = 0.0F;
            if (collision_box.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_tangent_value = {};
            else {
                if (collision_x) velocity_tangent_value.x = 0.0F;
                if (collision_y) velocity_tangent_value.y = 0.0F;
                if (collision_z) velocity_tangent_value.z = 0.0F;
            }
            store(predicted_position_tangent, particle, position_tangent_value);
            store(predicted_velocity_tangent, particle, velocity_tangent_value);
        }

        __global__ void predict_vjp_kernel(const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> non_pressure_accelerations, const simulation::VectorView<const float> pressure_accelerations, const simulation::VectorView<const double> predicted_position_adjoint, const simulation::VectorView<const double> predicted_velocity_adjoint, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint, const simulation::VectorView<double> non_pressure_acceleration_adjoint, const simulation::VectorView<double> pressure_acceleration_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> predicted_velocity = (load(velocities, particle) + ((load(non_pressure_accelerations, particle) + load(pressure_accelerations, particle)) * time_step));
            const Vector3<float> predicted_position = (load(positions, particle) + (predicted_velocity * time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(collision_box, predicted_position, collision_x, collision_y, collision_z);
            double position_x = collision_x ? 0.0 : predicted_position_adjoint.x[particle];
            double position_y = collision_y ? 0.0 : predicted_position_adjoint.y[particle];
            double position_z = collision_z ? 0.0 : predicted_position_adjoint.z[particle];
            double velocity_x = predicted_velocity_adjoint.x[particle] + time_step * position_x;
            double velocity_y = predicted_velocity_adjoint.y[particle] + time_step * position_y;
            double velocity_z = predicted_velocity_adjoint.z[particle] + time_step * position_z;
            if (collision_box.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_x = velocity_y = velocity_z = 0.0;
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

        __global__ void add_kernel(const std::uint32_t particle_count, const simulation::VectorView<const float> first, const simulation::VectorView<const float> second, const simulation::VectorView<float> output) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            output.x[particle] = first.x[particle] + second.x[particle];
            output.y[particle] = first.y[particle] + second.y[particle];
            output.z[particle] = first.z[particle] + second.z[particle];
        }

        __global__ void add_adjoint_kernel(const std::uint32_t particle_count, const simulation::VectorView<const double> output_adjoint, const simulation::VectorView<double> first_adjoint, const simulation::VectorView<double> second_adjoint) {
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

    void non_pressure_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const float gravity_x, const float gravity_y, const float gravity_z, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> external_accelerations, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const simulation::VectorView<float> accelerations) {
        non_pressure_forward_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, support_radius, {gravity_x, gravity_y, gravity_z}, positions, velocities, external_accelerations, parameters, neighborhood, boundary, densities, accelerations);
    }

    void non_pressure_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> external_acceleration_tangent, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const device::ParticleParameterView parameters, const device::ParticleParameterTangentView parameter_tangent, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const float* density_tangent, const simulation::VectorView<float> acceleration_tangent) {
        non_pressure_jvp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, velocities, external_acceleration_tangent, position_tangent, velocity_tangent, parameters, parameter_tangent, neighborhood, boundary, densities, density_tangent, acceleration_tangent);
    }

    void non_pressure_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const simulation::VectorView<const double> acceleration_adjoint, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint, const simulation::VectorView<double> control_adjoint, double* density_adjoint, const device::ParticleParameterAdjointView parameter_adjoint) {
        non_pressure_vjp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, velocities, parameters, neighborhood, boundary, densities, acceleration_adjoint, position_adjoint, velocity_adjoint, control_adjoint, density_adjoint, parameter_adjoint);
    }

    void integrate_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> accelerations, const simulation::VectorView<float> next_positions, const simulation::VectorView<float> next_velocities) {
        integrate_forward_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, collision_box, positions, velocities, accelerations, next_positions, next_velocities);
    }

    void integrate_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> accelerations, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const simulation::VectorView<const float> acceleration_tangent, const simulation::VectorView<float> next_position_tangent, const simulation::VectorView<float> next_velocity_tangent) {
        integrate_jvp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, collision_box, positions, velocities, accelerations, position_tangent, velocity_tangent, acceleration_tangent, next_position_tangent, next_velocity_tangent);
    }

    void integrate_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> accelerations, const simulation::VectorView<const double> next_position_adjoint, const simulation::VectorView<const double> next_velocity_adjoint, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint, const simulation::VectorView<double> acceleration_adjoint) {
        integrate_vjp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, collision_box, positions, velocities, accelerations, next_position_adjoint, next_velocity_adjoint, position_adjoint, velocity_adjoint, acceleration_adjoint);
    }

    void predict_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> non_pressure_accelerations, const simulation::VectorView<const float> pressure_accelerations, const simulation::VectorView<float> predicted_positions, const simulation::VectorView<float> predicted_velocities) {
        predict_forward_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, collision_box, positions, velocities, non_pressure_accelerations, pressure_accelerations, predicted_positions, predicted_velocities);
    }

    void predict_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> non_pressure_accelerations, const simulation::VectorView<const float> pressure_accelerations, const simulation::VectorView<const float> position_tangent, const simulation::VectorView<const float> velocity_tangent, const simulation::VectorView<const float> non_pressure_acceleration_tangent, const simulation::VectorView<const float> pressure_acceleration_tangent, const simulation::VectorView<float> predicted_position_tangent, const simulation::VectorView<float> predicted_velocity_tangent) {
        predict_jvp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, collision_box, positions, velocities, non_pressure_accelerations, pressure_accelerations, position_tangent, velocity_tangent, non_pressure_acceleration_tangent, pressure_acceleration_tangent, predicted_position_tangent, predicted_velocity_tangent);
    }

    void predict_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const device::CollisionBox collision_box, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<const float> non_pressure_accelerations, const simulation::VectorView<const float> pressure_accelerations, const simulation::VectorView<const double> predicted_position_adjoint, const simulation::VectorView<const double> predicted_velocity_adjoint, const simulation::VectorView<double> position_adjoint, const simulation::VectorView<double> velocity_adjoint, const simulation::VectorView<double> non_pressure_acceleration_adjoint, const simulation::VectorView<double> pressure_acceleration_adjoint) {
        predict_vjp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, collision_box, positions, velocities, non_pressure_accelerations, pressure_accelerations, predicted_position_adjoint, predicted_velocity_adjoint, position_adjoint, velocity_adjoint, non_pressure_acceleration_adjoint, pressure_acceleration_adjoint);
    }

    void pressure_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const float* pressures, const simulation::VectorView<float> accelerations) {
        pressure_forward_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, parameters, neighborhood, boundary, densities, pressures, accelerations);
    }

    void pressure_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> position_tangent, const device::ParticleParameterView parameters, const device::ParticleParameterTangentView parameter_tangent, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const float* density_tangent, const float* pressures, const float* pressure_tangent, const simulation::VectorView<float> acceleration_tangent) {
        pressure_jvp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, position_tangent, parameters, parameter_tangent, neighborhood, boundary, densities, density_tangent, pressures, pressure_tangent, acceleration_tangent);
    }

    void pressure_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const float* densities, const float* pressures, const simulation::VectorView<const double> acceleration_adjoint, const simulation::VectorView<double> position_adjoint, double* density_adjoint, double* pressure_adjoint, const device::ParticleParameterAdjointView parameter_adjoint) {
        pressure_vjp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, support_radius, positions, parameters, neighborhood, boundary, densities, pressures, acceleration_adjoint, position_adjoint, density_adjoint, pressure_adjoint, parameter_adjoint);
    }

    void add(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const simulation::VectorView<const float> first, const simulation::VectorView<const float> second, const simulation::VectorView<float> output) {
        add_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, first, second, output);
    }

    void add_adjoint(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const simulation::VectorView<const double> output_adjoint, const simulation::VectorView<double> first_adjoint, const simulation::VectorView<double> second_adjoint) {
        add_adjoint_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, output_adjoint, first_adjoint, second_adjoint);
    }

} // namespace physica::fluids::liquid::solvers::sph::kernels::common
