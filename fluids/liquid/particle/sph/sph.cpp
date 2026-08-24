module;

#include "../domain/interop.h"
#include "../neighborhood/interop.h"
#include "dfsph.h"
#include "iisph.h"
#include "kernels.h"
#include "pcisph.h"
#include "wcsph.h"
#include <physica/cuda.h>

module physica.fluids.liquid.particle.sph;

import std;

namespace physica::fluids::liquid::particle {
    namespace {
        cuda_detail::Box collision_box(const DomainConfiguration& configuration, const std::uint64_t step_index) {
            const float time = static_cast<float>(step_index) * configuration.time_step;
            return {
                .minimum_x = configuration.boundary.minimum.x + time * configuration.boundary.velocity.x + configuration.particle_radius,
                .minimum_y = configuration.boundary.minimum.y + time * configuration.boundary.velocity.y + configuration.particle_radius,
                .minimum_z = configuration.boundary.minimum.z + time * configuration.boundary.velocity.z + configuration.particle_radius,
                .maximum_x = configuration.boundary.maximum.x + time * configuration.boundary.velocity.x - configuration.particle_radius,
                .maximum_y = configuration.boundary.maximum.y + time * configuration.boundary.velocity.y - configuration.particle_radius,
                .maximum_z = configuration.boundary.maximum.z + time * configuration.boundary.velocity.z - configuration.particle_radius,
                .velocity_x = configuration.boundary.velocity.x,
                .velocity_y = configuration.boundary.velocity.y,
                .velocity_z = configuration.boundary.velocity.z,
                .no_slip = configuration.boundary.no_slip ? 1u : 0u,
            };
        }

        float compute_reference_gradient_norm(const DomainConfiguration& configuration) {
            constexpr float pi = 3.14159265358979323846F;
            const float support_radius = configuration.support_radius;
            const float diameter = 2.0F * configuration.particle_radius;
            const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
            Vector3 gradient_sum{};
            float squared_gradient_sum = 0.0F;
            for (float x = -support_radius; x <= support_radius; x += diameter)
                for (float y = -support_radius; y <= support_radius; y += diameter)
                    for (float z = -support_radius; z <= support_radius; z += diameter) {
                        const float distance = std::sqrt(x * x + y * y + z * z);
                        if (distance == 0.0F || distance >= support_radius) continue;
                        const float q = distance / support_radius;
                        const float derivative = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
                        const float scale = coefficient * derivative / (support_radius * distance);
                        gradient_sum.x += scale * x;
                        gradient_sum.y += scale * y;
                        gradient_sum.z += scale * z;
                        squared_gradient_sum += scale * scale * (x * x + y * y + z * z);
                    }
            return gradient_sum.x * gradient_sum.x + gradient_sum.y * gradient_sum.y + gradient_sum.z * gradient_sum.z + squared_gradient_sum;
        }

        PressureIterationCache allocate_iteration_cache(const Domain& domain) {
            return {
                .iteration = 0u,
                .pressures = domain.allocate_scalar_field(domain.configuration.particle_count),
                .predicted_densities = domain.allocate_scalar_field(domain.configuration.particle_count),
                .pressure_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
                .predicted_positions = domain.allocate_vector_field(domain.configuration.particle_count),
                .predicted_velocities = domain.allocate_vector_field(domain.configuration.particle_count),
            };
        }

        PressureIterationTangent allocate_iteration_tangent(const Domain& domain) {
            return {
                .pressures = domain.allocate_scalar_field(domain.configuration.particle_count),
                .predicted_densities = domain.allocate_scalar_field(domain.configuration.particle_count),
                .pressure_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
                .predicted_positions = domain.allocate_vector_field(domain.configuration.particle_count),
                .predicted_velocities = domain.allocate_vector_field(domain.configuration.particle_count),
            };
        }

        PressureIterationAdjoint allocate_iteration_adjoint(const Domain& domain) {
            return {
                .pressures = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count),
                .predicted_densities = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count),
                .pressure_accelerations = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .predicted_positions = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .predicted_velocities = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
            };
        }

        void clear_iteration(const Domain& domain, PressureIterationCache& cache) {
            domain.clear(cache.pressures);
            domain.clear(cache.predicted_densities);
            domain.clear(cache.pressure_accelerations);
            domain.clear(cache.predicted_positions);
            domain.clear(cache.predicted_velocities);
        }

        void clear_iteration(const Domain& domain, PressureIterationTangent& tangent) {
            domain.clear(tangent.pressures);
            domain.clear(tangent.predicted_densities);
            domain.clear(tangent.pressure_accelerations);
            domain.clear(tangent.predicted_positions);
            domain.clear(tangent.predicted_velocities);
        }

        void clear_iteration(const Domain& domain, PressureIterationAdjoint& adjoint) {
            domain.clear(adjoint.pressures);
            domain.clear(adjoint.predicted_densities);
            domain.clear(adjoint.pressure_accelerations);
            domain.clear(adjoint.predicted_positions);
            domain.clear(adjoint.predicted_velocities);
        }

        void copy_iteration(const Domain& domain, const PressureIterationCache& source, PressureIterationCache& destination) {
            destination.iteration = source.iteration;
            domain.copy(source.pressures, destination.pressures);
            domain.copy(source.predicted_densities, destination.predicted_densities);
            domain.copy(source.pressure_accelerations, destination.pressure_accelerations);
            domain.copy(source.predicted_positions, destination.predicted_positions);
            domain.copy(source.predicted_velocities, destination.predicted_velocities);
        }

        template<class Buffer>
        Buffer allocate_buffer(const Domain& domain) {
            return Buffer{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), domain.configuration.particle_count, ::cuda::no_init};
        }

        void non_pressure_forward(const Domain& domain, const ParticleState& state, const Control& control, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, VectorField& accelerations) {
            cuda_detail::sph::non_pressure_forward(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, domain.configuration.gravity.x, domain.configuration.gravity.y, domain.configuration.gravity.z, cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(control.external_accelerations), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), cuda_detail::vector(accelerations));
        }

        void non_pressure_jvp(const Domain& domain, const ParticleState& state, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ParticleStateTangent& state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& parameter_tangent, const ScalarField& density_tangent, VectorField& acceleration_tangent) {
            cuda_detail::sph::non_pressure_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(control_tangent.external_accelerations), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::parameters(parameters), cuda_detail::parameter_tangent(parameter_tangent), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), density_tangent.values.data(), cuda_detail::vector(acceleration_tangent));
        }

        void non_pressure_vjp(const Domain& domain, const ParticleState& state, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const VectorAdjointField& acceleration_adjoint, ParticleStateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, ScalarAdjointField& density_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
            cuda_detail::sph::non_pressure_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), cuda_detail::adjoint_vector(acceleration_adjoint), cuda_detail::adjoint_vector(state_adjoint.positions), cuda_detail::adjoint_vector(state_adjoint.velocities), cuda_detail::adjoint_vector(control_adjoint.external_accelerations), density_adjoint.values.data(), cuda_detail::parameter_adjoint(parameter_adjoint));
        }

        void pressure_forward(const Domain& domain, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, VectorField& accelerations) {
            cuda_detail::sph::pressure_forward(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), pressures.values.data(), cuda_detail::vector(accelerations));
        }

        void pressure_jvp(const Domain& domain, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, const VectorField& position_tangent, const ParticleParameterTangent& parameter_tangent, const ScalarField& density_tangent, const ScalarField& pressure_tangent, VectorField& acceleration_tangent) {
            cuda_detail::sph::pressure_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(positions), cuda_detail::vector(position_tangent), cuda_detail::parameters(parameters), cuda_detail::parameter_tangent(parameter_tangent), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), density_tangent.values.data(), pressures.values.data(), pressure_tangent.values.data(), cuda_detail::vector(acceleration_tangent));
        }

        void pressure_vjp(const Domain& domain, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, const VectorAdjointField& acceleration_adjoint, VectorAdjointField& position_adjoint, ScalarAdjointField& density_adjoint, ScalarAdjointField& pressure_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
            cuda_detail::sph::pressure_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), pressures.values.data(), cuda_detail::adjoint_vector(acceleration_adjoint), cuda_detail::adjoint_vector(position_adjoint), density_adjoint.values.data(), pressure_adjoint.values.data(), cuda_detail::parameter_adjoint(parameter_adjoint));
        }

        void add(const Domain& domain, const VectorField& first, const VectorField& second, VectorField& output) {
            cuda_detail::sph::add(domain.stream, domain.configuration.particle_count, cuda_detail::vector(first), cuda_detail::vector(second), cuda_detail::vector(output));
        }

        void add_adjoint(const Domain& domain, const VectorAdjointField& output, VectorAdjointField& first, VectorAdjointField& second) {
            cuda_detail::sph::add_adjoint(domain.stream, domain.configuration.particle_count, cuda_detail::adjoint_vector(output), cuda_detail::adjoint_vector(first), cuda_detail::adjoint_vector(second));
        }
    } // namespace

    WCSPH::WCSPH(const Domain& domain, Configuration next_configuration, const ExecutionMode mode) : configuration(std::move(next_configuration)) {
        if (mode == ExecutionMode::differentiable) differentiation.emplace(Differentiation{
            .pressure_tangent = domain.allocate_scalar_field(domain.configuration.particle_count),
            .pressure_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
            .viscosity_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
            .surface_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
            .total_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
            .pressure_adjoint = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count),
            .total_acceleration_adjoint = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
        });
    }

    WCSPH::State WCSPH::allocate_state(const Domain&) const { return {}; }
    WCSPH::StateTangent WCSPH::allocate_state_tangent(const Domain&) const { return {}; }
    WCSPH::StateAdjoint WCSPH::allocate_state_adjoint(const Domain&) const { return {}; }

    WCSPH::Parameters WCSPH::allocate_parameters(const Domain& domain) const {
        return {.speed_of_sound = allocate_buffer<::cuda::device_buffer<float>>(domain), .tait_exponent = allocate_buffer<::cuda::device_buffer<float>>(domain), .boundary_surface_tension = allocate_buffer<::cuda::device_buffer<float>>(domain)};
    }

    WCSPH::ParameterTangent WCSPH::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent tangent{.speed_of_sound = allocate_buffer<::cuda::device_buffer<float>>(domain), .tait_exponent = allocate_buffer<::cuda::device_buffer<float>>(domain), .boundary_surface_tension = allocate_buffer<::cuda::device_buffer<float>>(domain)};
        ::cuda::fill_bytes(domain.stream, tangent.speed_of_sound, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.tait_exponent, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.boundary_surface_tension, 0u);
        return tangent;
    }

    WCSPH::ParameterAdjoint WCSPH::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint adjoint{.speed_of_sound = allocate_buffer<::cuda::device_buffer<double>>(domain), .tait_exponent = allocate_buffer<::cuda::device_buffer<double>>(domain), .boundary_surface_tension = allocate_buffer<::cuda::device_buffer<double>>(domain)};
        ::cuda::fill_bytes(domain.stream, adjoint.speed_of_sound, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.tait_exponent, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.boundary_surface_tension, 0u);
        return adjoint;
    }

    WCSPH::Cache WCSPH::allocate_cache(const Domain& domain) const {
        return {
            .pressures = domain.allocate_scalar_field(domain.configuration.particle_count),
            .pressure_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
            .viscosity_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
            .surface_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
            .external_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
            .total_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
        };
    }

    void WCSPH::copy_state(const Domain&, const State&, State&) const {}
    void WCSPH::copy_state_tangent(const Domain&, const StateTangent&, StateTangent&) const {}
    void WCSPH::copy_state_adjoint(const Domain&, const StateAdjoint&, StateAdjoint&) const {}
    void WCSPH::accumulate_state_adjoint(const Domain&, const StateAdjoint&, StateAdjoint&) const {}

    void WCSPH::forward(const Domain& domain, const ParticleState& state, const State&, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, ParticleState& next_state, State&, Cache& cache) {
        cuda_detail::wcsph::launch_eos_forward(domain.stream, domain.configuration.particle_count, densities.values.data(), cuda_detail::parameters(particles), parameters.speed_of_sound.data(), parameters.tait_exponent.data(), cache.pressures.values.data());
        pressure_forward(domain, state.positions, particles, neighborhood, densities, cache.pressures, cache.pressure_accelerations);
        cuda_detail::wcsph::launch_artificial_viscosity_forward(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::parameters(particles), parameters.speed_of_sound.data(), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), cuda_detail::vector(cache.viscosity_accelerations));
        cuda_detail::wcsph::launch_surface_forward(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, domain.configuration.particle_radius, cuda_detail::vector(state.positions), cuda_detail::parameters(particles), parameters.boundary_surface_tension.data(), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), cuda_detail::vector(cache.surface_accelerations));
        add(domain, cache.pressure_accelerations, cache.viscosity_accelerations, cache.total_accelerations);
        add(domain, cache.total_accelerations, cache.surface_accelerations, cache.total_accelerations);
        cuda_detail::wcsph::launch_external_forward(domain.stream, domain.configuration.particle_count, domain.configuration.gravity.x, domain.configuration.gravity.y, domain.configuration.gravity.z, cuda_detail::vector(control.external_accelerations), cuda_detail::vector(cache.external_accelerations));
        add(domain, cache.total_accelerations, cache.external_accelerations, cache.total_accelerations);
        cuda_detail::sph::integrate_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.total_accelerations), cuda_detail::vector(next_state.positions), cuda_detail::vector(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void WCSPH::jvp(const Domain& domain, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent&, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent&) {
        Differentiation& workspace = *differentiation;
        cuda_detail::wcsph::launch_eos_jvp(domain.stream, domain.configuration.particle_count, densities.values.data(), density_tangent.values.data(), cuda_detail::parameters(particles), cuda_detail::parameter_tangent(particle_tangent), parameters.speed_of_sound.data(), parameter_tangent.speed_of_sound.data(), parameters.tait_exponent.data(), parameter_tangent.tait_exponent.data(), workspace.pressure_tangent.values.data());
        pressure_jvp(domain, state.positions, particles, neighborhood, densities, cache.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.pressure_tangent, workspace.pressure_acceleration_tangent);
        cuda_detail::wcsph::launch_artificial_viscosity_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::parameters(particles), cuda_detail::parameter_tangent(particle_tangent), parameters.speed_of_sound.data(), parameter_tangent.speed_of_sound.data(), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), density_tangent.values.data(), cuda_detail::vector(workspace.viscosity_acceleration_tangent));
        cuda_detail::wcsph::launch_surface_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, domain.configuration.particle_radius, cuda_detail::vector(state.positions), cuda_detail::vector(state_tangent.positions), cuda_detail::parameters(particles), cuda_detail::parameter_tangent(particle_tangent), parameters.boundary_surface_tension.data(), parameter_tangent.boundary_surface_tension.data(), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), cuda_detail::vector(workspace.surface_acceleration_tangent));
        add(domain, workspace.pressure_acceleration_tangent, workspace.viscosity_acceleration_tangent, workspace.total_acceleration_tangent);
        add(domain, workspace.total_acceleration_tangent, workspace.surface_acceleration_tangent, workspace.total_acceleration_tangent);
        add(domain, workspace.total_acceleration_tangent, control_tangent.external_accelerations, workspace.total_acceleration_tangent);
        cuda_detail::sph::integrate_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.total_accelerations), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(workspace.total_acceleration_tangent), cuda_detail::vector(next_state_tangent.positions), cuda_detail::vector(next_state_tangent.velocities));
    }

    void WCSPH::vjp(const Domain& domain, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint&, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint&, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarAdjointField& density_adjoint) {
        Differentiation& workspace = *differentiation;
        domain.clear(workspace.pressure_adjoint);
        domain.clear(workspace.total_acceleration_adjoint);
        cuda_detail::sph::integrate_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.total_accelerations), cuda_detail::adjoint_vector(next_state_adjoint.positions), cuda_detail::adjoint_vector(next_state_adjoint.velocities), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), cuda_detail::adjoint_vector(workspace.total_acceleration_adjoint));
        domain.accumulate(workspace.total_acceleration_adjoint, control_adjoint.external_accelerations);
        cuda_detail::wcsph::launch_surface_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, domain.configuration.particle_radius, cuda_detail::vector(state.positions), cuda_detail::parameters(particles), parameters.boundary_surface_tension.data(), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), cuda_detail::adjoint_vector(workspace.total_acceleration_adjoint), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::parameter_adjoint(particle_adjoint), parameter_adjoint.boundary_surface_tension.data());
        cuda_detail::wcsph::launch_artificial_viscosity_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::parameters(particles), parameters.speed_of_sound.data(), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data(), cuda_detail::adjoint_vector(workspace.total_acceleration_adjoint), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), density_adjoint.values.data(), cuda_detail::parameter_adjoint(particle_adjoint), parameter_adjoint.speed_of_sound.data());
        pressure_vjp(domain, state.positions, particles, neighborhood, densities, cache.pressures, workspace.total_acceleration_adjoint, previous_state_adjoint.positions, density_adjoint, workspace.pressure_adjoint, particle_adjoint);
        cuda_detail::wcsph::launch_eos_vjp(domain.stream, domain.configuration.particle_count, densities.values.data(), cuda_detail::parameters(particles), parameters.speed_of_sound.data(), parameters.tait_exponent.data(), workspace.pressure_adjoint.values.data(), density_adjoint.values.data(), cuda_detail::parameter_adjoint(particle_adjoint), parameter_adjoint.speed_of_sound.data(), parameter_adjoint.tait_exponent.data());
    }

    PCISPH::PCISPH(const Domain& domain, Configuration next_configuration, const ExecutionMode mode)
        : configuration(std::move(next_configuration)), reference_gradient_norm(compute_reference_gradient_norm(domain.configuration)), primal(allocate_iteration_cache(domain)) {
        if (mode == ExecutionMode::differentiable) {
            recomputed_iterations.reserve(configuration.checkpoint_interval + 1u);
            for (std::uint32_t iteration = 0u; iteration <= configuration.checkpoint_interval; ++iteration) recomputed_iterations.push_back(allocate_iteration_cache(domain));
            differentiation.emplace(Differentiation{
                .tangent = allocate_iteration_tangent(domain),
                .adjoint = allocate_iteration_adjoint(domain),
                .previous_adjoint = allocate_iteration_adjoint(domain),
                .non_pressure_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
                .non_pressure_acceleration_adjoint = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
            });
        }
    }

    PCISPH::State PCISPH::allocate_state(const Domain&) const { return {}; }
    PCISPH::StateTangent PCISPH::allocate_state_tangent(const Domain&) const { return {}; }
    PCISPH::StateAdjoint PCISPH::allocate_state_adjoint(const Domain&) const { return {}; }
    PCISPH::Parameters PCISPH::allocate_parameters(const Domain& domain) const { return {.pressure_relaxation = allocate_buffer<::cuda::device_buffer<float>>(domain)}; }

    PCISPH::ParameterTangent PCISPH::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent tangent{.pressure_relaxation = allocate_buffer<::cuda::device_buffer<float>>(domain)};
        ::cuda::fill_bytes(domain.stream, tangent.pressure_relaxation, 0u);
        return tangent;
    }

    PCISPH::ParameterAdjoint PCISPH::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint adjoint{.pressure_relaxation = allocate_buffer<::cuda::device_buffer<double>>(domain)};
        ::cuda::fill_bytes(domain.stream, adjoint.pressure_relaxation, 0u);
        return adjoint;
    }

    PCISPH::Cache PCISPH::allocate_cache(const Domain& domain) const {
        Cache cache{.non_pressure_accelerations = domain.allocate_vector_field(domain.configuration.particle_count)};
        const std::uint32_t count = 1u + configuration.pressure_iterations / configuration.checkpoint_interval + (configuration.pressure_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.checkpoints.reserve(count);
        for (std::uint32_t checkpoint = 0u; checkpoint < count; ++checkpoint) cache.checkpoints.push_back(allocate_iteration_cache(domain));
        return cache;
    }

    void PCISPH::copy_state(const Domain&, const State&, State&) const {}
    void PCISPH::copy_state_tangent(const Domain&, const StateTangent&, StateTangent&) const {}
    void PCISPH::copy_state_adjoint(const Domain&, const StateAdjoint&, StateAdjoint&) const {}
    void PCISPH::accumulate_state_adjoint(const Domain&, const StateAdjoint&, StateAdjoint&) const {}

    void PCISPH::forward(const Domain& domain, const ParticleState& state, const State&, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, ParticleState& next_state, State&, Cache& cache) {
        non_pressure_forward(domain, state, control, particles, neighborhood, densities, cache.non_pressure_accelerations);
        clear_iteration(domain, primal);
        primal.iteration = 0u;
        copy_iteration(domain, primal, cache.checkpoints[0]);
        std::uint32_t checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            cuda_detail::pcisph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(primal.predicted_positions), cuda_detail::vector(primal.predicted_velocities));
            density::sph_forward(domain, state.positions, primal.predicted_positions, particles, neighborhood, primal.predicted_densities);
            cuda_detail::pcisph::launch_pressure_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.pressure_relaxation.data(), primal.pressures.values.data());
            pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
            primal.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.pressure_iterations) copy_iteration(domain, primal, cache.checkpoints[checkpoint++]);
        }
        cuda_detail::pcisph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(next_state.positions), cuda_detail::vector(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void PCISPH::jvp(const Domain& domain, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent&, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent&) {
        Differentiation& workspace = *differentiation;
        non_pressure_jvp(domain, state, particles, neighborhood, densities, state_tangent, control_tangent, particle_tangent, density_tangent, workspace.non_pressure_acceleration_tangent);
        clear_iteration(domain, primal);
        clear_iteration(domain, workspace.tangent);
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            cuda_detail::pcisph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(primal.predicted_positions), cuda_detail::vector(primal.predicted_velocities));
            cuda_detail::pcisph::launch_predict_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(workspace.non_pressure_acceleration_tangent), cuda_detail::vector(workspace.tangent.pressure_accelerations), cuda_detail::vector(workspace.tangent.predicted_positions), cuda_detail::vector(workspace.tangent.predicted_velocities));
            density::sph_forward(domain, state.positions, primal.predicted_positions, particles, neighborhood, primal.predicted_densities);
            density::sph_jvp(domain, state.positions, primal.predicted_positions, workspace.tangent.predicted_positions, particles, particle_tangent, neighborhood, workspace.tangent.predicted_densities);
            cuda_detail::pcisph::launch_pressure_update_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), cuda_detail::parameter_tangent(particle_tangent), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.pressure_relaxation.data(), workspace.tangent.pressures.values.data(), workspace.tangent.predicted_densities.values.data(), parameter_tangent.pressure_relaxation.data(), workspace.tangent.pressures.values.data());
            cuda_detail::pcisph::launch_pressure_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.pressure_relaxation.data(), primal.pressures.values.data());
            pressure_jvp(domain, state.positions, particles, neighborhood, densities, primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
            pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
        }
        cuda_detail::pcisph::launch_predict_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(workspace.non_pressure_acceleration_tangent), cuda_detail::vector(workspace.tangent.pressure_accelerations), cuda_detail::vector(next_state_tangent.positions), cuda_detail::vector(next_state_tangent.velocities));
    }

    void PCISPH::vjp(const Domain& domain, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint&, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint&, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarAdjointField& density_adjoint) {
        Differentiation& workspace = *differentiation;
        domain.clear(workspace.non_pressure_acceleration_adjoint);
        clear_iteration(domain, workspace.adjoint);
        clear_iteration(domain, workspace.previous_adjoint);
        cuda_detail::pcisph::launch_predict_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(cache.checkpoints.back().pressure_accelerations), cuda_detail::adjoint_vector(next_state_adjoint.positions), cuda_detail::adjoint_vector(next_state_adjoint.velocities), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), cuda_detail::adjoint_vector(workspace.non_pressure_acceleration_adjoint), cuda_detail::adjoint_vector(workspace.adjoint.pressure_accelerations));
        for (std::size_t checkpoint = cache.checkpoints.size() - 1uz; checkpoint > 0uz; --checkpoint) {
            const PressureIterationCache& first = cache.checkpoints[checkpoint - 1uz];
            const PressureIterationCache& last = cache.checkpoints[checkpoint];
            copy_iteration(domain, first, recomputed_iterations[0]);
            for (std::uint32_t iteration = first.iteration + 1u; iteration <= last.iteration; ++iteration) {
                PressureIterationCache& previous = recomputed_iterations[iteration - first.iteration - 1u];
                PressureIterationCache& current = recomputed_iterations[iteration - first.iteration];
                current.iteration = iteration;
                cuda_detail::pcisph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(previous.pressure_accelerations), cuda_detail::vector(current.predicted_positions), cuda_detail::vector(current.predicted_velocities));
                density::sph_forward(domain, state.positions, current.predicted_positions, particles, neighborhood, current.predicted_densities);
                cuda_detail::pcisph::launch_pressure_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), previous.pressures.values.data(), current.predicted_densities.values.data(), parameters.pressure_relaxation.data(), current.pressures.values.data());
                pressure_forward(domain, state.positions, particles, neighborhood, densities, current.pressures, current.pressure_accelerations);
            }
            for (std::uint32_t iteration = last.iteration; iteration > first.iteration; --iteration) {
                PressureIterationCache& previous = recomputed_iterations[iteration - first.iteration - 1u];
                PressureIterationCache& current = recomputed_iterations[iteration - first.iteration];
                clear_iteration(domain, workspace.previous_adjoint);
                pressure_vjp(domain, state.positions, particles, neighborhood, densities, current.pressures, workspace.adjoint.pressure_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.adjoint.pressures, particle_adjoint);
                cuda_detail::pcisph::launch_pressure_update_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), previous.pressures.values.data(), current.predicted_densities.values.data(), parameters.pressure_relaxation.data(), workspace.adjoint.pressures.values.data(), cuda_detail::parameter_adjoint(particle_adjoint), workspace.previous_adjoint.pressures.values.data(), workspace.adjoint.predicted_densities.values.data(), parameter_adjoint.pressure_relaxation.data());
                density::sph_vjp(domain, state.positions, current.predicted_positions, particles, neighborhood, workspace.adjoint.predicted_densities, workspace.adjoint.predicted_positions, particle_adjoint);
                cuda_detail::pcisph::launch_predict_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(previous.pressure_accelerations), cuda_detail::adjoint_vector(workspace.adjoint.predicted_positions), cuda_detail::adjoint_vector(workspace.adjoint.predicted_velocities), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), cuda_detail::adjoint_vector(workspace.non_pressure_acceleration_adjoint), cuda_detail::adjoint_vector(workspace.previous_adjoint.pressure_accelerations));
                std::swap(workspace.adjoint, workspace.previous_adjoint);
            }
        }
        non_pressure_vjp(domain, state, particles, neighborhood, densities, workspace.non_pressure_acceleration_adjoint, previous_state_adjoint, control_adjoint, density_adjoint, particle_adjoint);
    }
    IISPH::IISPH(const Domain& domain, Configuration next_configuration, const ExecutionMode mode)
        : configuration(std::move(next_configuration)), reference_gradient_norm(compute_reference_gradient_norm(domain.configuration)), primal(allocate_iteration_cache(domain)) {
        if (mode == ExecutionMode::differentiable) {
            recomputed_iterations.reserve(configuration.checkpoint_interval + 1u);
            for (std::uint32_t iteration = 0u; iteration <= configuration.checkpoint_interval; ++iteration) recomputed_iterations.push_back(allocate_iteration_cache(domain));
            differentiation.emplace(Differentiation{
                .tangent = allocate_iteration_tangent(domain),
                .adjoint = allocate_iteration_adjoint(domain),
                .previous_adjoint = allocate_iteration_adjoint(domain),
                .non_pressure_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
                .non_pressure_acceleration_adjoint = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
            });
        }
    }

    IISPH::State IISPH::allocate_state(const Domain&) const { return {}; }
    IISPH::StateTangent IISPH::allocate_state_tangent(const Domain&) const { return {}; }
    IISPH::StateAdjoint IISPH::allocate_state_adjoint(const Domain&) const { return {}; }
    IISPH::Parameters IISPH::allocate_parameters(const Domain& domain) const { return {.jacobi_relaxation = allocate_buffer<::cuda::device_buffer<float>>(domain)}; }

    IISPH::ParameterTangent IISPH::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent tangent{.jacobi_relaxation = allocate_buffer<::cuda::device_buffer<float>>(domain)};
        ::cuda::fill_bytes(domain.stream, tangent.jacobi_relaxation, 0u);
        return tangent;
    }

    IISPH::ParameterAdjoint IISPH::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint adjoint{.jacobi_relaxation = allocate_buffer<::cuda::device_buffer<double>>(domain)};
        ::cuda::fill_bytes(domain.stream, adjoint.jacobi_relaxation, 0u);
        return adjoint;
    }

    IISPH::Cache IISPH::allocate_cache(const Domain& domain) const {
        Cache cache{.non_pressure_accelerations = domain.allocate_vector_field(domain.configuration.particle_count)};
        const std::uint32_t count = 1u + configuration.pressure_iterations / configuration.checkpoint_interval + (configuration.pressure_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.checkpoints.reserve(count);
        for (std::uint32_t checkpoint = 0u; checkpoint < count; ++checkpoint) cache.checkpoints.push_back(allocate_iteration_cache(domain));
        return cache;
    }

    void IISPH::copy_state(const Domain&, const State&, State&) const {}
    void IISPH::copy_state_tangent(const Domain&, const StateTangent&, StateTangent&) const {}
    void IISPH::copy_state_adjoint(const Domain&, const StateAdjoint&, StateAdjoint&) const {}
    void IISPH::accumulate_state_adjoint(const Domain&, const StateAdjoint&, StateAdjoint&) const {}

    void IISPH::forward(const Domain& domain, const ParticleState& state, const State&, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, ParticleState& next_state, State&, Cache& cache) {
        non_pressure_forward(domain, state, control, particles, neighborhood, densities, cache.non_pressure_accelerations);
        clear_iteration(domain, primal);
        primal.iteration = 0u;
        copy_iteration(domain, primal, cache.checkpoints[0]);
        std::uint32_t checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            cuda_detail::iisph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(primal.predicted_positions), cuda_detail::vector(primal.predicted_velocities));
            density::sph_forward(domain, state.positions, primal.predicted_positions, particles, neighborhood, primal.predicted_densities);
            cuda_detail::iisph::launch_jacobi_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), primal.pressures.values.data());
            pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
            primal.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.pressure_iterations) copy_iteration(domain, primal, cache.checkpoints[checkpoint++]);
        }
        cuda_detail::iisph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(next_state.positions), cuda_detail::vector(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void IISPH::jvp(const Domain& domain, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent&, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent&) {
        Differentiation& workspace = *differentiation;
        non_pressure_jvp(domain, state, particles, neighborhood, densities, state_tangent, control_tangent, particle_tangent, density_tangent, workspace.non_pressure_acceleration_tangent);
        clear_iteration(domain, primal);
        clear_iteration(domain, workspace.tangent);
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            cuda_detail::iisph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(primal.predicted_positions), cuda_detail::vector(primal.predicted_velocities));
            cuda_detail::iisph::launch_predict_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(workspace.non_pressure_acceleration_tangent), cuda_detail::vector(workspace.tangent.pressure_accelerations), cuda_detail::vector(workspace.tangent.predicted_positions), cuda_detail::vector(workspace.tangent.predicted_velocities));
            density::sph_forward(domain, state.positions, primal.predicted_positions, particles, neighborhood, primal.predicted_densities);
            density::sph_jvp(domain, state.positions, primal.predicted_positions, workspace.tangent.predicted_positions, particles, particle_tangent, neighborhood, workspace.tangent.predicted_densities);
            cuda_detail::iisph::launch_jacobi_update_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), cuda_detail::parameter_tangent(particle_tangent), densities.values.data(), density_tangent.values.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), workspace.tangent.pressures.values.data(), workspace.tangent.predicted_densities.values.data(), parameter_tangent.jacobi_relaxation.data(), workspace.tangent.pressures.values.data());
            cuda_detail::iisph::launch_jacobi_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), primal.pressures.values.data());
            pressure_jvp(domain, state.positions, particles, neighborhood, densities, primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
            pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
        }
        cuda_detail::iisph::launch_predict_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(workspace.non_pressure_acceleration_tangent), cuda_detail::vector(workspace.tangent.pressure_accelerations), cuda_detail::vector(next_state_tangent.positions), cuda_detail::vector(next_state_tangent.velocities));
    }

    void IISPH::vjp(const Domain& domain, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint&, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint&, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarAdjointField& density_adjoint) {
        Differentiation& workspace = *differentiation;
        domain.clear(workspace.non_pressure_acceleration_adjoint);
        clear_iteration(domain, workspace.adjoint);
        clear_iteration(domain, workspace.previous_adjoint);
        cuda_detail::iisph::launch_predict_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(cache.checkpoints.back().pressure_accelerations), cuda_detail::adjoint_vector(next_state_adjoint.positions), cuda_detail::adjoint_vector(next_state_adjoint.velocities), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), cuda_detail::adjoint_vector(workspace.non_pressure_acceleration_adjoint), cuda_detail::adjoint_vector(workspace.adjoint.pressure_accelerations));
        for (std::size_t checkpoint = cache.checkpoints.size() - 1uz; checkpoint > 0uz; --checkpoint) {
            const PressureIterationCache& first = cache.checkpoints[checkpoint - 1uz];
            const PressureIterationCache& last = cache.checkpoints[checkpoint];
            copy_iteration(domain, first, recomputed_iterations[0]);
            for (std::uint32_t iteration = first.iteration + 1u; iteration <= last.iteration; ++iteration) {
                PressureIterationCache& previous = recomputed_iterations[iteration - first.iteration - 1u];
                PressureIterationCache& current = recomputed_iterations[iteration - first.iteration];
                current.iteration = iteration;
                cuda_detail::iisph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(previous.pressure_accelerations), cuda_detail::vector(current.predicted_positions), cuda_detail::vector(current.predicted_velocities));
                density::sph_forward(domain, state.positions, current.predicted_positions, particles, neighborhood, current.predicted_densities);
                cuda_detail::iisph::launch_jacobi_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), previous.pressures.values.data(), current.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), current.pressures.values.data());
                pressure_forward(domain, state.positions, particles, neighborhood, densities, current.pressures, current.pressure_accelerations);
            }
            for (std::uint32_t iteration = last.iteration; iteration > first.iteration; --iteration) {
                PressureIterationCache& previous = recomputed_iterations[iteration - first.iteration - 1u];
                PressureIterationCache& current = recomputed_iterations[iteration - first.iteration];
                clear_iteration(domain, workspace.previous_adjoint);
                pressure_vjp(domain, state.positions, particles, neighborhood, densities, current.pressures, workspace.adjoint.pressure_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.adjoint.pressures, particle_adjoint);
                cuda_detail::iisph::launch_jacobi_update_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), previous.pressures.values.data(), current.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), workspace.adjoint.pressures.values.data(), cuda_detail::parameter_adjoint(particle_adjoint), density_adjoint.values.data(), workspace.previous_adjoint.pressures.values.data(), workspace.adjoint.predicted_densities.values.data(), parameter_adjoint.jacobi_relaxation.data());
                density::sph_vjp(domain, state.positions, current.predicted_positions, particles, neighborhood, workspace.adjoint.predicted_densities, workspace.adjoint.predicted_positions, particle_adjoint);
                cuda_detail::iisph::launch_predict_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(previous.pressure_accelerations), cuda_detail::adjoint_vector(workspace.adjoint.predicted_positions), cuda_detail::adjoint_vector(workspace.adjoint.predicted_velocities), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), cuda_detail::adjoint_vector(workspace.non_pressure_acceleration_adjoint), cuda_detail::adjoint_vector(workspace.previous_adjoint.pressure_accelerations));
                std::swap(workspace.adjoint, workspace.previous_adjoint);
            }
        }
        non_pressure_vjp(domain, state, particles, neighborhood, densities, workspace.non_pressure_acceleration_adjoint, previous_state_adjoint, control_adjoint, density_adjoint, particle_adjoint);
    }

    DFSPH::DFSPH(const Domain& domain, Configuration next_configuration, const ExecutionMode mode)
        : configuration(std::move(next_configuration)), reference_gradient_norm(compute_reference_gradient_norm(domain.configuration)), primal(allocate_iteration_cache(domain)), total_pressure_acceleration(domain.allocate_vector_field(domain.configuration.particle_count)) {
        if (mode == ExecutionMode::differentiable) {
            recomputed_iterations.reserve(configuration.checkpoint_interval + 1u);
            for (std::uint32_t iteration = 0u; iteration <= configuration.checkpoint_interval; ++iteration) recomputed_iterations.push_back(allocate_iteration_cache(domain));
            differentiation.emplace(Differentiation{
                .tangent = allocate_iteration_tangent(domain),
                .adjoint = allocate_iteration_adjoint(domain),
                .previous_adjoint = allocate_iteration_adjoint(domain),
                .non_pressure_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
                .divergence_pressure_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
                .total_pressure_acceleration_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
                .target_density_adjoint = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count),
                .non_pressure_acceleration_adjoint = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .divergence_pressure_acceleration_adjoint = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .total_pressure_acceleration_adjoint = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
            });
        }
    }

    DFSPH::State DFSPH::allocate_state(const Domain& domain) const {
        State state{.warm_divergence_pressure = domain.allocate_scalar_field(domain.configuration.particle_count), .warm_density_pressure = domain.allocate_scalar_field(domain.configuration.particle_count)};
        domain.clear(state.warm_divergence_pressure);
        domain.clear(state.warm_density_pressure);
        return state;
    }

    DFSPH::StateTangent DFSPH::allocate_state_tangent(const Domain& domain) const {
        StateTangent tangent{.warm_divergence_pressure = domain.allocate_scalar_field(domain.configuration.particle_count), .warm_density_pressure = domain.allocate_scalar_field(domain.configuration.particle_count)};
        domain.clear(tangent.warm_divergence_pressure);
        domain.clear(tangent.warm_density_pressure);
        return tangent;
    }

    DFSPH::StateAdjoint DFSPH::allocate_state_adjoint(const Domain& domain) const {
        StateAdjoint adjoint{.warm_divergence_pressure = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count), .warm_density_pressure = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count)};
        domain.clear(adjoint.warm_divergence_pressure);
        domain.clear(adjoint.warm_density_pressure);
        return adjoint;
    }

    DFSPH::Parameters DFSPH::allocate_parameters(const Domain& domain) const {
        return {.divergence_relaxation = allocate_buffer<::cuda::device_buffer<float>>(domain), .density_relaxation = allocate_buffer<::cuda::device_buffer<float>>(domain)};
    }

    DFSPH::ParameterTangent DFSPH::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent tangent{.divergence_relaxation = allocate_buffer<::cuda::device_buffer<float>>(domain), .density_relaxation = allocate_buffer<::cuda::device_buffer<float>>(domain)};
        ::cuda::fill_bytes(domain.stream, tangent.divergence_relaxation, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.density_relaxation, 0u);
        return tangent;
    }

    DFSPH::ParameterAdjoint DFSPH::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint adjoint{.divergence_relaxation = allocate_buffer<::cuda::device_buffer<double>>(domain), .density_relaxation = allocate_buffer<::cuda::device_buffer<double>>(domain)};
        ::cuda::fill_bytes(domain.stream, adjoint.divergence_relaxation, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.density_relaxation, 0u);
        return adjoint;
    }

    DFSPH::Cache DFSPH::allocate_cache(const Domain& domain) const {
        Cache cache{
            .non_pressure_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
            .divergence_pressure_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
            .total_pressure_accelerations = domain.allocate_vector_field(domain.configuration.particle_count),
        };
        const std::uint32_t divergence_count = 1u + configuration.divergence_iterations / configuration.checkpoint_interval + (configuration.divergence_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.divergence_checkpoints.reserve(divergence_count);
        for (std::uint32_t checkpoint = 0u; checkpoint < divergence_count; ++checkpoint) cache.divergence_checkpoints.push_back(allocate_iteration_cache(domain));
        const std::uint32_t density_count = 1u + configuration.density_iterations / configuration.checkpoint_interval + (configuration.density_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.density_checkpoints.reserve(density_count);
        for (std::uint32_t checkpoint = 0u; checkpoint < density_count; ++checkpoint) cache.density_checkpoints.push_back(allocate_iteration_cache(domain));
        return cache;
    }

    void DFSPH::copy_state(const Domain& domain, const State& source, State& destination) const {
        domain.copy(source.warm_divergence_pressure, destination.warm_divergence_pressure);
        domain.copy(source.warm_density_pressure, destination.warm_density_pressure);
    }

    void DFSPH::copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const {
        domain.copy(source.warm_divergence_pressure, destination.warm_divergence_pressure);
        domain.copy(source.warm_density_pressure, destination.warm_density_pressure);
    }

    void DFSPH::copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const {
        domain.copy(source.warm_divergence_pressure, destination.warm_divergence_pressure);
        domain.copy(source.warm_density_pressure, destination.warm_density_pressure);
    }

    void DFSPH::accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const {
        domain.accumulate(source.warm_divergence_pressure, destination.warm_divergence_pressure);
        domain.accumulate(source.warm_density_pressure, destination.warm_density_pressure);
    }

    void DFSPH::forward(const Domain& domain, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, ParticleState& next_state, State& next_method_state, Cache& cache) {
        non_pressure_forward(domain, state, control, particles, neighborhood, densities, cache.non_pressure_accelerations);
        clear_iteration(domain, primal);
        if (configuration.pressure_warm_start) domain.copy(method_state.warm_divergence_pressure, primal.pressures);
        pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
        primal.iteration = 0u;
        copy_iteration(domain, primal, cache.divergence_checkpoints[0]);
        std::uint32_t checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.divergence_iterations; ++iteration) {
            cuda_detail::dfsph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(primal.predicted_positions), cuda_detail::vector(primal.predicted_velocities));
            density::sph_forward(domain, state.positions, primal.predicted_positions, particles, neighborhood, primal.predicted_densities);
            cuda_detail::dfsph::launch_projection_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), densities.values.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.divergence_relaxation.data(), primal.pressures.values.data());
            pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
            primal.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.divergence_iterations) copy_iteration(domain, primal, cache.divergence_checkpoints[checkpoint++]);
        }
        domain.copy(primal.pressure_accelerations, cache.divergence_pressure_accelerations);
        domain.copy(primal.pressures, next_method_state.warm_divergence_pressure);

        clear_iteration(domain, primal);
        if (configuration.pressure_warm_start) domain.copy(method_state.warm_density_pressure, primal.pressures);
        pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
        primal.iteration = 0u;
        copy_iteration(domain, primal, cache.density_checkpoints[0]);
        checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.density_iterations; ++iteration) {
            add(domain, cache.divergence_pressure_accelerations, primal.pressure_accelerations, total_pressure_acceleration);
            cuda_detail::dfsph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(total_pressure_acceleration), cuda_detail::vector(primal.predicted_positions), cuda_detail::vector(primal.predicted_velocities));
            density::sph_forward(domain, state.positions, primal.predicted_positions, particles, neighborhood, primal.predicted_densities);
            cuda_detail::dfsph::launch_projection_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), particles.rest_densities.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.density_relaxation.data(), primal.pressures.values.data());
            pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
            primal.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.density_iterations) copy_iteration(domain, primal, cache.density_checkpoints[checkpoint++]);
        }
        add(domain, cache.divergence_pressure_accelerations, primal.pressure_accelerations, cache.total_pressure_accelerations);
        domain.copy(primal.pressures, next_method_state.warm_density_pressure);
        cuda_detail::dfsph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(cache.total_pressure_accelerations), cuda_detail::vector(next_state.positions), cuda_detail::vector(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void DFSPH::jvp(const Domain& domain, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const ScalarField& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent) {
        Differentiation& workspace = *differentiation;
        non_pressure_jvp(domain, state, particles, neighborhood, densities, state_tangent, control_tangent, particle_tangent, density_tangent, workspace.non_pressure_acceleration_tangent);
        clear_iteration(domain, primal);
        clear_iteration(domain, workspace.tangent);
        if (configuration.pressure_warm_start) {
            domain.copy(method_state.warm_divergence_pressure, primal.pressures);
            domain.copy(method_state_tangent.warm_divergence_pressure, workspace.tangent.pressures);
        }
        pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
        pressure_jvp(domain, state.positions, particles, neighborhood, densities, primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
        for (std::uint32_t iteration = 1u; iteration <= configuration.divergence_iterations; ++iteration) {
            cuda_detail::dfsph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(primal.predicted_positions), cuda_detail::vector(primal.predicted_velocities));
            cuda_detail::dfsph::launch_predict_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(primal.pressure_accelerations), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(workspace.non_pressure_acceleration_tangent), cuda_detail::vector(workspace.tangent.pressure_accelerations), cuda_detail::vector(workspace.tangent.predicted_positions), cuda_detail::vector(workspace.tangent.predicted_velocities));
            density::sph_forward(domain, state.positions, primal.predicted_positions, particles, neighborhood, primal.predicted_densities);
            density::sph_jvp(domain, state.positions, primal.predicted_positions, workspace.tangent.predicted_positions, particles, particle_tangent, neighborhood, workspace.tangent.predicted_densities);
            cuda_detail::dfsph::launch_projection_update_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), cuda_detail::parameter_tangent(particle_tangent), densities.values.data(), density_tangent.values.data(), densities.values.data(), density_tangent.values.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.divergence_relaxation.data(), workspace.tangent.pressures.values.data(), workspace.tangent.predicted_densities.values.data(), parameter_tangent.divergence_relaxation.data(), workspace.tangent.pressures.values.data());
            cuda_detail::dfsph::launch_projection_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), densities.values.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.divergence_relaxation.data(), primal.pressures.values.data());
            pressure_jvp(domain, state.positions, particles, neighborhood, densities, primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
            pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
        }
        domain.copy(workspace.tangent.pressure_accelerations, workspace.divergence_pressure_acceleration_tangent);
        domain.copy(workspace.tangent.pressures, next_method_state_tangent.warm_divergence_pressure);

        clear_iteration(domain, primal);
        clear_iteration(domain, workspace.tangent);
        if (configuration.pressure_warm_start) {
            domain.copy(method_state.warm_density_pressure, primal.pressures);
            domain.copy(method_state_tangent.warm_density_pressure, workspace.tangent.pressures);
        }
        pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
        pressure_jvp(domain, state.positions, particles, neighborhood, densities, primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
        for (std::uint32_t iteration = 1u; iteration <= configuration.density_iterations; ++iteration) {
            add(domain, cache.divergence_pressure_accelerations, primal.pressure_accelerations, total_pressure_acceleration);
            add(domain, workspace.divergence_pressure_acceleration_tangent, workspace.tangent.pressure_accelerations, workspace.total_pressure_acceleration_tangent);
            cuda_detail::dfsph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(total_pressure_acceleration), cuda_detail::vector(primal.predicted_positions), cuda_detail::vector(primal.predicted_velocities));
            cuda_detail::dfsph::launch_predict_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(total_pressure_acceleration), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(workspace.non_pressure_acceleration_tangent), cuda_detail::vector(workspace.total_pressure_acceleration_tangent), cuda_detail::vector(workspace.tangent.predicted_positions), cuda_detail::vector(workspace.tangent.predicted_velocities));
            density::sph_forward(domain, state.positions, primal.predicted_positions, particles, neighborhood, primal.predicted_densities);
            density::sph_jvp(domain, state.positions, primal.predicted_positions, workspace.tangent.predicted_positions, particles, particle_tangent, neighborhood, workspace.tangent.predicted_densities);
            cuda_detail::dfsph::launch_projection_update_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), cuda_detail::parameter_tangent(particle_tangent), densities.values.data(), density_tangent.values.data(), particles.rest_densities.data(), particle_tangent.rest_densities.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.density_relaxation.data(), workspace.tangent.pressures.values.data(), workspace.tangent.predicted_densities.values.data(), parameter_tangent.density_relaxation.data(), workspace.tangent.pressures.values.data());
            cuda_detail::dfsph::launch_projection_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), particles.rest_densities.data(), primal.pressures.values.data(), primal.predicted_densities.values.data(), parameters.density_relaxation.data(), primal.pressures.values.data());
            pressure_jvp(domain, state.positions, particles, neighborhood, densities, primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
            pressure_forward(domain, state.positions, particles, neighborhood, densities, primal.pressures, primal.pressure_accelerations);
        }
        add(domain, workspace.divergence_pressure_acceleration_tangent, workspace.tangent.pressure_accelerations, workspace.total_pressure_acceleration_tangent);
        domain.copy(workspace.tangent.pressures, next_method_state_tangent.warm_density_pressure);
        cuda_detail::dfsph::launch_predict_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(cache.total_pressure_accelerations), cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(workspace.non_pressure_acceleration_tangent), cuda_detail::vector(workspace.total_pressure_acceleration_tangent), cuda_detail::vector(next_state_tangent.positions), cuda_detail::vector(next_state_tangent.velocities));
    }

    void DFSPH::vjp(const Domain& domain, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, ScalarAdjointField& density_adjoint) {
        Differentiation& workspace = *differentiation;
        domain.clear(workspace.target_density_adjoint);
        domain.clear(workspace.non_pressure_acceleration_adjoint);
        domain.clear(workspace.divergence_pressure_acceleration_adjoint);
        domain.clear(workspace.total_pressure_acceleration_adjoint);
        clear_iteration(domain, workspace.adjoint);
        clear_iteration(domain, workspace.previous_adjoint);

        cuda_detail::dfsph::launch_predict_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(cache.total_pressure_accelerations), cuda_detail::adjoint_vector(next_state_adjoint.positions), cuda_detail::adjoint_vector(next_state_adjoint.velocities), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), cuda_detail::adjoint_vector(workspace.non_pressure_acceleration_adjoint), cuda_detail::adjoint_vector(workspace.total_pressure_acceleration_adjoint));
        domain.copy(workspace.total_pressure_acceleration_adjoint, workspace.divergence_pressure_acceleration_adjoint);
        domain.copy(workspace.total_pressure_acceleration_adjoint, workspace.adjoint.pressure_accelerations);
        domain.copy(next_method_state_adjoint.warm_density_pressure, workspace.adjoint.pressures);

        const auto reverse_phase = [&](const std::vector<PressureIterationCache>& checkpoints, const ::cuda::device_buffer<float>& relaxation, ::cuda::device_buffer<double>& relaxation_adjoint, const float* target_densities, double* target_density_adjoint, const VectorField* base_pressure_accelerations, VectorAdjointField* base_pressure_acceleration_adjoint, ScalarAdjointField& warm_pressure_adjoint) {
            for (std::size_t checkpoint = checkpoints.size(); checkpoint-- > 1uz;) {
                const PressureIterationCache& first = checkpoints[checkpoint - 1uz];
                const PressureIterationCache& last = checkpoints[checkpoint];
                copy_iteration(domain, first, recomputed_iterations[0]);
                for (std::uint32_t iteration = first.iteration + 1u; iteration <= last.iteration; ++iteration) {
                    PressureIterationCache& previous = recomputed_iterations[iteration - first.iteration - 1u];
                    PressureIterationCache& current = recomputed_iterations[iteration - first.iteration];
                    current.iteration = iteration;
                    const VectorField* prediction_pressure = &previous.pressure_accelerations;
                    if (base_pressure_accelerations != nullptr) {
                        add(domain, *base_pressure_accelerations, previous.pressure_accelerations, total_pressure_acceleration);
                        prediction_pressure = &total_pressure_acceleration;
                    }
                    cuda_detail::dfsph::launch_predict_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(*prediction_pressure), cuda_detail::vector(current.predicted_positions), cuda_detail::vector(current.predicted_velocities));
                    density::sph_forward(domain, state.positions, current.predicted_positions, particles, neighborhood, current.predicted_densities);
                    cuda_detail::dfsph::launch_projection_update_forward(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), target_densities, previous.pressures.values.data(), current.predicted_densities.values.data(), relaxation.data(), current.pressures.values.data());
                    pressure_forward(domain, state.positions, particles, neighborhood, densities, current.pressures, current.pressure_accelerations);
                }
                for (std::uint32_t iteration = last.iteration; iteration > first.iteration; --iteration) {
                    PressureIterationCache& previous = recomputed_iterations[iteration - first.iteration - 1u];
                    PressureIterationCache& current = recomputed_iterations[iteration - first.iteration];
                    clear_iteration(domain, workspace.previous_adjoint);
                    pressure_vjp(domain, state.positions, particles, neighborhood, densities, current.pressures, workspace.adjoint.pressure_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.adjoint.pressures, particle_adjoint);
                    cuda_detail::dfsph::launch_projection_update_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, reference_gradient_norm, cuda_detail::parameters(particles), densities.values.data(), target_densities, previous.pressures.values.data(), current.predicted_densities.values.data(), relaxation.data(), workspace.adjoint.pressures.values.data(), cuda_detail::parameter_adjoint(particle_adjoint), density_adjoint.values.data(), target_density_adjoint, workspace.previous_adjoint.pressures.values.data(), workspace.adjoint.predicted_densities.values.data(), relaxation_adjoint.data());
                    density::sph_vjp(domain, state.positions, current.predicted_positions, particles, neighborhood, workspace.adjoint.predicted_densities, workspace.adjoint.predicted_positions, particle_adjoint);
                    if (base_pressure_accelerations == nullptr) {
                        cuda_detail::dfsph::launch_predict_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(previous.pressure_accelerations), cuda_detail::adjoint_vector(workspace.adjoint.predicted_positions), cuda_detail::adjoint_vector(workspace.adjoint.predicted_velocities), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), cuda_detail::adjoint_vector(workspace.non_pressure_acceleration_adjoint), cuda_detail::adjoint_vector(workspace.previous_adjoint.pressure_accelerations));
                    } else {
                        domain.clear(workspace.total_pressure_acceleration_adjoint);
                        add(domain, *base_pressure_accelerations, previous.pressure_accelerations, total_pressure_acceleration);
                        cuda_detail::dfsph::launch_predict_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.time_step, collision_box(domain.configuration, state.step_index), cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(cache.non_pressure_accelerations), cuda_detail::vector(total_pressure_acceleration), cuda_detail::adjoint_vector(workspace.adjoint.predicted_positions), cuda_detail::adjoint_vector(workspace.adjoint.predicted_velocities), cuda_detail::adjoint_vector(previous_state_adjoint.positions), cuda_detail::adjoint_vector(previous_state_adjoint.velocities), cuda_detail::adjoint_vector(workspace.non_pressure_acceleration_adjoint), cuda_detail::adjoint_vector(workspace.total_pressure_acceleration_adjoint));
                        add_adjoint(domain, workspace.total_pressure_acceleration_adjoint, *base_pressure_acceleration_adjoint, workspace.previous_adjoint.pressure_accelerations);
                    }
                    std::swap(workspace.adjoint, workspace.previous_adjoint);
                }
            }
            pressure_vjp(domain, state.positions, particles, neighborhood, densities, checkpoints.front().pressures, workspace.adjoint.pressure_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.adjoint.pressures, particle_adjoint);
            if (configuration.pressure_warm_start) domain.copy(workspace.adjoint.pressures, warm_pressure_adjoint);
        };

        reverse_phase(cache.density_checkpoints, parameters.density_relaxation, parameter_adjoint.density_relaxation, particles.rest_densities.data(), particle_adjoint.rest_densities.data(), &cache.divergence_pressure_accelerations, &workspace.divergence_pressure_acceleration_adjoint, previous_method_state_adjoint.warm_density_pressure);
        clear_iteration(domain, workspace.adjoint);
        domain.copy(workspace.divergence_pressure_acceleration_adjoint, workspace.adjoint.pressure_accelerations);
        domain.copy(next_method_state_adjoint.warm_divergence_pressure, workspace.adjoint.pressures);
        reverse_phase(cache.divergence_checkpoints, parameters.divergence_relaxation, parameter_adjoint.divergence_relaxation, densities.values.data(), workspace.target_density_adjoint.values.data(), nullptr, nullptr, previous_method_state_adjoint.warm_divergence_pressure);
        domain.accumulate(workspace.target_density_adjoint, density_adjoint);
        non_pressure_vjp(domain, state, particles, neighborhood, densities, workspace.non_pressure_acceleration_adjoint, previous_state_adjoint, control_adjoint, density_adjoint, particle_adjoint);
    }
} // namespace physica::fluids::liquid::particle
