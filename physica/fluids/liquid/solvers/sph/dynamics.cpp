module;

#include "dynamics-kernels.h"
#include <fluids/liquid/interop.h>
#include <physica/cuda.h>

module physica.fluids.liquid.solvers.sph.dynamics;

import std;

namespace physica::fluids::liquid::solvers::sph {
    namespace {
        float compute_reference_gradient_norm(const meshfree::Configuration& configuration) {
            const float support_radius = configuration.support_radius;
            const float diameter       = 2.0F * configuration.particle_radius;
            const float coefficient    = 8.0F / (std::numbers::pi_v<float> * support_radius * support_radius * support_radius);
            Vector3<float> gradient_sum{};
            float squared_gradient_sum = 0.0F;
            for (float x = -support_radius; x <= support_radius; x += diameter)
                for (float y = -support_radius; y <= support_radius; y += diameter)
                    for (float z = -support_radius; z <= support_radius; z += diameter) {
                        const float distance = std::sqrt(x * x + y * y + z * z);
                        if (distance == 0.0F || distance >= support_radius) continue;
                        const float q          = distance / support_radius;
                        const float derivative = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
                        const float scale      = coefficient * derivative / (support_radius * distance);
                        gradient_sum.x += scale * x;
                        gradient_sum.y += scale * y;
                        gradient_sum.z += scale * z;
                        squared_gradient_sum += scale * scale * (x * x + y * y + z * z);
                    }
            return gradient_sum.x * gradient_sum.x + gradient_sum.y * gradient_sum.y + gradient_sum.z * gradient_sum.z + squared_gradient_sum;
        }

        PressureIterationCache allocate_iteration_cache(const meshfree::Model& model) {
            return {
                .iteration              = 0u,
                .pressures              = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
                .predicted_densities    = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
                .pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
                .predicted_positions    = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
                .predicted_velocities   = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            };
        }

        PressureIterationTangent allocate_iteration_tangent(const meshfree::Model& model) {
            return {
                .pressures              = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
                .predicted_densities    = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
                .pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
                .predicted_positions    = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
                .predicted_velocities   = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            };
        }

        PressureIterationAdjoint allocate_iteration_adjoint(const meshfree::Model& model) {
            return {
                .pressures              = simulation::ScalarField<double>(model.stream, model.configuration.particle_count),
                .predicted_densities    = simulation::ScalarField<double>(model.stream, model.configuration.particle_count),
                .pressure_accelerations = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
                .predicted_positions    = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
                .predicted_velocities   = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            };
        }

        void clear_iteration(const meshfree::Model& model, PressureIterationCache& cache) {
            simulation::clear(model.stream, cache.pressures);
            simulation::clear(model.stream, cache.predicted_densities);
            simulation::clear(model.stream, cache.pressure_accelerations);
            simulation::clear(model.stream, cache.predicted_positions);
            simulation::clear(model.stream, cache.predicted_velocities);
        }

        void clear_iteration(const meshfree::Model& model, PressureIterationTangent& tangent) {
            simulation::clear(model.stream, tangent.pressures);
            simulation::clear(model.stream, tangent.predicted_densities);
            simulation::clear(model.stream, tangent.pressure_accelerations);
            simulation::clear(model.stream, tangent.predicted_positions);
            simulation::clear(model.stream, tangent.predicted_velocities);
        }

        void clear_iteration(const meshfree::Model& model, PressureIterationAdjoint& adjoint) {
            simulation::clear(model.stream, adjoint.pressures);
            simulation::clear(model.stream, adjoint.predicted_densities);
            simulation::clear(model.stream, adjoint.pressure_accelerations);
            simulation::clear(model.stream, adjoint.predicted_positions);
            simulation::clear(model.stream, adjoint.predicted_velocities);
        }

        void copy_iteration(const meshfree::Model& model, const PressureIterationCache& source, PressureIterationCache& destination) {
            destination.iteration = source.iteration;
            simulation::copy(model.stream, source.pressures, destination.pressures);
            simulation::copy(model.stream, source.predicted_densities, destination.predicted_densities);
            simulation::copy(model.stream, source.pressure_accelerations, destination.pressure_accelerations);
            simulation::copy(model.stream, source.predicted_positions, destination.predicted_positions);
            simulation::copy(model.stream, source.predicted_velocities, destination.predicted_velocities);
        }

        template <class Buffer>
        Buffer allocate_buffer(const meshfree::Model& model) {
            return Buffer{model.stream, ::cuda::device_default_memory_pool(model.stream.device()), model.configuration.particle_count, ::cuda::no_init};
        }

        void non_pressure_forward(const meshfree::Model& model, const Vector3<float>& gravity, const ParticleState& state, const Control& control, const ParticleParameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, simulation::VectorField<float>& accelerations) {
            kernels::common::non_pressure_forward(model.stream, model.configuration.particle_count, model.configuration.support_radius, gravity.x, gravity.y, gravity.z, simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_accelerations), device::particle_parameters(parameters), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), simulation::view(accelerations));
        }

        void non_pressure_jvp(const meshfree::Model& model, const ParticleState& state, const ParticleParameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const ParticleStateTangent& state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& parameter_tangent, const simulation::ScalarField<float>& density_tangent, simulation::VectorField<float>& acceleration_tangent) {
            kernels::common::non_pressure_jvp(model.stream, model.configuration.particle_count, model.configuration.support_radius, simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control_tangent.external_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), device::particle_parameters(parameters), device::particle_parameter_tangent(parameter_tangent), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), density_tangent.values.data(), simulation::view(acceleration_tangent));
        }

        void non_pressure_vjp(const meshfree::Model& model, const ParticleState& state, const ParticleParameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const simulation::VectorField<double>& acceleration_adjoint, ParticleStateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, simulation::ScalarField<double>& density_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
            kernels::common::non_pressure_vjp(model.stream, model.configuration.particle_count, model.configuration.support_radius, simulation::view(state.positions), simulation::view(state.velocities), device::particle_parameters(parameters), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), simulation::view(acceleration_adjoint), simulation::view(state_adjoint.positions), simulation::view(state_adjoint.velocities), simulation::view(control_adjoint.external_accelerations), density_adjoint.values.data(), device::particle_parameter_adjoint(parameter_adjoint));
        }

        void pressure_forward(const meshfree::Model& model, const simulation::VectorField<float>& positions, const ParticleParameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const simulation::ScalarField<float>& pressures, simulation::VectorField<float>& accelerations) {
            kernels::common::pressure_forward(model.stream, model.configuration.particle_count, model.configuration.support_radius, simulation::view(positions), device::particle_parameters(parameters), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), pressures.values.data(), simulation::view(accelerations));
        }

        void pressure_jvp(const meshfree::Model& model, const simulation::VectorField<float>& positions, const ParticleParameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const simulation::ScalarField<float>& pressures, const simulation::VectorField<float>& position_tangent, const ParticleParameterTangent& parameter_tangent, const simulation::ScalarField<float>& density_tangent, const simulation::ScalarField<float>& pressure_tangent, simulation::VectorField<float>& acceleration_tangent) {
            kernels::common::pressure_jvp(model.stream, model.configuration.particle_count, model.configuration.support_radius, simulation::view(positions), simulation::view(position_tangent), device::particle_parameters(parameters), device::particle_parameter_tangent(parameter_tangent), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), density_tangent.values.data(), pressures.values.data(), pressure_tangent.values.data(), simulation::view(acceleration_tangent));
        }

        void pressure_vjp(const meshfree::Model& model, const simulation::VectorField<float>& positions, const ParticleParameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const simulation::ScalarField<float>& pressures, const simulation::VectorField<double>& acceleration_adjoint, simulation::VectorField<double>& position_adjoint, simulation::ScalarField<double>& density_adjoint, simulation::ScalarField<double>& pressure_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
            kernels::common::pressure_vjp(model.stream, model.configuration.particle_count, model.configuration.support_radius, simulation::view(positions), device::particle_parameters(parameters), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), pressures.values.data(), simulation::view(acceleration_adjoint), simulation::view(position_adjoint), density_adjoint.values.data(), pressure_adjoint.values.data(), device::particle_parameter_adjoint(parameter_adjoint));
        }

        void add(const meshfree::Model& model, const simulation::VectorField<float>& first, const simulation::VectorField<float>& second, simulation::VectorField<float>& output) {
            kernels::common::add(model.stream, model.configuration.particle_count, simulation::view(first), simulation::view(second), simulation::view(output));
        }

        void add_adjoint(const meshfree::Model& model, const simulation::VectorField<double>& output, simulation::VectorField<double>& first, simulation::VectorField<double>& second) {
            kernels::common::add_adjoint(model.stream, model.configuration.particle_count, simulation::view(output), simulation::view(first), simulation::view(second));
        }
    } // namespace

    WeaklyCompressible::WeaklyCompressible(const meshfree::Model&, Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    WeaklyCompressible::State WeaklyCompressible::allocate_state(const meshfree::Model&) const {
        return {};
    }
    WeaklyCompressible::StateTangent WeaklyCompressible::allocate_state_tangent(const meshfree::Model&) const {
        return {};
    }
    WeaklyCompressible::StateAdjoint WeaklyCompressible::allocate_state_adjoint(const meshfree::Model&) const {
        return {};
    }

    WeaklyCompressible::Parameters WeaklyCompressible::allocate_parameters(const meshfree::Model& model) const {
        return {.speed_of_sound = allocate_buffer<::cuda::device_buffer<float>>(model), .tait_exponent = allocate_buffer<::cuda::device_buffer<float>>(model), .boundary_surface_tension = allocate_buffer<::cuda::device_buffer<float>>(model)};
    }

    WeaklyCompressible::ParameterTangent WeaklyCompressible::allocate_parameter_tangent(const meshfree::Model& model) const {
        ParameterTangent tangent{.speed_of_sound = allocate_buffer<::cuda::device_buffer<float>>(model), .tait_exponent = allocate_buffer<::cuda::device_buffer<float>>(model), .boundary_surface_tension = allocate_buffer<::cuda::device_buffer<float>>(model)};
        ::cuda::fill_bytes(model.stream, tangent.speed_of_sound, 0u);
        ::cuda::fill_bytes(model.stream, tangent.tait_exponent, 0u);
        ::cuda::fill_bytes(model.stream, tangent.boundary_surface_tension, 0u);
        return tangent;
    }

    WeaklyCompressible::ParameterAdjoint WeaklyCompressible::allocate_parameter_adjoint(const meshfree::Model& model) const {
        ParameterAdjoint adjoint{.speed_of_sound = allocate_buffer<::cuda::device_buffer<double>>(model), .tait_exponent = allocate_buffer<::cuda::device_buffer<double>>(model), .boundary_surface_tension = allocate_buffer<::cuda::device_buffer<double>>(model)};
        ::cuda::fill_bytes(model.stream, adjoint.speed_of_sound, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.tait_exponent, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.boundary_surface_tension, 0u);
        return adjoint;
    }

    WeaklyCompressible::Cache WeaklyCompressible::allocate_cache(const meshfree::Model& model) const {
        return {
            .pressures               = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .pressure_accelerations  = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .viscosity_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .surface_accelerations   = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .external_accelerations  = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .total_accelerations     = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
    }

    WeaklyCompressible::Workspace WeaklyCompressible::allocate_workspace(const meshfree::Model&) const {
        return {};
    }

    WeaklyCompressible::TangentWorkspace WeaklyCompressible::allocate_tangent_workspace(const meshfree::Model& model) const {
        return {
            .pressures               = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .pressure_accelerations  = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .viscosity_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .surface_accelerations   = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .total_accelerations     = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
    }

    WeaklyCompressible::AdjointWorkspace WeaklyCompressible::allocate_adjoint_workspace(const meshfree::Model& model) const {
        return {
            .pressures           = simulation::ScalarField<double>(model.stream, model.configuration.particle_count),
            .total_accelerations = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
        };
    }

    void WeaklyCompressible::copy_state(const meshfree::Model&, const State&, State&) const {}
    void WeaklyCompressible::copy_state_tangent(const meshfree::Model&, const StateTangent&, StateTangent&) const {}
    void WeaklyCompressible::copy_state_adjoint(const meshfree::Model&, const StateAdjoint&, StateAdjoint&) const {}
    void WeaklyCompressible::accumulate_state_adjoint(const meshfree::Model&, const StateAdjoint&, StateAdjoint&) const {}

    void WeaklyCompressible::forward(const meshfree::Model& model, const ParticleState& state, const State&, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, ParticleState& next_state, State&, Cache& cache, Workspace&) const {
        kernels::wcsph::launch_eos_forward(model.stream, model.configuration.particle_count, densities.values.data(), device::particle_parameters(particles), parameters.speed_of_sound.data(), parameters.tait_exponent.data(), cache.pressures.values.data());
        pressure_forward(model, state.positions, particles, neighborhood, densities, cache.pressures, cache.pressure_accelerations);
        kernels::wcsph::launch_artificial_viscosity_forward(model.stream, model.configuration.particle_count, model.configuration.support_radius, simulation::view(state.positions), simulation::view(state.velocities), device::particle_parameters(particles), parameters.speed_of_sound.data(), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), simulation::view(cache.viscosity_accelerations));
        kernels::wcsph::launch_surface_forward(model.stream, model.configuration.particle_count, model.configuration.support_radius, model.configuration.particle_radius, simulation::view(state.positions), device::particle_parameters(particles), parameters.boundary_surface_tension.data(), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), simulation::view(cache.surface_accelerations));
        add(model, cache.pressure_accelerations, cache.viscosity_accelerations, cache.total_accelerations);
        add(model, cache.total_accelerations, cache.surface_accelerations, cache.total_accelerations);
        kernels::wcsph::launch_external_forward(model.stream, model.configuration.particle_count, configuration.gravity.x, configuration.gravity.y, configuration.gravity.z, simulation::view(control.external_accelerations), simulation::view(cache.external_accelerations));
        add(model, cache.total_accelerations, cache.external_accelerations, cache.total_accelerations);
        kernels::common::integrate_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.total_accelerations), simulation::view(next_state.positions), simulation::view(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void WeaklyCompressible::jvp(const meshfree::Model& model, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent&, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const simulation::ScalarField<float>& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent&, TangentWorkspace& workspace) const {
        kernels::wcsph::launch_eos_jvp(model.stream, model.configuration.particle_count, densities.values.data(), density_tangent.values.data(), device::particle_parameters(particles), device::particle_parameter_tangent(particle_tangent), parameters.speed_of_sound.data(), parameter_tangent.speed_of_sound.data(), parameters.tait_exponent.data(), parameter_tangent.tait_exponent.data(), workspace.pressures.values.data());
        pressure_jvp(model, state.positions, particles, neighborhood, densities, cache.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.pressures, workspace.pressure_accelerations);
        kernels::wcsph::launch_artificial_viscosity_jvp(model.stream, model.configuration.particle_count, model.configuration.support_radius, simulation::view(state.positions), simulation::view(state.velocities), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), device::particle_parameters(particles), device::particle_parameter_tangent(particle_tangent), parameters.speed_of_sound.data(), parameter_tangent.speed_of_sound.data(), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), density_tangent.values.data(), simulation::view(workspace.viscosity_accelerations));
        kernels::wcsph::launch_surface_jvp(model.stream, model.configuration.particle_count, model.configuration.support_radius, model.configuration.particle_radius, simulation::view(state.positions), simulation::view(state_tangent.positions), device::particle_parameters(particles), device::particle_parameter_tangent(particle_tangent), parameters.boundary_surface_tension.data(), parameter_tangent.boundary_surface_tension.data(), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), simulation::view(workspace.surface_accelerations));
        add(model, workspace.pressure_accelerations, workspace.viscosity_accelerations, workspace.total_accelerations);
        add(model, workspace.total_accelerations, workspace.surface_accelerations, workspace.total_accelerations);
        add(model, workspace.total_accelerations, control_tangent.external_accelerations, workspace.total_accelerations);
        kernels::common::integrate_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.total_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(workspace.total_accelerations), simulation::view(next_state_tangent.positions), simulation::view(next_state_tangent.velocities));
    }

    void WeaklyCompressible::vjp(const meshfree::Model& model, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint&, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint&, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, simulation::ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const {
        simulation::clear(model.stream, workspace.pressures);
        simulation::clear(model.stream, workspace.total_accelerations);
        kernels::common::integrate_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.total_accelerations), simulation::view(next_state_adjoint.positions), simulation::view(next_state_adjoint.velocities), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(workspace.total_accelerations));
        simulation::accumulate(model.stream, workspace.total_accelerations, control_adjoint.external_accelerations);
        kernels::wcsph::launch_surface_vjp(model.stream, model.configuration.particle_count, model.configuration.support_radius, model.configuration.particle_radius, simulation::view(state.positions), device::particle_parameters(particles), parameters.boundary_surface_tension.data(), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), simulation::view(workspace.total_accelerations), simulation::view(previous_state_adjoint.positions), device::particle_parameter_adjoint(particle_adjoint), parameter_adjoint.boundary_surface_tension.data());
        kernels::wcsph::launch_artificial_viscosity_vjp(model.stream, model.configuration.particle_count, model.configuration.support_radius, simulation::view(state.positions), simulation::view(state.velocities), device::particle_parameters(particles), parameters.speed_of_sound.data(), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data(), simulation::view(workspace.total_accelerations), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), density_adjoint.values.data(), device::particle_parameter_adjoint(particle_adjoint), parameter_adjoint.speed_of_sound.data());
        pressure_vjp(model, state.positions, particles, neighborhood, densities, cache.pressures, workspace.total_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.pressures, particle_adjoint);
        kernels::wcsph::launch_eos_vjp(model.stream, model.configuration.particle_count, densities.values.data(), device::particle_parameters(particles), parameters.speed_of_sound.data(), parameters.tait_exponent.data(), workspace.pressures.values.data(), density_adjoint.values.data(), device::particle_parameter_adjoint(particle_adjoint), parameter_adjoint.speed_of_sound.data(), parameter_adjoint.tait_exponent.data());
    }

    PredictiveCorrective::PredictiveCorrective(const meshfree::Model& model, Configuration next_configuration) : configuration(std::move(next_configuration)), reference_gradient_norm(compute_reference_gradient_norm(model.configuration)), density({}) {}

    PredictiveCorrective::State PredictiveCorrective::allocate_state(const meshfree::Model&) const {
        return {};
    }
    PredictiveCorrective::StateTangent PredictiveCorrective::allocate_state_tangent(const meshfree::Model&) const {
        return {};
    }
    PredictiveCorrective::StateAdjoint PredictiveCorrective::allocate_state_adjoint(const meshfree::Model&) const {
        return {};
    }
    PredictiveCorrective::Parameters PredictiveCorrective::allocate_parameters(const meshfree::Model& model) const {
        return {.pressure_relaxation = allocate_buffer<::cuda::device_buffer<float>>(model)};
    }

    PredictiveCorrective::ParameterTangent PredictiveCorrective::allocate_parameter_tangent(const meshfree::Model& model) const {
        ParameterTangent tangent{.pressure_relaxation = allocate_buffer<::cuda::device_buffer<float>>(model)};
        ::cuda::fill_bytes(model.stream, tangent.pressure_relaxation, 0u);
        return tangent;
    }

    PredictiveCorrective::ParameterAdjoint PredictiveCorrective::allocate_parameter_adjoint(const meshfree::Model& model) const {
        ParameterAdjoint adjoint{.pressure_relaxation = allocate_buffer<::cuda::device_buffer<double>>(model)};
        ::cuda::fill_bytes(model.stream, adjoint.pressure_relaxation, 0u);
        return adjoint;
    }

    PredictiveCorrective::Cache PredictiveCorrective::allocate_cache(const meshfree::Model& model) const {
        Cache cache{.non_pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count)};
        const std::uint32_t count = 1u + configuration.pressure_iterations / configuration.checkpoint_interval + (configuration.pressure_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.checkpoints.reserve(count);
        for (std::uint32_t checkpoint = 0u; checkpoint < count; ++checkpoint) cache.checkpoints.push_back(allocate_iteration_cache(model));
        return cache;
    }

    PredictiveCorrective::Workspace PredictiveCorrective::allocate_workspace(const meshfree::Model& model) const {
        return {.primal = allocate_iteration_cache(model)};
    }

    PredictiveCorrective::TangentWorkspace PredictiveCorrective::allocate_tangent_workspace(const meshfree::Model& model) const {
        return {
            .primal                     = allocate_iteration_cache(model),
            .tangent                    = allocate_iteration_tangent(model),
            .non_pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
    }

    PredictiveCorrective::AdjointWorkspace PredictiveCorrective::allocate_adjoint_workspace(const meshfree::Model& model) const {
        AdjointWorkspace workspace{
            .adjoint                    = allocate_iteration_adjoint(model),
            .previous_adjoint           = allocate_iteration_adjoint(model),
            .non_pressure_accelerations = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
        };
        workspace.recomputed_iterations.reserve(configuration.checkpoint_interval + 1u);
        for (std::uint32_t iteration = 0u; iteration <= configuration.checkpoint_interval; ++iteration) workspace.recomputed_iterations.push_back(allocate_iteration_cache(model));
        return workspace;
    }
    void PredictiveCorrective::copy_state(const meshfree::Model&, const State&, State&) const {}
    void PredictiveCorrective::copy_state_tangent(const meshfree::Model&, const StateTangent&, StateTangent&) const {}
    void PredictiveCorrective::copy_state_adjoint(const meshfree::Model&, const StateAdjoint&, StateAdjoint&) const {}
    void PredictiveCorrective::accumulate_state_adjoint(const meshfree::Model&, const StateAdjoint&, StateAdjoint&) const {}

    void PredictiveCorrective::forward(const meshfree::Model& model, const ParticleState& state, const State&, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, ParticleState& next_state, State&, Cache& cache, Workspace& workspace) const {
        non_pressure_forward(model, configuration.gravity, state, control, particles, neighborhood, densities, cache.non_pressure_accelerations);
        clear_iteration(model, workspace.primal);
        workspace.primal.iteration = 0u;
        copy_iteration(model, workspace.primal, cache.checkpoints[0]);
        std::uint32_t checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(workspace.primal.predicted_positions), simulation::view(workspace.primal.predicted_velocities));
            density.forward(model, state.positions, workspace.primal.predicted_positions, particles, neighborhood, workspace.primal.predicted_densities);
            kernels::pcisph::launch_pressure_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.pressure_relaxation.data(), workspace.primal.pressures.values.data());
            pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
            workspace.primal.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.pressure_iterations) copy_iteration(model, workspace.primal, cache.checkpoints[checkpoint++]);
        }
        kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(next_state.positions), simulation::view(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void PredictiveCorrective::jvp(const meshfree::Model& model, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent&, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const simulation::ScalarField<float>& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent&, TangentWorkspace& workspace) const {
        non_pressure_jvp(model, state, particles, neighborhood, densities, state_tangent, control_tangent, particle_tangent, density_tangent, workspace.non_pressure_accelerations);
        clear_iteration(model, workspace.primal);
        clear_iteration(model, workspace.tangent);
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(workspace.primal.predicted_positions), simulation::view(workspace.primal.predicted_velocities));
            kernels::common::predict_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.tangent.pressure_accelerations), simulation::view(workspace.tangent.predicted_positions), simulation::view(workspace.tangent.predicted_velocities));
            density.forward(model, state.positions, workspace.primal.predicted_positions, particles, neighborhood, workspace.primal.predicted_densities);
            density.jvp(model, state.positions, workspace.primal.predicted_positions, workspace.tangent.predicted_positions, particles, particle_tangent, neighborhood, workspace.tangent.predicted_densities);
            kernels::pcisph::launch_pressure_update_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), device::particle_parameter_tangent(particle_tangent), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.pressure_relaxation.data(), workspace.tangent.pressures.values.data(), workspace.tangent.predicted_densities.values.data(), parameter_tangent.pressure_relaxation.data(), workspace.tangent.pressures.values.data());
            kernels::pcisph::launch_pressure_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.pressure_relaxation.data(), workspace.primal.pressures.values.data());
            pressure_jvp(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
            pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
        }
        kernels::common::predict_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.tangent.pressure_accelerations), simulation::view(next_state_tangent.positions), simulation::view(next_state_tangent.velocities));
    }

    void PredictiveCorrective::vjp(const meshfree::Model& model, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint&, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint&, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, simulation::ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const {
        simulation::clear(model.stream, workspace.non_pressure_accelerations);
        clear_iteration(model, workspace.adjoint);
        clear_iteration(model, workspace.previous_adjoint);
        kernels::common::predict_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(cache.checkpoints.back().pressure_accelerations), simulation::view(next_state_adjoint.positions), simulation::view(next_state_adjoint.velocities), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.adjoint.pressure_accelerations));
        for (std::size_t checkpoint = cache.checkpoints.size() - 1uz; checkpoint > 0uz; --checkpoint) {
            const PressureIterationCache& first = cache.checkpoints[checkpoint - 1uz];
            const PressureIterationCache& last  = cache.checkpoints[checkpoint];
            copy_iteration(model, first, workspace.recomputed_iterations[0]);
            for (std::uint32_t iteration = first.iteration + 1u; iteration <= last.iteration; ++iteration) {
                PressureIterationCache& previous = workspace.recomputed_iterations[iteration - first.iteration - 1u];
                PressureIterationCache& current  = workspace.recomputed_iterations[iteration - first.iteration];
                current.iteration                = iteration;
                kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(previous.pressure_accelerations), simulation::view(current.predicted_positions), simulation::view(current.predicted_velocities));
                density.forward(model, state.positions, current.predicted_positions, particles, neighborhood, current.predicted_densities);
                kernels::pcisph::launch_pressure_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), previous.pressures.values.data(), current.predicted_densities.values.data(), parameters.pressure_relaxation.data(), current.pressures.values.data());
                pressure_forward(model, state.positions, particles, neighborhood, densities, current.pressures, current.pressure_accelerations);
            }
            for (std::uint32_t iteration = last.iteration; iteration > first.iteration; --iteration) {
                PressureIterationCache& previous = workspace.recomputed_iterations[iteration - first.iteration - 1u];
                PressureIterationCache& current  = workspace.recomputed_iterations[iteration - first.iteration];
                clear_iteration(model, workspace.previous_adjoint);
                pressure_vjp(model, state.positions, particles, neighborhood, densities, current.pressures, workspace.adjoint.pressure_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.adjoint.pressures, particle_adjoint);
                kernels::pcisph::launch_pressure_update_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), previous.pressures.values.data(), current.predicted_densities.values.data(), parameters.pressure_relaxation.data(), workspace.adjoint.pressures.values.data(), device::particle_parameter_adjoint(particle_adjoint), workspace.previous_adjoint.pressures.values.data(), workspace.adjoint.predicted_densities.values.data(), parameter_adjoint.pressure_relaxation.data());
                density.vjp(model, state.positions, current.predicted_positions, particles, neighborhood, workspace.adjoint.predicted_densities, workspace.adjoint.predicted_positions, particle_adjoint);
                kernels::common::predict_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(previous.pressure_accelerations), simulation::view(workspace.adjoint.predicted_positions), simulation::view(workspace.adjoint.predicted_velocities), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.previous_adjoint.pressure_accelerations));
                std::swap(workspace.adjoint, workspace.previous_adjoint);
            }
        }
        non_pressure_vjp(model, state, particles, neighborhood, densities, workspace.non_pressure_accelerations, previous_state_adjoint, control_adjoint, density_adjoint, particle_adjoint);
    }
    ImplicitIncompressible::ImplicitIncompressible(const meshfree::Model& model, Configuration next_configuration) : configuration(std::move(next_configuration)), reference_gradient_norm(compute_reference_gradient_norm(model.configuration)), density({}) {}

    ImplicitIncompressible::State ImplicitIncompressible::allocate_state(const meshfree::Model&) const {
        return {};
    }
    ImplicitIncompressible::StateTangent ImplicitIncompressible::allocate_state_tangent(const meshfree::Model&) const {
        return {};
    }
    ImplicitIncompressible::StateAdjoint ImplicitIncompressible::allocate_state_adjoint(const meshfree::Model&) const {
        return {};
    }
    ImplicitIncompressible::Parameters ImplicitIncompressible::allocate_parameters(const meshfree::Model& model) const {
        return {.jacobi_relaxation = allocate_buffer<::cuda::device_buffer<float>>(model)};
    }

    ImplicitIncompressible::ParameterTangent ImplicitIncompressible::allocate_parameter_tangent(const meshfree::Model& model) const {
        ParameterTangent tangent{.jacobi_relaxation = allocate_buffer<::cuda::device_buffer<float>>(model)};
        ::cuda::fill_bytes(model.stream, tangent.jacobi_relaxation, 0u);
        return tangent;
    }

    ImplicitIncompressible::ParameterAdjoint ImplicitIncompressible::allocate_parameter_adjoint(const meshfree::Model& model) const {
        ParameterAdjoint adjoint{.jacobi_relaxation = allocate_buffer<::cuda::device_buffer<double>>(model)};
        ::cuda::fill_bytes(model.stream, adjoint.jacobi_relaxation, 0u);
        return adjoint;
    }

    ImplicitIncompressible::Cache ImplicitIncompressible::allocate_cache(const meshfree::Model& model) const {
        Cache cache{.non_pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count)};
        const std::uint32_t count = 1u + configuration.pressure_iterations / configuration.checkpoint_interval + (configuration.pressure_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.checkpoints.reserve(count);
        for (std::uint32_t checkpoint = 0u; checkpoint < count; ++checkpoint) cache.checkpoints.push_back(allocate_iteration_cache(model));
        return cache;
    }

    ImplicitIncompressible::Workspace ImplicitIncompressible::allocate_workspace(const meshfree::Model& model) const {
        return {.primal = allocate_iteration_cache(model)};
    }

    ImplicitIncompressible::TangentWorkspace ImplicitIncompressible::allocate_tangent_workspace(const meshfree::Model& model) const {
        return {
            .primal                     = allocate_iteration_cache(model),
            .tangent                    = allocate_iteration_tangent(model),
            .non_pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
    }

    ImplicitIncompressible::AdjointWorkspace ImplicitIncompressible::allocate_adjoint_workspace(const meshfree::Model& model) const {
        AdjointWorkspace workspace{
            .adjoint                    = allocate_iteration_adjoint(model),
            .previous_adjoint           = allocate_iteration_adjoint(model),
            .non_pressure_accelerations = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
        };
        workspace.recomputed_iterations.reserve(configuration.checkpoint_interval + 1u);
        for (std::uint32_t iteration = 0u; iteration <= configuration.checkpoint_interval; ++iteration) workspace.recomputed_iterations.push_back(allocate_iteration_cache(model));
        return workspace;
    }

    void ImplicitIncompressible::copy_state(const meshfree::Model&, const State&, State&) const {}
    void ImplicitIncompressible::copy_state_tangent(const meshfree::Model&, const StateTangent&, StateTangent&) const {}
    void ImplicitIncompressible::copy_state_adjoint(const meshfree::Model&, const StateAdjoint&, StateAdjoint&) const {}
    void ImplicitIncompressible::accumulate_state_adjoint(const meshfree::Model&, const StateAdjoint&, StateAdjoint&) const {}

    void ImplicitIncompressible::forward(const meshfree::Model& model, const ParticleState& state, const State&, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, ParticleState& next_state, State&, Cache& cache, Workspace& workspace) const {
        non_pressure_forward(model, configuration.gravity, state, control, particles, neighborhood, densities, cache.non_pressure_accelerations);
        clear_iteration(model, workspace.primal);
        workspace.primal.iteration = 0u;
        copy_iteration(model, workspace.primal, cache.checkpoints[0]);
        std::uint32_t checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(workspace.primal.predicted_positions), simulation::view(workspace.primal.predicted_velocities));
            density.forward(model, state.positions, workspace.primal.predicted_positions, particles, neighborhood, workspace.primal.predicted_densities);
            kernels::iisph::launch_jacobi_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), workspace.primal.pressures.values.data());
            pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
            workspace.primal.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.pressure_iterations) copy_iteration(model, workspace.primal, cache.checkpoints[checkpoint++]);
        }
        kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(next_state.positions), simulation::view(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void ImplicitIncompressible::jvp(const meshfree::Model& model, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent&, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const simulation::ScalarField<float>& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent&, TangentWorkspace& workspace) const {
        non_pressure_jvp(model, state, particles, neighborhood, densities, state_tangent, control_tangent, particle_tangent, density_tangent, workspace.non_pressure_accelerations);
        clear_iteration(model, workspace.primal);
        clear_iteration(model, workspace.tangent);
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(workspace.primal.predicted_positions), simulation::view(workspace.primal.predicted_velocities));
            kernels::common::predict_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.tangent.pressure_accelerations), simulation::view(workspace.tangent.predicted_positions), simulation::view(workspace.tangent.predicted_velocities));
            density.forward(model, state.positions, workspace.primal.predicted_positions, particles, neighborhood, workspace.primal.predicted_densities);
            density.jvp(model, state.positions, workspace.primal.predicted_positions, workspace.tangent.predicted_positions, particles, particle_tangent, neighborhood, workspace.tangent.predicted_densities);
            kernels::iisph::launch_jacobi_update_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), device::particle_parameter_tangent(particle_tangent), densities.values.data(), density_tangent.values.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), workspace.tangent.pressures.values.data(), workspace.tangent.predicted_densities.values.data(), parameter_tangent.jacobi_relaxation.data(), workspace.tangent.pressures.values.data());
            kernels::iisph::launch_jacobi_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), workspace.primal.pressures.values.data());
            pressure_jvp(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
            pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
        }
        kernels::common::predict_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.tangent.pressure_accelerations), simulation::view(next_state_tangent.positions), simulation::view(next_state_tangent.velocities));
    }

    void ImplicitIncompressible::vjp(const meshfree::Model& model, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint&, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint&, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, simulation::ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const {
        simulation::clear(model.stream, workspace.non_pressure_accelerations);
        clear_iteration(model, workspace.adjoint);
        clear_iteration(model, workspace.previous_adjoint);
        kernels::common::predict_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(cache.checkpoints.back().pressure_accelerations), simulation::view(next_state_adjoint.positions), simulation::view(next_state_adjoint.velocities), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.adjoint.pressure_accelerations));
        for (std::size_t checkpoint = cache.checkpoints.size() - 1uz; checkpoint > 0uz; --checkpoint) {
            const PressureIterationCache& first = cache.checkpoints[checkpoint - 1uz];
            const PressureIterationCache& last  = cache.checkpoints[checkpoint];
            copy_iteration(model, first, workspace.recomputed_iterations[0]);
            for (std::uint32_t iteration = first.iteration + 1u; iteration <= last.iteration; ++iteration) {
                PressureIterationCache& previous = workspace.recomputed_iterations[iteration - first.iteration - 1u];
                PressureIterationCache& current  = workspace.recomputed_iterations[iteration - first.iteration];
                current.iteration                = iteration;
                kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(previous.pressure_accelerations), simulation::view(current.predicted_positions), simulation::view(current.predicted_velocities));
                density.forward(model, state.positions, current.predicted_positions, particles, neighborhood, current.predicted_densities);
                kernels::iisph::launch_jacobi_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), previous.pressures.values.data(), current.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), current.pressures.values.data());
                pressure_forward(model, state.positions, particles, neighborhood, densities, current.pressures, current.pressure_accelerations);
            }
            for (std::uint32_t iteration = last.iteration; iteration > first.iteration; --iteration) {
                PressureIterationCache& previous = workspace.recomputed_iterations[iteration - first.iteration - 1u];
                PressureIterationCache& current  = workspace.recomputed_iterations[iteration - first.iteration];
                clear_iteration(model, workspace.previous_adjoint);
                pressure_vjp(model, state.positions, particles, neighborhood, densities, current.pressures, workspace.adjoint.pressure_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.adjoint.pressures, particle_adjoint);
                kernels::iisph::launch_jacobi_update_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), previous.pressures.values.data(), current.predicted_densities.values.data(), parameters.jacobi_relaxation.data(), workspace.adjoint.pressures.values.data(), device::particle_parameter_adjoint(particle_adjoint), density_adjoint.values.data(), workspace.previous_adjoint.pressures.values.data(), workspace.adjoint.predicted_densities.values.data(), parameter_adjoint.jacobi_relaxation.data());
                density.vjp(model, state.positions, current.predicted_positions, particles, neighborhood, workspace.adjoint.predicted_densities, workspace.adjoint.predicted_positions, particle_adjoint);
                kernels::common::predict_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(previous.pressure_accelerations), simulation::view(workspace.adjoint.predicted_positions), simulation::view(workspace.adjoint.predicted_velocities), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.previous_adjoint.pressure_accelerations));
                std::swap(workspace.adjoint, workspace.previous_adjoint);
            }
        }
        non_pressure_vjp(model, state, particles, neighborhood, densities, workspace.non_pressure_accelerations, previous_state_adjoint, control_adjoint, density_adjoint, particle_adjoint);
    }

    DivergenceFree::DivergenceFree(const meshfree::Model& model, Configuration next_configuration) : configuration(std::move(next_configuration)), reference_gradient_norm(compute_reference_gradient_norm(model.configuration)), density({}) {}

    DivergenceFree::State DivergenceFree::allocate_state(const meshfree::Model& model) const {
        State state{.warm_divergence_pressure = simulation::ScalarField<float>(model.stream, model.configuration.particle_count), .warm_density_pressure = simulation::ScalarField<float>(model.stream, model.configuration.particle_count)};
        simulation::clear(model.stream, state.warm_divergence_pressure);
        simulation::clear(model.stream, state.warm_density_pressure);
        return state;
    }

    DivergenceFree::StateTangent DivergenceFree::allocate_state_tangent(const meshfree::Model& model) const {
        StateTangent tangent{.warm_divergence_pressure = simulation::ScalarField<float>(model.stream, model.configuration.particle_count), .warm_density_pressure = simulation::ScalarField<float>(model.stream, model.configuration.particle_count)};
        simulation::clear(model.stream, tangent.warm_divergence_pressure);
        simulation::clear(model.stream, tangent.warm_density_pressure);
        return tangent;
    }

    DivergenceFree::StateAdjoint DivergenceFree::allocate_state_adjoint(const meshfree::Model& model) const {
        StateAdjoint adjoint{.warm_divergence_pressure = simulation::ScalarField<double>(model.stream, model.configuration.particle_count), .warm_density_pressure = simulation::ScalarField<double>(model.stream, model.configuration.particle_count)};
        simulation::clear(model.stream, adjoint.warm_divergence_pressure);
        simulation::clear(model.stream, adjoint.warm_density_pressure);
        return adjoint;
    }

    DivergenceFree::Parameters DivergenceFree::allocate_parameters(const meshfree::Model& model) const {
        return {.divergence_relaxation = allocate_buffer<::cuda::device_buffer<float>>(model), .density_relaxation = allocate_buffer<::cuda::device_buffer<float>>(model)};
    }

    DivergenceFree::ParameterTangent DivergenceFree::allocate_parameter_tangent(const meshfree::Model& model) const {
        ParameterTangent tangent{.divergence_relaxation = allocate_buffer<::cuda::device_buffer<float>>(model), .density_relaxation = allocate_buffer<::cuda::device_buffer<float>>(model)};
        ::cuda::fill_bytes(model.stream, tangent.divergence_relaxation, 0u);
        ::cuda::fill_bytes(model.stream, tangent.density_relaxation, 0u);
        return tangent;
    }

    DivergenceFree::ParameterAdjoint DivergenceFree::allocate_parameter_adjoint(const meshfree::Model& model) const {
        ParameterAdjoint adjoint{.divergence_relaxation = allocate_buffer<::cuda::device_buffer<double>>(model), .density_relaxation = allocate_buffer<::cuda::device_buffer<double>>(model)};
        ::cuda::fill_bytes(model.stream, adjoint.divergence_relaxation, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.density_relaxation, 0u);
        return adjoint;
    }

    DivergenceFree::Cache DivergenceFree::allocate_cache(const meshfree::Model& model) const {
        Cache cache{
            .non_pressure_accelerations        = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .divergence_pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .total_pressure_accelerations      = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
        const std::uint32_t divergence_count = 1u + configuration.divergence_iterations / configuration.checkpoint_interval + (configuration.divergence_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.divergence_checkpoints.reserve(divergence_count);
        for (std::uint32_t checkpoint = 0u; checkpoint < divergence_count; ++checkpoint) cache.divergence_checkpoints.push_back(allocate_iteration_cache(model));
        const std::uint32_t density_count = 1u + configuration.density_iterations / configuration.checkpoint_interval + (configuration.density_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.density_checkpoints.reserve(density_count);
        for (std::uint32_t checkpoint = 0u; checkpoint < density_count; ++checkpoint) cache.density_checkpoints.push_back(allocate_iteration_cache(model));
        return cache;
    }

    DivergenceFree::Workspace DivergenceFree::allocate_workspace(const meshfree::Model& model) const {
        return {
            .primal                       = allocate_iteration_cache(model),
            .total_pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
    }

    DivergenceFree::TangentWorkspace DivergenceFree::allocate_tangent_workspace(const meshfree::Model& model) const {
        return {
            .primal                               = allocate_iteration_cache(model),
            .tangent                              = allocate_iteration_tangent(model),
            .non_pressure_accelerations           = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .divergence_pressure_accelerations    = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .primal_total_pressure_accelerations  = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .tangent_total_pressure_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
    }

    DivergenceFree::AdjointWorkspace DivergenceFree::allocate_adjoint_workspace(const meshfree::Model& model) const {
        AdjointWorkspace workspace{
            .adjoint                              = allocate_iteration_adjoint(model),
            .previous_adjoint                     = allocate_iteration_adjoint(model),
            .total_pressure_accelerations         = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .target_densities                     = simulation::ScalarField<double>(model.stream, model.configuration.particle_count),
            .non_pressure_accelerations           = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .divergence_pressure_accelerations    = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .total_pressure_accelerations_adjoint = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
        };
        workspace.recomputed_iterations.reserve(configuration.checkpoint_interval + 1u);
        for (std::uint32_t iteration = 0u; iteration <= configuration.checkpoint_interval; ++iteration) workspace.recomputed_iterations.push_back(allocate_iteration_cache(model));
        return workspace;
    }

    void DivergenceFree::copy_state(const meshfree::Model& model, const State& source, State& destination) const {
        simulation::copy(model.stream, source.warm_divergence_pressure, destination.warm_divergence_pressure);
        simulation::copy(model.stream, source.warm_density_pressure, destination.warm_density_pressure);
    }

    void DivergenceFree::copy_state_tangent(const meshfree::Model& model, const StateTangent& source, StateTangent& destination) const {
        simulation::copy(model.stream, source.warm_divergence_pressure, destination.warm_divergence_pressure);
        simulation::copy(model.stream, source.warm_density_pressure, destination.warm_density_pressure);
    }

    void DivergenceFree::copy_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const {
        simulation::copy(model.stream, source.warm_divergence_pressure, destination.warm_divergence_pressure);
        simulation::copy(model.stream, source.warm_density_pressure, destination.warm_density_pressure);
    }

    void DivergenceFree::accumulate_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const {
        simulation::accumulate(model.stream, source.warm_divergence_pressure, destination.warm_divergence_pressure);
        simulation::accumulate(model.stream, source.warm_density_pressure, destination.warm_density_pressure);
    }

    void DivergenceFree::forward(const meshfree::Model& model, const ParticleState& state, const State& method_state, const Control& control, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, ParticleState& next_state, State& next_method_state, Cache& cache, Workspace& workspace) const {
        non_pressure_forward(model, configuration.gravity, state, control, particles, neighborhood, densities, cache.non_pressure_accelerations);
        clear_iteration(model, workspace.primal);
        if (configuration.pressure_warm_start) simulation::copy(model.stream, method_state.warm_divergence_pressure, workspace.primal.pressures);
        pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
        workspace.primal.iteration = 0u;
        copy_iteration(model, workspace.primal, cache.divergence_checkpoints[0]);
        std::uint32_t checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.divergence_iterations; ++iteration) {
            kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(workspace.primal.predicted_positions), simulation::view(workspace.primal.predicted_velocities));
            density.forward(model, state.positions, workspace.primal.predicted_positions, particles, neighborhood, workspace.primal.predicted_densities);
            kernels::dfsph::launch_projection_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), densities.values.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.divergence_relaxation.data(), workspace.primal.pressures.values.data());
            pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
            workspace.primal.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.divergence_iterations) copy_iteration(model, workspace.primal, cache.divergence_checkpoints[checkpoint++]);
        }
        simulation::copy(model.stream, workspace.primal.pressure_accelerations, cache.divergence_pressure_accelerations);
        simulation::copy(model.stream, workspace.primal.pressures, next_method_state.warm_divergence_pressure);

        clear_iteration(model, workspace.primal);
        if (configuration.pressure_warm_start) simulation::copy(model.stream, method_state.warm_density_pressure, workspace.primal.pressures);
        pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
        workspace.primal.iteration = 0u;
        copy_iteration(model, workspace.primal, cache.density_checkpoints[0]);
        checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.density_iterations; ++iteration) {
            add(model, cache.divergence_pressure_accelerations, workspace.primal.pressure_accelerations, workspace.total_pressure_accelerations);
            kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.total_pressure_accelerations), simulation::view(workspace.primal.predicted_positions), simulation::view(workspace.primal.predicted_velocities));
            density.forward(model, state.positions, workspace.primal.predicted_positions, particles, neighborhood, workspace.primal.predicted_densities);
            kernels::dfsph::launch_projection_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), particles.rest_densities.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.density_relaxation.data(), workspace.primal.pressures.values.data());
            pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
            workspace.primal.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.density_iterations) copy_iteration(model, workspace.primal, cache.density_checkpoints[checkpoint++]);
        }
        add(model, cache.divergence_pressure_accelerations, workspace.primal.pressure_accelerations, cache.total_pressure_accelerations);
        simulation::copy(model.stream, workspace.primal.pressures, next_method_state.warm_density_pressure);
        kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(cache.total_pressure_accelerations), simulation::view(next_state.positions), simulation::view(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void DivergenceFree::jvp(const meshfree::Model& model, const ParticleState& state, const State& method_state, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const Cache& cache, const ParticleStateTangent& state_tangent, const StateTangent& method_state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& particle_tangent, const ParameterTangent& parameter_tangent, const simulation::ScalarField<float>& density_tangent, ParticleStateTangent& next_state_tangent, StateTangent& next_method_state_tangent, TangentWorkspace& workspace) const {
        non_pressure_jvp(model, state, particles, neighborhood, densities, state_tangent, control_tangent, particle_tangent, density_tangent, workspace.non_pressure_accelerations);
        clear_iteration(model, workspace.primal);
        clear_iteration(model, workspace.tangent);
        if (configuration.pressure_warm_start) {
            simulation::copy(model.stream, method_state.warm_divergence_pressure, workspace.primal.pressures);
            simulation::copy(model.stream, method_state_tangent.warm_divergence_pressure, workspace.tangent.pressures);
        }
        pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
        pressure_jvp(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
        for (std::uint32_t iteration = 1u; iteration <= configuration.divergence_iterations; ++iteration) {
            kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(workspace.primal.predicted_positions), simulation::view(workspace.primal.predicted_velocities));
            kernels::common::predict_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal.pressure_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.tangent.pressure_accelerations), simulation::view(workspace.tangent.predicted_positions), simulation::view(workspace.tangent.predicted_velocities));
            density.forward(model, state.positions, workspace.primal.predicted_positions, particles, neighborhood, workspace.primal.predicted_densities);
            density.jvp(model, state.positions, workspace.primal.predicted_positions, workspace.tangent.predicted_positions, particles, particle_tangent, neighborhood, workspace.tangent.predicted_densities);
            kernels::dfsph::launch_projection_update_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), device::particle_parameter_tangent(particle_tangent), densities.values.data(), density_tangent.values.data(), densities.values.data(), density_tangent.values.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.divergence_relaxation.data(), workspace.tangent.pressures.values.data(), workspace.tangent.predicted_densities.values.data(), parameter_tangent.divergence_relaxation.data(), workspace.tangent.pressures.values.data());
            kernels::dfsph::launch_projection_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), densities.values.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.divergence_relaxation.data(), workspace.primal.pressures.values.data());
            pressure_jvp(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
            pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
        }
        simulation::copy(model.stream, workspace.tangent.pressure_accelerations, workspace.divergence_pressure_accelerations);
        simulation::copy(model.stream, workspace.tangent.pressures, next_method_state_tangent.warm_divergence_pressure);

        clear_iteration(model, workspace.primal);
        clear_iteration(model, workspace.tangent);
        if (configuration.pressure_warm_start) {
            simulation::copy(model.stream, method_state.warm_density_pressure, workspace.primal.pressures);
            simulation::copy(model.stream, method_state_tangent.warm_density_pressure, workspace.tangent.pressures);
        }
        pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
        pressure_jvp(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
        for (std::uint32_t iteration = 1u; iteration <= configuration.density_iterations; ++iteration) {
            add(model, cache.divergence_pressure_accelerations, workspace.primal.pressure_accelerations, workspace.primal_total_pressure_accelerations);
            add(model, workspace.divergence_pressure_accelerations, workspace.tangent.pressure_accelerations, workspace.tangent_total_pressure_accelerations);
            kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal_total_pressure_accelerations), simulation::view(workspace.primal.predicted_positions), simulation::view(workspace.primal.predicted_velocities));
            kernels::common::predict_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.primal_total_pressure_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.tangent_total_pressure_accelerations), simulation::view(workspace.tangent.predicted_positions), simulation::view(workspace.tangent.predicted_velocities));
            density.forward(model, state.positions, workspace.primal.predicted_positions, particles, neighborhood, workspace.primal.predicted_densities);
            density.jvp(model, state.positions, workspace.primal.predicted_positions, workspace.tangent.predicted_positions, particles, particle_tangent, neighborhood, workspace.tangent.predicted_densities);
            kernels::dfsph::launch_projection_update_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), device::particle_parameter_tangent(particle_tangent), densities.values.data(), density_tangent.values.data(), particles.rest_densities.data(), particle_tangent.rest_densities.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.density_relaxation.data(), workspace.tangent.pressures.values.data(), workspace.tangent.predicted_densities.values.data(), parameter_tangent.density_relaxation.data(), workspace.tangent.pressures.values.data());
            kernels::dfsph::launch_projection_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), particles.rest_densities.data(), workspace.primal.pressures.values.data(), workspace.primal.predicted_densities.values.data(), parameters.density_relaxation.data(), workspace.primal.pressures.values.data());
            pressure_jvp(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, state_tangent.positions, particle_tangent, density_tangent, workspace.tangent.pressures, workspace.tangent.pressure_accelerations);
            pressure_forward(model, state.positions, particles, neighborhood, densities, workspace.primal.pressures, workspace.primal.pressure_accelerations);
        }
        add(model, workspace.divergence_pressure_accelerations, workspace.tangent.pressure_accelerations, workspace.tangent_total_pressure_accelerations);
        simulation::copy(model.stream, workspace.tangent.pressures, next_method_state_tangent.warm_density_pressure);
        kernels::common::predict_jvp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(cache.total_pressure_accelerations), simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.tangent_total_pressure_accelerations), simulation::view(next_state_tangent.positions), simulation::view(next_state_tangent.velocities));
    }

    void DivergenceFree::vjp(const meshfree::Model& model, const ParticleState& state, const State&, const ParticleParameters& particles, const Parameters& parameters, const operators::Neighborhood& neighborhood, const simulation::ScalarField<float>& densities, const Cache& cache, const ParticleStateAdjoint& next_state_adjoint, const StateAdjoint& next_method_state_adjoint, ParticleStateAdjoint& previous_state_adjoint, StateAdjoint& previous_method_state_adjoint, ControlAdjoint& control_adjoint, ParticleParameterAdjoint& particle_adjoint, ParameterAdjoint& parameter_adjoint, simulation::ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const {
        simulation::clear(model.stream, workspace.target_densities);
        simulation::clear(model.stream, workspace.non_pressure_accelerations);
        simulation::clear(model.stream, workspace.divergence_pressure_accelerations);
        simulation::clear(model.stream, workspace.total_pressure_accelerations_adjoint);
        clear_iteration(model, workspace.adjoint);
        clear_iteration(model, workspace.previous_adjoint);

        kernels::common::predict_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(cache.total_pressure_accelerations), simulation::view(next_state_adjoint.positions), simulation::view(next_state_adjoint.velocities), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.total_pressure_accelerations_adjoint));
        simulation::copy(model.stream, workspace.total_pressure_accelerations_adjoint, workspace.divergence_pressure_accelerations);
        simulation::copy(model.stream, workspace.total_pressure_accelerations_adjoint, workspace.adjoint.pressure_accelerations);
        simulation::copy(model.stream, next_method_state_adjoint.warm_density_pressure, workspace.adjoint.pressures);

        const auto reverse_phase = [&](const std::vector<PressureIterationCache>& checkpoints, const ::cuda::device_buffer<float>& relaxation, ::cuda::device_buffer<double>& relaxation_adjoint, const float* target_densities, double* target_density_adjoint, const simulation::VectorField<float>* base_pressure_accelerations, simulation::VectorField<double>* base_pressure_acceleration_adjoint, simulation::ScalarField<double>& warm_pressure_adjoint) {
            for (std::size_t checkpoint = checkpoints.size(); checkpoint-- > 1uz;) {
                const PressureIterationCache& first = checkpoints[checkpoint - 1uz];
                const PressureIterationCache& last  = checkpoints[checkpoint];
                copy_iteration(model, first, workspace.recomputed_iterations[0]);
                for (std::uint32_t iteration = first.iteration + 1u; iteration <= last.iteration; ++iteration) {
                    PressureIterationCache& previous                          = workspace.recomputed_iterations[iteration - first.iteration - 1u];
                    PressureIterationCache& current                           = workspace.recomputed_iterations[iteration - first.iteration];
                    current.iteration                                         = iteration;
                    const simulation::VectorField<float>* prediction_pressure = &previous.pressure_accelerations;
                    if (base_pressure_accelerations != nullptr) {
                        add(model, *base_pressure_accelerations, previous.pressure_accelerations, workspace.total_pressure_accelerations);
                        prediction_pressure = &workspace.total_pressure_accelerations;
                    }
                    kernels::common::predict_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(*prediction_pressure), simulation::view(current.predicted_positions), simulation::view(current.predicted_velocities));
                    density.forward(model, state.positions, current.predicted_positions, particles, neighborhood, current.predicted_densities);
                    kernels::dfsph::launch_projection_update_forward(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), target_densities, previous.pressures.values.data(), current.predicted_densities.values.data(), relaxation.data(), current.pressures.values.data());
                    pressure_forward(model, state.positions, particles, neighborhood, densities, current.pressures, current.pressure_accelerations);
                }
                for (std::uint32_t iteration = last.iteration; iteration > first.iteration; --iteration) {
                    PressureIterationCache& previous = workspace.recomputed_iterations[iteration - first.iteration - 1u];
                    PressureIterationCache& current  = workspace.recomputed_iterations[iteration - first.iteration];
                    clear_iteration(model, workspace.previous_adjoint);
                    pressure_vjp(model, state.positions, particles, neighborhood, densities, current.pressures, workspace.adjoint.pressure_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.adjoint.pressures, particle_adjoint);
                    kernels::dfsph::launch_projection_update_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, reference_gradient_norm, device::particle_parameters(particles), densities.values.data(), target_densities, previous.pressures.values.data(), current.predicted_densities.values.data(), relaxation.data(), workspace.adjoint.pressures.values.data(), device::particle_parameter_adjoint(particle_adjoint), density_adjoint.values.data(), target_density_adjoint, workspace.previous_adjoint.pressures.values.data(), workspace.adjoint.predicted_densities.values.data(), relaxation_adjoint.data());
                    density.vjp(model, state.positions, current.predicted_positions, particles, neighborhood, workspace.adjoint.predicted_densities, workspace.adjoint.predicted_positions, particle_adjoint);
                    if (base_pressure_accelerations == nullptr) {
                        kernels::common::predict_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(previous.pressure_accelerations), simulation::view(workspace.adjoint.predicted_positions), simulation::view(workspace.adjoint.predicted_velocities), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.previous_adjoint.pressure_accelerations));
                    } else {
                        simulation::clear(model.stream, workspace.total_pressure_accelerations_adjoint);
                        add(model, *base_pressure_accelerations, previous.pressure_accelerations, workspace.total_pressure_accelerations);
                        kernels::common::predict_vjp(model.stream, model.configuration.particle_count, model.configuration.time_step, device::collision_box(model.configuration, state.step_index), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(cache.non_pressure_accelerations), simulation::view(workspace.total_pressure_accelerations), simulation::view(workspace.adjoint.predicted_positions), simulation::view(workspace.adjoint.predicted_velocities), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(workspace.non_pressure_accelerations), simulation::view(workspace.total_pressure_accelerations_adjoint));
                        add_adjoint(model, workspace.total_pressure_accelerations_adjoint, *base_pressure_acceleration_adjoint, workspace.previous_adjoint.pressure_accelerations);
                    }
                    std::swap(workspace.adjoint, workspace.previous_adjoint);
                }
            }
            pressure_vjp(model, state.positions, particles, neighborhood, densities, checkpoints.front().pressures, workspace.adjoint.pressure_accelerations, previous_state_adjoint.positions, density_adjoint, workspace.adjoint.pressures, particle_adjoint);
            if (configuration.pressure_warm_start) simulation::copy(model.stream, workspace.adjoint.pressures, warm_pressure_adjoint);
        };

        reverse_phase(cache.density_checkpoints, parameters.density_relaxation, parameter_adjoint.density_relaxation, particles.rest_densities.data(), particle_adjoint.rest_densities.data(), &cache.divergence_pressure_accelerations, &workspace.divergence_pressure_accelerations, previous_method_state_adjoint.warm_density_pressure);
        clear_iteration(model, workspace.adjoint);
        simulation::copy(model.stream, workspace.divergence_pressure_accelerations, workspace.adjoint.pressure_accelerations);
        simulation::copy(model.stream, next_method_state_adjoint.warm_divergence_pressure, workspace.adjoint.pressures);
        reverse_phase(cache.divergence_checkpoints, parameters.divergence_relaxation, parameter_adjoint.divergence_relaxation, densities.values.data(), workspace.target_densities.values.data(), nullptr, nullptr, previous_method_state_adjoint.warm_divergence_pressure);
        simulation::accumulate(model.stream, workspace.target_densities, density_adjoint);
        non_pressure_vjp(model, state, particles, neighborhood, densities, workspace.non_pressure_accelerations, previous_state_adjoint, control_adjoint, density_adjoint, particle_adjoint);
    }
} // namespace physica::fluids::liquid::solvers::sph
