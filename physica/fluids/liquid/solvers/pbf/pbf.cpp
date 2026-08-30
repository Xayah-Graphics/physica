module;

#include <fluids/liquid/interop.h>
#include "pbf-kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.solvers.pbf;

import std;

namespace physica::fluids::liquid::solvers::pbf {
    namespace {
        template <class Buffer>
        Buffer allocate_buffer(const meshfree::Model& model) {
            return Buffer{model.stream, ::cuda::device_default_memory_pool(model.stream.device()), model.configuration.particle_count, ::cuda::no_init};
        }
    } // namespace

    Solver::Solver(Configuration next_configuration) : configuration(std::move(next_configuration)), density({}) {}

    Solver::State Solver::allocate_state(const meshfree::Model& model) const {
        State state{
            .positions  = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .velocities = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
        simulation::clear(model.stream, state.positions);
        simulation::clear(model.stream, state.velocities);
        return state;
    }

    Solver::Control Solver::allocate_control(const meshfree::Model& model) const {
        Control control{.external_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count)};
        simulation::clear(model.stream, control.external_accelerations);
        return control;
    }

    Solver::Parameters Solver::allocate_parameters(const meshfree::Model& model) const {
        return {
            .masses                       = allocate_buffer<::cuda::device_buffer<float>>(model),
            .rest_densities               = allocate_buffer<::cuda::device_buffer<float>>(model),
            .viscosities                  = allocate_buffer<::cuda::device_buffer<float>>(model),
            .surface_tensions             = allocate_buffer<::cuda::device_buffer<float>>(model),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<float>>(model),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<float>>(model),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<float>>(model),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<float>>(model),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<float>>(model),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<float>>(model),
        };
    }

    Solver::StepCache Solver::allocate_step_cache(const meshfree::Model& model) const {
        StepCache cache{
            .neighborhood             = neighborhood.allocate_cache(model),
            .predicted_positions      = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .corrected_positions      = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .reconstructed_velocities = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .vorticities              = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .vorticity_magnitudes     = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .vorticity_normals        = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .vorticity_normalizers    = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .confined_velocities      = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
        cache.checkpoints.push_back({.iteration = 0u, .positions = simulation::VectorField<float>(model.stream, model.configuration.particle_count)});
        for (std::uint32_t iteration = configuration.checkpoint_interval; iteration < configuration.pressure_iterations; iteration += configuration.checkpoint_interval) cache.checkpoints.push_back({.iteration = iteration, .positions = simulation::VectorField<float>(model.stream, model.configuration.particle_count)});
        if (configuration.pressure_iterations != 0u) cache.checkpoints.push_back({.iteration = configuration.pressure_iterations, .positions = simulation::VectorField<float>(model.stream, model.configuration.particle_count)});
        return cache;
    }

    Solver::Workspace Solver::allocate_workspace(const meshfree::Model& model) const {
        return {
            .neighborhood = neighborhood.allocate_workspace(model),
            .iteration    = allocate_iteration_workspace(model),
        };
    }

    Solver::StateTangent Solver::allocate_state_tangent(const meshfree::Model& model) const {
        StateTangent tangent{
            .positions  = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .velocities = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
        simulation::clear(model.stream, tangent.positions);
        simulation::clear(model.stream, tangent.velocities);
        return tangent;
    }

    Solver::ControlTangent Solver::allocate_control_tangent(const meshfree::Model& model) const {
        ControlTangent tangent{.external_accelerations = simulation::VectorField<float>(model.stream, model.configuration.particle_count)};
        simulation::clear(model.stream, tangent.external_accelerations);
        return tangent;
    }

    Solver::ParameterTangent Solver::allocate_parameter_tangent(const meshfree::Model& model) const {
        ParameterTangent tangent{
            .masses                       = allocate_buffer<::cuda::device_buffer<float>>(model),
            .rest_densities               = allocate_buffer<::cuda::device_buffer<float>>(model),
            .viscosities                  = allocate_buffer<::cuda::device_buffer<float>>(model),
            .surface_tensions             = allocate_buffer<::cuda::device_buffer<float>>(model),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<float>>(model),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<float>>(model),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<float>>(model),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<float>>(model),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<float>>(model),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<float>>(model),
        };
        ::cuda::fill_bytes(model.stream, tangent.masses, 0u);
        ::cuda::fill_bytes(model.stream, tangent.rest_densities, 0u);
        ::cuda::fill_bytes(model.stream, tangent.viscosities, 0u);
        ::cuda::fill_bytes(model.stream, tangent.surface_tensions, 0u);
        ::cuda::fill_bytes(model.stream, tangent.relaxation, 0u);
        ::cuda::fill_bytes(model.stream, tangent.artificial_pressure_strength, 0u);
        ::cuda::fill_bytes(model.stream, tangent.artificial_pressure_exponent, 0u);
        ::cuda::fill_bytes(model.stream, tangent.artificial_pressure_radius, 0u);
        ::cuda::fill_bytes(model.stream, tangent.xsph_viscosity, 0u);
        ::cuda::fill_bytes(model.stream, tangent.vorticity_confinement, 0u);
        return tangent;
    }

    Solver::TangentWorkspace Solver::allocate_tangent_workspace(const meshfree::Model& model) const {
        return {
            .positions                = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .iteration                = allocate_iteration_workspace(model),
            .current_positions        = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .next_positions           = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .densities                = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .gradient_sums            = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .denominators             = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .lambdas                  = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .corrections              = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .reconstructed_velocities = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .vorticities              = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .vorticity_magnitudes     = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .vorticity_normals        = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .confined_velocities      = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
        };
    }

    Solver::StateAdjoint Solver::allocate_state_adjoint(const meshfree::Model& model) const {
        StateAdjoint adjoint{
            .positions  = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .velocities = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
        };
        simulation::clear(model.stream, adjoint.positions);
        simulation::clear(model.stream, adjoint.velocities);
        return adjoint;
    }

    Solver::ControlAdjoint Solver::allocate_control_adjoint(const meshfree::Model& model) const {
        ControlAdjoint adjoint{.external_accelerations = simulation::VectorField<double>(model.stream, model.configuration.particle_count)};
        simulation::clear(model.stream, adjoint.external_accelerations);
        return adjoint;
    }

    Solver::ParameterAdjoint Solver::allocate_parameter_adjoint(const meshfree::Model& model) const {
        ParameterAdjoint adjoint{
            .masses                       = allocate_buffer<::cuda::device_buffer<double>>(model),
            .rest_densities               = allocate_buffer<::cuda::device_buffer<double>>(model),
            .viscosities                  = allocate_buffer<::cuda::device_buffer<double>>(model),
            .surface_tensions             = allocate_buffer<::cuda::device_buffer<double>>(model),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<double>>(model),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<double>>(model),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<double>>(model),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<double>>(model),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<double>>(model),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<double>>(model),
        };
        ::cuda::fill_bytes(model.stream, adjoint.masses, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.rest_densities, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.viscosities, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.surface_tensions, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.relaxation, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.artificial_pressure_strength, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.artificial_pressure_exponent, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.artificial_pressure_radius, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.xsph_viscosity, 0u);
        ::cuda::fill_bytes(model.stream, adjoint.vorticity_confinement, 0u);
        return adjoint;
    }

    Solver::AdjointWorkspace Solver::allocate_adjoint_workspace(const meshfree::Model& model) const {
        AdjointWorkspace workspace{
            .current_positions        = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .next_positions           = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .corrections              = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .lambdas                  = simulation::ScalarField<double>(model.stream, model.configuration.particle_count),
            .densities                = simulation::ScalarField<double>(model.stream, model.configuration.particle_count),
            .reconstructed_velocities = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .vorticities              = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .vorticity_normals        = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
            .confined_velocities      = simulation::VectorField<double>(model.stream, model.configuration.particle_count),
        };
        workspace.position_history.reserve(configuration.checkpoint_interval + 1u);
        for (std::uint32_t index = 0u; index <= configuration.checkpoint_interval; ++index) workspace.position_history.push_back(simulation::VectorField<float>(model.stream, model.configuration.particle_count));
        workspace.iteration_history.reserve(configuration.checkpoint_interval);
        for (std::uint32_t index = 0u; index < configuration.checkpoint_interval; ++index) workspace.iteration_history.push_back(allocate_iteration_workspace(model));
        return workspace;
    }

    void Solver::copy_state(const meshfree::Model& model, const State& source, State& destination) const {
        simulation::copy(model.stream, source.positions, destination.positions);
        simulation::copy(model.stream, source.velocities, destination.velocities);
        destination.step_index = source.step_index;
    }

    void Solver::copy_state_tangent(const meshfree::Model& model, const StateTangent& source, StateTangent& destination) const {
        simulation::copy(model.stream, source.positions, destination.positions);
        simulation::copy(model.stream, source.velocities, destination.velocities);
    }

    void Solver::copy_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const {
        simulation::copy(model.stream, source.positions, destination.positions);
        simulation::copy(model.stream, source.velocities, destination.velocities);
    }

    void Solver::accumulate_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const {
        simulation::accumulate(model.stream, source.positions, destination.positions);
        simulation::accumulate(model.stream, source.velocities, destination.velocities);
    }

    void Solver::forward(const meshfree::Model& model, const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
        const std::uint32_t count = model.configuration.particle_count;
        kernels::launch_predict_forward(model.stream, count, model.configuration.time_step, configuration.gravity.x, configuration.gravity.y, configuration.gravity.z, simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_accelerations), simulation::view(cache.predicted_positions));
        neighborhood.build(model, state.step_index + 1u, cache.predicted_positions, cache.neighborhood, workspace.neighborhood);
        simulation::copy(model.stream, cache.predicted_positions, cache.corrected_positions);
        simulation::copy(model.stream, cache.corrected_positions, cache.checkpoints[0].positions);
        std::size_t checkpoint = 1uz;
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            density.forward(model, cache.predicted_positions, cache.corrected_positions, parameters, cache.neighborhood, workspace.iteration.densities);
            kernels::launch_lambda_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), parameters.relaxation.data(), simulation::view(workspace.iteration.gradient_sums), workspace.iteration.denominators.values.data(), workspace.iteration.lambdas.values.data());
            kernels::launch_correction_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), simulation::view(workspace.iteration.corrections));
            kernels::launch_project_forward(model.stream, count, device::collision_box(model.configuration, state.step_index + 1u), simulation::view(cache.corrected_positions), simulation::view(workspace.iteration.corrections), workspace.iteration.collision_masks.data(), simulation::view(cache.corrected_positions));
            if (checkpoint < cache.checkpoints.size() && cache.checkpoints[checkpoint].iteration == iteration + 1u) {
                simulation::copy(model.stream, cache.corrected_positions, cache.checkpoints[checkpoint].positions);
                ++checkpoint;
            }
        }
        kernels::launch_reconstruct_forward(model.stream, count, 1.0F / model.configuration.time_step, simulation::view(state.positions), simulation::view(cache.corrected_positions), simulation::view(cache.reconstructed_velocities));
        kernels::launch_vorticity_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.reconstructed_velocities), device::neighborhood(cache.neighborhood), simulation::view(cache.vorticities));
        kernels::launch_normal_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.vorticities), device::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), simulation::view(cache.vorticity_normals), cache.vorticity_normalizers.values.data());
        kernels::launch_confinement_forward(model.stream, count, model.configuration.time_step, simulation::view(cache.reconstructed_velocities), simulation::view(cache.vorticities), simulation::view(cache.vorticity_normals), parameters.vorticity_confinement.data(), simulation::view(cache.confined_velocities));
        kernels::launch_xsph_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.confined_velocities), device::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), simulation::view(next_state.velocities));
        simulation::copy(model.stream, cache.corrected_positions, next_state.positions);
        next_state.step_index = state.step_index + 1u;
    }

    void Solver::jvp(const meshfree::Model& model, const State& state, const Parameters& parameters, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const {
        const std::uint32_t count = model.configuration.particle_count;
        kernels::launch_predict_jvp(model.stream, count, model.configuration.time_step, simulation::view(state_tangent.positions), simulation::view(state_tangent.velocities), simulation::view(control_tangent.external_accelerations), simulation::view(workspace.current_positions));
        simulation::copy(model.stream, cache.predicted_positions, workspace.positions);
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            density.forward(model, cache.predicted_positions, workspace.positions, parameters, cache.neighborhood, workspace.iteration.densities);
            density.jvp(model, cache.predicted_positions, workspace.positions, workspace.current_positions, parameters, parameter_tangent, cache.neighborhood, workspace.densities);
            kernels::launch_lambda_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(workspace.positions), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), parameters.relaxation.data(), simulation::view(workspace.iteration.gradient_sums), workspace.iteration.denominators.values.data(), workspace.iteration.lambdas.values.data());
            kernels::launch_lambda_jvp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(workspace.positions), simulation::view(workspace.current_positions), device::particle_parameters(parameters), device::particle_parameter_tangent(parameter_tangent), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), workspace.densities.values.data(), parameters.relaxation.data(), parameter_tangent.relaxation.data(), simulation::view(workspace.gradient_sums), workspace.denominators.values.data(), workspace.lambdas.values.data());
            kernels::launch_correction_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(workspace.positions), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), simulation::view(workspace.iteration.corrections));
            kernels::launch_correction_jvp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(workspace.positions), simulation::view(workspace.current_positions), device::particle_parameters(parameters), device::particle_parameter_tangent(parameter_tangent), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), workspace.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameter_tangent.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameter_tangent.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), parameter_tangent.artificial_pressure_radius.data(), simulation::view(workspace.corrections));
            kernels::launch_project_forward(model.stream, count, device::collision_box(model.configuration, state.step_index + 1u), simulation::view(workspace.positions), simulation::view(workspace.iteration.corrections), workspace.iteration.collision_masks.data(), simulation::view(workspace.positions));
            kernels::launch_project_jvp(model.stream, count, workspace.iteration.collision_masks.data(), simulation::view(workspace.current_positions), simulation::view(workspace.corrections), simulation::view(workspace.next_positions));
            std::swap(workspace.current_positions, workspace.next_positions);
        }
        kernels::launch_reconstruct_jvp(model.stream, count, 1.0F / model.configuration.time_step, simulation::view(state_tangent.positions), simulation::view(workspace.current_positions), simulation::view(workspace.reconstructed_velocities));
        kernels::launch_vorticity_jvp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.reconstructed_velocities), simulation::view(workspace.current_positions), simulation::view(workspace.reconstructed_velocities), device::neighborhood(cache.neighborhood), simulation::view(workspace.vorticities));
        kernels::launch_normal_jvp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.vorticities), simulation::view(workspace.current_positions), simulation::view(workspace.vorticities), device::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), simulation::view(cache.vorticity_normals), cache.vorticity_normalizers.values.data(), workspace.vorticity_magnitudes.values.data(), simulation::view(workspace.vorticity_normals));
        kernels::launch_confinement_jvp(model.stream, count, model.configuration.time_step, simulation::view(cache.vorticities), simulation::view(cache.vorticity_normals), parameters.vorticity_confinement.data(), simulation::view(workspace.reconstructed_velocities), simulation::view(workspace.vorticities), simulation::view(workspace.vorticity_normals), parameter_tangent.vorticity_confinement.data(), simulation::view(workspace.confined_velocities));
        kernels::launch_xsph_jvp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.confined_velocities), simulation::view(workspace.current_positions), simulation::view(workspace.confined_velocities), device::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), parameter_tangent.xsph_viscosity.data(), simulation::view(next_state_tangent.velocities));
        simulation::copy(model.stream, workspace.current_positions, next_state_tangent.positions);
    }

    void Solver::vjp(const meshfree::Model& model, const State& state, const Parameters& parameters, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
        const std::uint32_t count = model.configuration.particle_count;
        simulation::copy(model.stream, next_state_adjoint.positions, workspace.current_positions);
        simulation::clear(model.stream, workspace.confined_velocities);
        kernels::launch_xsph_vjp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.confined_velocities), device::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), simulation::view(next_state_adjoint.velocities), simulation::view(workspace.current_positions), simulation::view(workspace.confined_velocities), parameter_adjoint.xsph_viscosity.data());

        simulation::clear(model.stream, workspace.reconstructed_velocities);
        simulation::clear(model.stream, workspace.vorticities);
        simulation::clear(model.stream, workspace.vorticity_normals);
        kernels::launch_confinement_vjp(model.stream, count, model.configuration.time_step, simulation::view(cache.vorticities), simulation::view(cache.vorticity_normals), parameters.vorticity_confinement.data(), simulation::view(workspace.confined_velocities), simulation::view(workspace.reconstructed_velocities), simulation::view(workspace.vorticities), simulation::view(workspace.vorticity_normals), parameter_adjoint.vorticity_confinement.data());
        kernels::launch_normal_vjp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.vorticities), device::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), simulation::view(cache.vorticity_normals), cache.vorticity_normalizers.values.data(), simulation::view(workspace.vorticity_normals), simulation::view(workspace.current_positions), simulation::view(workspace.vorticities));
        kernels::launch_vorticity_vjp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(cache.corrected_positions), simulation::view(cache.reconstructed_velocities), device::neighborhood(cache.neighborhood), simulation::view(workspace.vorticities), simulation::view(workspace.current_positions), simulation::view(workspace.reconstructed_velocities));
        kernels::launch_reconstruct_vjp(model.stream, count, 1.0F / model.configuration.time_step, simulation::view(workspace.reconstructed_velocities), simulation::view(previous_state_adjoint.positions), simulation::view(workspace.current_positions));

        for (std::size_t segment = cache.checkpoints.size(); segment-- > 1uz;) {
            const StepCache::IterationCheckpoint& start = cache.checkpoints[segment - 1uz];
            const StepCache::IterationCheckpoint& end   = cache.checkpoints[segment];
            const std::uint32_t segment_length          = end.iteration - start.iteration;
            simulation::copy(model.stream, start.positions, workspace.position_history[0]);
            for (std::uint32_t local_iteration = 0u; local_iteration < segment_length; ++local_iteration) {
                IterationWorkspace& iteration = workspace.iteration_history[local_iteration];
                density.forward(model, cache.predicted_positions, workspace.position_history[local_iteration], parameters, cache.neighborhood, iteration.densities);
                kernels::launch_lambda_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(workspace.position_history[local_iteration]), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), iteration.densities.values.data(), parameters.relaxation.data(), simulation::view(iteration.gradient_sums), iteration.denominators.values.data(), iteration.lambdas.values.data());
                kernels::launch_correction_forward(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(workspace.position_history[local_iteration]), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), simulation::view(iteration.corrections));
                kernels::launch_project_forward(model.stream, count, device::collision_box(model.configuration, state.step_index + 1u), simulation::view(workspace.position_history[local_iteration]), simulation::view(iteration.corrections), iteration.collision_masks.data(), simulation::view(workspace.position_history[local_iteration + 1u]));
            }
            for (std::uint32_t local_iteration = segment_length; local_iteration-- > 0u;) {
                IterationWorkspace& iteration = workspace.iteration_history[local_iteration];
                simulation::clear(model.stream, workspace.next_positions);
                simulation::clear(model.stream, workspace.corrections);
                simulation::clear(model.stream, workspace.lambdas);
                simulation::clear(model.stream, workspace.densities);
                kernels::launch_project_vjp(model.stream, count, iteration.collision_masks.data(), simulation::view(workspace.current_positions), simulation::view(workspace.next_positions), simulation::view(workspace.corrections));
                kernels::launch_correction_vjp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(workspace.position_history[local_iteration]), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), simulation::view(workspace.corrections), simulation::view(workspace.next_positions), workspace.lambdas.values.data(), device::particle_parameter_adjoint(parameter_adjoint), parameter_adjoint.artificial_pressure_strength.data(), parameter_adjoint.artificial_pressure_exponent.data(), parameter_adjoint.artificial_pressure_radius.data());
                kernels::launch_lambda_vjp(model.stream, count, model.configuration.support_radius, simulation::view(cache.predicted_positions), simulation::view(workspace.position_history[local_iteration]), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), iteration.densities.values.data(), simulation::view(iteration.gradient_sums), iteration.denominators.values.data(), workspace.lambdas.values.data(), simulation::view(workspace.next_positions), workspace.densities.values.data(), device::particle_parameter_adjoint(parameter_adjoint), parameter_adjoint.relaxation.data());
                density.vjp(model, cache.predicted_positions, workspace.position_history[local_iteration], parameters, cache.neighborhood, workspace.densities, workspace.next_positions, parameter_adjoint);
                std::swap(workspace.current_positions, workspace.next_positions);
            }
        }
        kernels::launch_predict_vjp(model.stream, count, model.configuration.time_step, simulation::view(workspace.current_positions), simulation::view(previous_state_adjoint.positions), simulation::view(previous_state_adjoint.velocities), simulation::view(control_adjoint.external_accelerations));
    }

    Solver::IterationWorkspace Solver::allocate_iteration_workspace(const meshfree::Model& model) const {
        return {
            .densities       = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .gradient_sums   = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .denominators    = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .lambdas         = simulation::ScalarField<float>(model.stream, model.configuration.particle_count),
            .corrections     = simulation::VectorField<float>(model.stream, model.configuration.particle_count),
            .collision_masks = allocate_buffer<::cuda::device_buffer<std::uint32_t>>(model),
        };
    }
} // namespace physica::fluids::liquid::solvers::pbf
