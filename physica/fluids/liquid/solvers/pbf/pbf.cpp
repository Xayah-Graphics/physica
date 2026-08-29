module;

#include <physica/fluids/liquid/interop.h>
#include "pbf-kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.pbf;

import std;

namespace physica::fluids::liquid::pbf {
    namespace {
        template <class Buffer>
        Buffer allocate_buffer(const meshfree::Model& model) {
            return Buffer{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), model.configuration.particle_count, ::cuda::no_init};
        }
    } // namespace

    Solver::Solver(Configuration next_configuration) : configuration(std::move(next_configuration)), density({}) {}

    Solver::State Solver::allocate_state(const meshfree::Model& model) const {
        State state{
            .positions  = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .velocities = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
        };
        model.fields.clear(state.positions);
        model.fields.clear(state.velocities);
        return state;
    }

    Solver::Control Solver::allocate_control(const meshfree::Model& model) const {
        Control control{.external_accelerations = model.fields.allocate_vector_field<float>(model.configuration.particle_count)};
        model.fields.clear(control.external_accelerations);
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
            .predicted_positions      = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .corrected_positions      = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .reconstructed_velocities = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .vorticities              = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .vorticity_magnitudes     = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .vorticity_normals        = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .vorticity_normalizers    = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .confined_velocities      = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
        };
        cache.checkpoints.push_back({.iteration = 0u, .positions = model.fields.allocate_vector_field<float>(model.configuration.particle_count)});
        for (std::uint32_t iteration = configuration.checkpoint_interval; iteration < configuration.pressure_iterations; iteration += configuration.checkpoint_interval) cache.checkpoints.push_back({.iteration = iteration, .positions = model.fields.allocate_vector_field<float>(model.configuration.particle_count)});
        if (configuration.pressure_iterations != 0u) cache.checkpoints.push_back({.iteration = configuration.pressure_iterations, .positions = model.fields.allocate_vector_field<float>(model.configuration.particle_count)});
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
            .positions  = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .velocities = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
        };
        model.fields.clear(tangent.positions);
        model.fields.clear(tangent.velocities);
        return tangent;
    }

    Solver::ControlTangent Solver::allocate_control_tangent(const meshfree::Model& model) const {
        ControlTangent tangent{.external_accelerations = model.fields.allocate_vector_field<float>(model.configuration.particle_count)};
        model.fields.clear(tangent.external_accelerations);
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
        ::cuda::fill_bytes(model.fields.stream, tangent.masses, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.rest_densities, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.viscosities, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.surface_tensions, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.relaxation, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.artificial_pressure_strength, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.artificial_pressure_exponent, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.artificial_pressure_radius, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.xsph_viscosity, 0u);
        ::cuda::fill_bytes(model.fields.stream, tangent.vorticity_confinement, 0u);
        return tangent;
    }

    Solver::TangentWorkspace Solver::allocate_tangent_workspace(const meshfree::Model& model) const {
        return {
            .positions                = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .iteration                = allocate_iteration_workspace(model),
            .current_positions        = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .next_positions           = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .densities                = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .gradient_sums            = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .denominators             = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .lambdas                  = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .corrections              = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .reconstructed_velocities = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .vorticities              = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .vorticity_magnitudes     = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .vorticity_normals        = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .confined_velocities      = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
        };
    }

    Solver::StateAdjoint Solver::allocate_state_adjoint(const meshfree::Model& model) const {
        StateAdjoint adjoint{
            .positions  = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
            .velocities = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
        };
        model.fields.clear(adjoint.positions);
        model.fields.clear(adjoint.velocities);
        return adjoint;
    }

    Solver::ControlAdjoint Solver::allocate_control_adjoint(const meshfree::Model& model) const {
        ControlAdjoint adjoint{.external_accelerations = model.fields.allocate_vector_field<double>(model.configuration.particle_count)};
        model.fields.clear(adjoint.external_accelerations);
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
        ::cuda::fill_bytes(model.fields.stream, adjoint.masses, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.rest_densities, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.viscosities, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.surface_tensions, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.relaxation, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.artificial_pressure_strength, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.artificial_pressure_exponent, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.artificial_pressure_radius, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.xsph_viscosity, 0u);
        ::cuda::fill_bytes(model.fields.stream, adjoint.vorticity_confinement, 0u);
        return adjoint;
    }

    Solver::AdjointWorkspace Solver::allocate_adjoint_workspace(const meshfree::Model& model) const {
        AdjointWorkspace workspace{
            .current_positions        = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
            .next_positions           = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
            .corrections              = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
            .lambdas                  = model.fields.allocate_scalar_field<double>(model.configuration.particle_count),
            .densities                = model.fields.allocate_scalar_field<double>(model.configuration.particle_count),
            .reconstructed_velocities = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
            .vorticities              = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
            .vorticity_normals        = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
            .confined_velocities      = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
        };
        workspace.position_history.reserve(configuration.checkpoint_interval + 1u);
        for (std::uint32_t index = 0u; index <= configuration.checkpoint_interval; ++index) workspace.position_history.push_back(model.fields.allocate_vector_field<float>(model.configuration.particle_count));
        workspace.iteration_history.reserve(configuration.checkpoint_interval);
        for (std::uint32_t index = 0u; index < configuration.checkpoint_interval; ++index) workspace.iteration_history.push_back(allocate_iteration_workspace(model));
        return workspace;
    }

    void Solver::copy_state(const meshfree::Model& model, const State& source, State& destination) const {
        model.fields.copy(source.positions, destination.positions);
        model.fields.copy(source.velocities, destination.velocities);
        destination.step_index = source.step_index;
    }

    void Solver::copy_state_tangent(const meshfree::Model& model, const StateTangent& source, StateTangent& destination) const {
        model.fields.copy(source.positions, destination.positions);
        model.fields.copy(source.velocities, destination.velocities);
    }

    void Solver::copy_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const {
        model.fields.copy(source.positions, destination.positions);
        model.fields.copy(source.velocities, destination.velocities);
    }

    void Solver::accumulate_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const {
        model.fields.accumulate(source.positions, destination.positions);
        model.fields.accumulate(source.velocities, destination.velocities);
    }

    void Solver::forward(const meshfree::Model& model, const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
        const std::uint32_t count = model.configuration.particle_count;
        kernels::launch_predict_forward(model.fields.stream, count, model.configuration.time_step, configuration.gravity.x, configuration.gravity.y, configuration.gravity.z, field::view(state.positions), field::view(state.velocities), field::view(control.external_accelerations), field::view(cache.predicted_positions));
        neighborhood.build(model, state.step_index + 1u, cache.predicted_positions, cache.neighborhood, workspace.neighborhood);
        model.fields.copy(cache.predicted_positions, cache.corrected_positions);
        model.fields.copy(cache.corrected_positions, cache.checkpoints[0].positions);
        std::size_t checkpoint = 1uz;
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            density.forward(model, cache.predicted_positions, cache.corrected_positions, parameters, cache.neighborhood, workspace.iteration.densities);
            kernels::launch_lambda_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), parameters.relaxation.data(), field::view(workspace.iteration.gradient_sums), workspace.iteration.denominators.values.data(), workspace.iteration.lambdas.values.data());
            kernels::launch_correction_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), field::view(workspace.iteration.corrections));
            kernels::launch_project_forward(model.fields.stream, count, device::collision_box(model.configuration, state.step_index + 1u), field::view(cache.corrected_positions), field::view(workspace.iteration.corrections), workspace.iteration.collision_masks.data(), field::view(cache.corrected_positions));
            if (checkpoint < cache.checkpoints.size() && cache.checkpoints[checkpoint].iteration == iteration + 1u) {
                model.fields.copy(cache.corrected_positions, cache.checkpoints[checkpoint].positions);
                ++checkpoint;
            }
        }
        kernels::launch_reconstruct_forward(model.fields.stream, count, 1.0F / model.configuration.time_step, field::view(state.positions), field::view(cache.corrected_positions), field::view(cache.reconstructed_velocities));
        kernels::launch_vorticity_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.reconstructed_velocities), device::neighborhood(cache.neighborhood), field::view(cache.vorticities));
        kernels::launch_normal_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.vorticities), device::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), field::view(cache.vorticity_normals), cache.vorticity_normalizers.values.data());
        kernels::launch_confinement_forward(model.fields.stream, count, model.configuration.time_step, field::view(cache.reconstructed_velocities), field::view(cache.vorticities), field::view(cache.vorticity_normals), parameters.vorticity_confinement.data(), field::view(cache.confined_velocities));
        kernels::launch_xsph_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.confined_velocities), device::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), field::view(next_state.velocities));
        model.fields.copy(cache.corrected_positions, next_state.positions);
        next_state.step_index = state.step_index + 1u;
    }

    void Solver::jvp(const meshfree::Model& model, const State& state, const Parameters& parameters, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const {
        const std::uint32_t count = model.configuration.particle_count;
        kernels::launch_predict_jvp(model.fields.stream, count, model.configuration.time_step, field::view(state_tangent.positions), field::view(state_tangent.velocities), field::view(control_tangent.external_accelerations), field::view(workspace.current_positions));
        model.fields.copy(cache.predicted_positions, workspace.positions);
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            density.forward(model, cache.predicted_positions, workspace.positions, parameters, cache.neighborhood, workspace.iteration.densities);
            density.jvp(model, cache.predicted_positions, workspace.positions, workspace.current_positions, parameters, parameter_tangent, cache.neighborhood, workspace.densities);
            kernels::launch_lambda_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(workspace.positions), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), parameters.relaxation.data(), field::view(workspace.iteration.gradient_sums), workspace.iteration.denominators.values.data(), workspace.iteration.lambdas.values.data());
            kernels::launch_lambda_jvp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(workspace.positions), field::view(workspace.current_positions), device::particle_parameters(parameters), device::particle_parameter_tangent(parameter_tangent), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), workspace.densities.values.data(), parameters.relaxation.data(), parameter_tangent.relaxation.data(), field::view(workspace.gradient_sums), workspace.denominators.values.data(), workspace.lambdas.values.data());
            kernels::launch_correction_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(workspace.positions), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), field::view(workspace.iteration.corrections));
            kernels::launch_correction_jvp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(workspace.positions), field::view(workspace.current_positions), device::particle_parameters(parameters), device::particle_parameter_tangent(parameter_tangent), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), workspace.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameter_tangent.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameter_tangent.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), parameter_tangent.artificial_pressure_radius.data(), field::view(workspace.corrections));
            kernels::launch_project_forward(model.fields.stream, count, device::collision_box(model.configuration, state.step_index + 1u), field::view(workspace.positions), field::view(workspace.iteration.corrections), workspace.iteration.collision_masks.data(), field::view(workspace.positions));
            kernels::launch_project_jvp(model.fields.stream, count, workspace.iteration.collision_masks.data(), field::view(workspace.current_positions), field::view(workspace.corrections), field::view(workspace.next_positions));
            std::swap(workspace.current_positions, workspace.next_positions);
        }
        kernels::launch_reconstruct_jvp(model.fields.stream, count, 1.0F / model.configuration.time_step, field::view(state_tangent.positions), field::view(workspace.current_positions), field::view(workspace.reconstructed_velocities));
        kernels::launch_vorticity_jvp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.reconstructed_velocities), field::view(workspace.current_positions), field::view(workspace.reconstructed_velocities), device::neighborhood(cache.neighborhood), field::view(workspace.vorticities));
        kernels::launch_normal_jvp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.vorticities), field::view(workspace.current_positions), field::view(workspace.vorticities), device::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), field::view(cache.vorticity_normals), cache.vorticity_normalizers.values.data(), workspace.vorticity_magnitudes.values.data(), field::view(workspace.vorticity_normals));
        kernels::launch_confinement_jvp(model.fields.stream, count, model.configuration.time_step, field::view(cache.vorticities), field::view(cache.vorticity_normals), parameters.vorticity_confinement.data(), field::view(workspace.reconstructed_velocities), field::view(workspace.vorticities), field::view(workspace.vorticity_normals), parameter_tangent.vorticity_confinement.data(), field::view(workspace.confined_velocities));
        kernels::launch_xsph_jvp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.confined_velocities), field::view(workspace.current_positions), field::view(workspace.confined_velocities), device::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), parameter_tangent.xsph_viscosity.data(), field::view(next_state_tangent.velocities));
        model.fields.copy(workspace.current_positions, next_state_tangent.positions);
    }

    void Solver::vjp(const meshfree::Model& model, const State& state, const Parameters& parameters, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
        const std::uint32_t count = model.configuration.particle_count;
        model.fields.copy(next_state_adjoint.positions, workspace.current_positions);
        model.fields.clear(workspace.confined_velocities);
        kernels::launch_xsph_vjp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.confined_velocities), device::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), field::view(next_state_adjoint.velocities), field::view(workspace.current_positions), field::view(workspace.confined_velocities), parameter_adjoint.xsph_viscosity.data());

        model.fields.clear(workspace.reconstructed_velocities);
        model.fields.clear(workspace.vorticities);
        model.fields.clear(workspace.vorticity_normals);
        kernels::launch_confinement_vjp(model.fields.stream, count, model.configuration.time_step, field::view(cache.vorticities), field::view(cache.vorticity_normals), parameters.vorticity_confinement.data(), field::view(workspace.confined_velocities), field::view(workspace.reconstructed_velocities), field::view(workspace.vorticities), field::view(workspace.vorticity_normals), parameter_adjoint.vorticity_confinement.data());
        kernels::launch_normal_vjp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.vorticities), device::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), field::view(cache.vorticity_normals), cache.vorticity_normalizers.values.data(), field::view(workspace.vorticity_normals), field::view(workspace.current_positions), field::view(workspace.vorticities));
        kernels::launch_vorticity_vjp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(cache.corrected_positions), field::view(cache.reconstructed_velocities), device::neighborhood(cache.neighborhood), field::view(workspace.vorticities), field::view(workspace.current_positions), field::view(workspace.reconstructed_velocities));
        kernels::launch_reconstruct_vjp(model.fields.stream, count, 1.0F / model.configuration.time_step, field::view(workspace.reconstructed_velocities), field::view(previous_state_adjoint.positions), field::view(workspace.current_positions));

        for (std::size_t segment = cache.checkpoints.size(); segment-- > 1uz;) {
            const StepCache::IterationCheckpoint& start = cache.checkpoints[segment - 1uz];
            const StepCache::IterationCheckpoint& end   = cache.checkpoints[segment];
            const std::uint32_t segment_length          = end.iteration - start.iteration;
            model.fields.copy(start.positions, workspace.position_history[0]);
            for (std::uint32_t local_iteration = 0u; local_iteration < segment_length; ++local_iteration) {
                IterationWorkspace& iteration = workspace.iteration_history[local_iteration];
                density.forward(model, cache.predicted_positions, workspace.position_history[local_iteration], parameters, cache.neighborhood, iteration.densities);
                kernels::launch_lambda_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(workspace.position_history[local_iteration]), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), iteration.densities.values.data(), parameters.relaxation.data(), field::view(iteration.gradient_sums), iteration.denominators.values.data(), iteration.lambdas.values.data());
                kernels::launch_correction_forward(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(workspace.position_history[local_iteration]), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), field::view(iteration.corrections));
                kernels::launch_project_forward(model.fields.stream, count, device::collision_box(model.configuration, state.step_index + 1u), field::view(workspace.position_history[local_iteration]), field::view(iteration.corrections), iteration.collision_masks.data(), field::view(workspace.position_history[local_iteration + 1u]));
            }
            for (std::uint32_t local_iteration = segment_length; local_iteration-- > 0u;) {
                IterationWorkspace& iteration = workspace.iteration_history[local_iteration];
                model.fields.clear(workspace.next_positions);
                model.fields.clear(workspace.corrections);
                model.fields.clear(workspace.lambdas);
                model.fields.clear(workspace.densities);
                kernels::launch_project_vjp(model.fields.stream, count, iteration.collision_masks.data(), field::view(workspace.current_positions), field::view(workspace.next_positions), field::view(workspace.corrections));
                kernels::launch_correction_vjp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(workspace.position_history[local_iteration]), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), field::view(workspace.corrections), field::view(workspace.next_positions), workspace.lambdas.values.data(), device::particle_parameter_adjoint(parameter_adjoint), parameter_adjoint.artificial_pressure_strength.data(), parameter_adjoint.artificial_pressure_exponent.data(), parameter_adjoint.artificial_pressure_radius.data());
                kernels::launch_lambda_vjp(model.fields.stream, count, model.configuration.support_radius, field::view(cache.predicted_positions), field::view(workspace.position_history[local_iteration]), device::particle_parameters(parameters), device::neighborhood(cache.neighborhood), device::boundary(model.boundary, cache.neighborhood), iteration.densities.values.data(), field::view(iteration.gradient_sums), iteration.denominators.values.data(), workspace.lambdas.values.data(), field::view(workspace.next_positions), workspace.densities.values.data(), device::particle_parameter_adjoint(parameter_adjoint), parameter_adjoint.relaxation.data());
                density.vjp(model, cache.predicted_positions, workspace.position_history[local_iteration], parameters, cache.neighborhood, workspace.densities, workspace.next_positions, parameter_adjoint);
                std::swap(workspace.current_positions, workspace.next_positions);
            }
        }
        kernels::launch_predict_vjp(model.fields.stream, count, model.configuration.time_step, field::view(workspace.current_positions), field::view(previous_state_adjoint.positions), field::view(previous_state_adjoint.velocities), field::view(control_adjoint.external_accelerations));
    }

    Solver::IterationWorkspace Solver::allocate_iteration_workspace(const meshfree::Model& model) const {
        return {
            .densities       = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .gradient_sums   = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .denominators    = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .lambdas         = model.fields.allocate_scalar_field<float>(model.configuration.particle_count),
            .corrections     = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
            .collision_masks = allocate_buffer<::cuda::device_buffer<std::uint32_t>>(model),
        };
    }
} // namespace physica::fluids::liquid::pbf
