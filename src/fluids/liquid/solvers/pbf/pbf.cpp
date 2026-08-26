module;

#include "../../detail/cuda/interop.h"
#include "pbf-kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.pbf;

import std;

namespace physica::fluids::liquid::pbf {
    namespace {
        cuda_detail::Box collision_box(const DomainConfiguration& configuration, const std::uint64_t step_index) {
            const float time = static_cast<float>(step_index) * configuration.time_step;
            return {
                .minimum_x  = configuration.boundary.minimum.x + time * configuration.boundary.velocity.x + configuration.particle_radius,
                .minimum_y  = configuration.boundary.minimum.y + time * configuration.boundary.velocity.y + configuration.particle_radius,
                .minimum_z  = configuration.boundary.minimum.z + time * configuration.boundary.velocity.z + configuration.particle_radius,
                .maximum_x  = configuration.boundary.maximum.x + time * configuration.boundary.velocity.x - configuration.particle_radius,
                .maximum_y  = configuration.boundary.maximum.y + time * configuration.boundary.velocity.y - configuration.particle_radius,
                .maximum_z  = configuration.boundary.maximum.z + time * configuration.boundary.velocity.z - configuration.particle_radius,
                .velocity_x = configuration.boundary.velocity.x,
                .velocity_y = configuration.boundary.velocity.y,
                .velocity_z = configuration.boundary.velocity.z,
                .no_slip    = configuration.boundary.no_slip ? 1u : 0u,
            };
        }

        template <class Buffer>
        Buffer allocate_buffer(const Domain& domain) {
            return Buffer{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), domain.configuration.particle_count, ::cuda::no_init};
        }
    } // namespace

    Solver::Solver(Configuration next_configuration) : configuration(std::move(next_configuration)), density({}) {}

    Solver::State Solver::allocate_state(const Domain& domain) const {
        State state{
            .positions  = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .velocities = domain.allocate_vector_field<float>(domain.configuration.particle_count),
        };
        domain.clear(state.positions);
        domain.clear(state.velocities);
        return state;
    }

    Solver::Control Solver::allocate_control(const Domain& domain) const {
        Control control{.external_accelerations = domain.allocate_vector_field<float>(domain.configuration.particle_count)};
        domain.clear(control.external_accelerations);
        return control;
    }

    Solver::Parameters Solver::allocate_parameters(const Domain& domain) const {
        return {
            .masses                       = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .rest_densities               = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .viscosities                  = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .surface_tensions             = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<float>>(domain),
        };
    }

    Solver::StepCache Solver::allocate_step_cache(const Domain& domain) const {
        StepCache cache{
            .neighborhood             = neighborhood.allocate_cache(domain),
            .predicted_positions      = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .corrected_positions      = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .reconstructed_velocities = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .vorticities              = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .vorticity_magnitudes     = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .vorticity_normals        = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .vorticity_normalizers    = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .confined_velocities      = domain.allocate_vector_field<float>(domain.configuration.particle_count),
        };
        cache.checkpoints.push_back({.iteration = 0u, .positions = domain.allocate_vector_field<float>(domain.configuration.particle_count)});
        for (std::uint32_t iteration = configuration.checkpoint_interval; iteration < configuration.pressure_iterations; iteration += configuration.checkpoint_interval) cache.checkpoints.push_back({.iteration = iteration, .positions = domain.allocate_vector_field<float>(domain.configuration.particle_count)});
        if (configuration.pressure_iterations != 0u) cache.checkpoints.push_back({.iteration = configuration.pressure_iterations, .positions = domain.allocate_vector_field<float>(domain.configuration.particle_count)});
        return cache;
    }

    Solver::Workspace Solver::allocate_workspace(const Domain& domain) const {
        return {
            .neighborhood = neighborhood.allocate_workspace(domain),
            .iteration    = allocate_iteration_workspace(domain),
        };
    }

    Solver::StateTangent Solver::allocate_state_tangent(const Domain& domain) const {
        StateTangent tangent{
            .positions  = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .velocities = domain.allocate_vector_field<float>(domain.configuration.particle_count),
        };
        domain.clear(tangent.positions);
        domain.clear(tangent.velocities);
        return tangent;
    }

    Solver::ControlTangent Solver::allocate_control_tangent(const Domain& domain) const {
        ControlTangent tangent{.external_accelerations = domain.allocate_vector_field<float>(domain.configuration.particle_count)};
        domain.clear(tangent.external_accelerations);
        return tangent;
    }

    Solver::ParameterTangent Solver::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent tangent{
            .masses                       = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .rest_densities               = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .viscosities                  = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .surface_tensions             = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<float>>(domain),
        };
        ::cuda::fill_bytes(domain.stream, tangent.masses, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.rest_densities, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.viscosities, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.surface_tensions, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.relaxation, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.artificial_pressure_strength, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.artificial_pressure_exponent, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.artificial_pressure_radius, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.xsph_viscosity, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.vorticity_confinement, 0u);
        return tangent;
    }

    Solver::TangentWorkspace Solver::allocate_tangent_workspace(const Domain& domain) const {
        return {
            .positions                = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .iteration                = allocate_iteration_workspace(domain),
            .current_positions        = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .next_positions           = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .densities                = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .gradient_sums            = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .denominators             = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .lambdas                  = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .corrections              = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .reconstructed_velocities = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .vorticities              = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .vorticity_magnitudes     = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .vorticity_normals        = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .confined_velocities      = domain.allocate_vector_field<float>(domain.configuration.particle_count),
        };
    }

    Solver::StateAdjoint Solver::allocate_state_adjoint(const Domain& domain) const {
        StateAdjoint adjoint{
            .positions  = domain.allocate_vector_field<double>(domain.configuration.particle_count),
            .velocities = domain.allocate_vector_field<double>(domain.configuration.particle_count),
        };
        domain.clear(adjoint.positions);
        domain.clear(adjoint.velocities);
        return adjoint;
    }

    Solver::ControlAdjoint Solver::allocate_control_adjoint(const Domain& domain) const {
        ControlAdjoint adjoint{.external_accelerations = domain.allocate_vector_field<double>(domain.configuration.particle_count)};
        domain.clear(adjoint.external_accelerations);
        return adjoint;
    }

    Solver::ParameterAdjoint Solver::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint adjoint{
            .masses                       = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .rest_densities               = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .viscosities                  = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .surface_tensions             = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<double>>(domain),
        };
        ::cuda::fill_bytes(domain.stream, adjoint.masses, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.rest_densities, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.viscosities, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.surface_tensions, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.relaxation, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.artificial_pressure_strength, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.artificial_pressure_exponent, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.artificial_pressure_radius, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.xsph_viscosity, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.vorticity_confinement, 0u);
        return adjoint;
    }

    Solver::AdjointWorkspace Solver::allocate_adjoint_workspace(const Domain& domain) const {
        AdjointWorkspace workspace{
            .current_positions        = domain.allocate_vector_field<double>(domain.configuration.particle_count),
            .next_positions           = domain.allocate_vector_field<double>(domain.configuration.particle_count),
            .corrections              = domain.allocate_vector_field<double>(domain.configuration.particle_count),
            .lambdas                  = domain.allocate_scalar_field<double>(domain.configuration.particle_count),
            .densities                = domain.allocate_scalar_field<double>(domain.configuration.particle_count),
            .reconstructed_velocities = domain.allocate_vector_field<double>(domain.configuration.particle_count),
            .vorticities              = domain.allocate_vector_field<double>(domain.configuration.particle_count),
            .vorticity_normals        = domain.allocate_vector_field<double>(domain.configuration.particle_count),
            .confined_velocities      = domain.allocate_vector_field<double>(domain.configuration.particle_count),
        };
        workspace.position_history.reserve(configuration.checkpoint_interval + 1u);
        for (std::uint32_t index = 0u; index <= configuration.checkpoint_interval; ++index) workspace.position_history.push_back(domain.allocate_vector_field<float>(domain.configuration.particle_count));
        workspace.iteration_history.reserve(configuration.checkpoint_interval);
        for (std::uint32_t index = 0u; index < configuration.checkpoint_interval; ++index) workspace.iteration_history.push_back(allocate_iteration_workspace(domain));
        return workspace;
    }

    void Solver::copy_state(const Domain& domain, const State& source, State& destination) const {
        domain.copy(source.positions, destination.positions);
        domain.copy(source.velocities, destination.velocities);
        destination.step_index = source.step_index;
    }

    void Solver::copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const {
        domain.copy(source.positions, destination.positions);
        domain.copy(source.velocities, destination.velocities);
    }

    void Solver::copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const {
        domain.copy(source.positions, destination.positions);
        domain.copy(source.velocities, destination.velocities);
    }

    void Solver::accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const {
        domain.accumulate(source.positions, destination.positions);
        domain.accumulate(source.velocities, destination.velocities);
    }

    void Solver::forward(const Domain& domain, const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
        const std::uint32_t count = domain.configuration.particle_count;
        cuda_detail::pbf::launch_predict_forward(domain.stream, count, domain.configuration.time_step, configuration.gravity.x, configuration.gravity.y, configuration.gravity.z, cuda_detail::vector(state.positions), cuda_detail::vector(state.velocities), cuda_detail::vector(control.external_accelerations), cuda_detail::vector(cache.predicted_positions));
        neighborhood.build(domain, state.step_index + 1u, cache.predicted_positions, cache.neighborhood, workspace.neighborhood);
        domain.copy(cache.predicted_positions, cache.corrected_positions);
        domain.copy(cache.corrected_positions, cache.checkpoints[0].positions);
        std::size_t checkpoint = 1uz;
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            density.forward(domain, cache.predicted_positions, cache.corrected_positions, parameters, cache.neighborhood, workspace.iteration.densities);
            cuda_detail::pbf::launch_lambda_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), parameters.relaxation.data(), cuda_detail::vector(workspace.iteration.gradient_sums), workspace.iteration.denominators.values.data(), workspace.iteration.lambdas.values.data());
            cuda_detail::pbf::launch_correction_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), cuda_detail::vector(workspace.iteration.corrections));
            cuda_detail::pbf::launch_project_forward(domain.stream, count, collision_box(domain.configuration, state.step_index + 1u), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(workspace.iteration.corrections), workspace.iteration.collision_masks.data(), cuda_detail::vector(cache.corrected_positions));
            if (checkpoint < cache.checkpoints.size() && cache.checkpoints[checkpoint].iteration == iteration + 1u) {
                domain.copy(cache.corrected_positions, cache.checkpoints[checkpoint].positions);
                ++checkpoint;
            }
        }
        cuda_detail::pbf::launch_reconstruct_forward(domain.stream, count, 1.0F / domain.configuration.time_step, cuda_detail::vector(state.positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.reconstructed_velocities));
        cuda_detail::pbf::launch_vorticity_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.reconstructed_velocities), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::vector(cache.vorticities));
        cuda_detail::pbf::launch_normal_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.vorticities), cuda_detail::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), cuda_detail::vector(cache.vorticity_normals), cache.vorticity_normalizers.values.data());
        cuda_detail::pbf::launch_confinement_forward(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(cache.reconstructed_velocities), cuda_detail::vector(cache.vorticities), cuda_detail::vector(cache.vorticity_normals), parameters.vorticity_confinement.data(), cuda_detail::vector(cache.confined_velocities));
        cuda_detail::pbf::launch_xsph_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.confined_velocities), cuda_detail::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), cuda_detail::vector(next_state.velocities));
        domain.copy(cache.corrected_positions, next_state.positions);
        next_state.step_index = state.step_index + 1u;
    }

    void Solver::jvp(const Domain& domain, const State& state, const Parameters& parameters, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const {
        const std::uint32_t count = domain.configuration.particle_count;
        cuda_detail::pbf::launch_predict_jvp(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(state_tangent.positions), cuda_detail::vector(state_tangent.velocities), cuda_detail::vector(control_tangent.external_accelerations), cuda_detail::vector(workspace.current_positions));
        domain.copy(cache.predicted_positions, workspace.positions);
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            density.forward(domain, cache.predicted_positions, workspace.positions, parameters, cache.neighborhood, workspace.iteration.densities);
            density.jvp(domain, cache.predicted_positions, workspace.positions, workspace.current_positions, parameters, parameter_tangent, cache.neighborhood, workspace.densities);
            cuda_detail::pbf::launch_lambda_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), parameters.relaxation.data(), cuda_detail::vector(workspace.iteration.gradient_sums), workspace.iteration.denominators.values.data(), workspace.iteration.lambdas.values.data());
            cuda_detail::pbf::launch_lambda_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.positions), cuda_detail::vector(workspace.current_positions), cuda_detail::parameters(parameters), cuda_detail::parameter_tangent(parameter_tangent), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), workspace.iteration.densities.values.data(), workspace.densities.values.data(), parameters.relaxation.data(), parameter_tangent.relaxation.data(), cuda_detail::vector(workspace.gradient_sums), workspace.denominators.values.data(), workspace.lambdas.values.data());
            cuda_detail::pbf::launch_correction_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), cuda_detail::vector(workspace.iteration.corrections));
            cuda_detail::pbf::launch_correction_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.positions), cuda_detail::vector(workspace.current_positions), cuda_detail::parameters(parameters), cuda_detail::parameter_tangent(parameter_tangent), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), workspace.iteration.lambdas.values.data(), workspace.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameter_tangent.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameter_tangent.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), parameter_tangent.artificial_pressure_radius.data(), cuda_detail::vector(workspace.corrections));
            cuda_detail::pbf::launch_project_forward(domain.stream, count, collision_box(domain.configuration, state.step_index + 1u), cuda_detail::vector(workspace.positions), cuda_detail::vector(workspace.iteration.corrections), workspace.iteration.collision_masks.data(), cuda_detail::vector(workspace.positions));
            cuda_detail::pbf::launch_project_jvp(domain.stream, count, workspace.iteration.collision_masks.data(), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.corrections), cuda_detail::vector(workspace.next_positions));
            std::swap(workspace.current_positions, workspace.next_positions);
        }
        cuda_detail::pbf::launch_reconstruct_jvp(domain.stream, count, 1.0F / domain.configuration.time_step, cuda_detail::vector(state_tangent.positions), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.reconstructed_velocities));
        cuda_detail::pbf::launch_vorticity_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.reconstructed_velocities), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.reconstructed_velocities), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::vector(workspace.vorticities));
        cuda_detail::pbf::launch_normal_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.vorticities), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.vorticities), cuda_detail::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), cuda_detail::vector(cache.vorticity_normals), cache.vorticity_normalizers.values.data(), workspace.vorticity_magnitudes.values.data(), cuda_detail::vector(workspace.vorticity_normals));
        cuda_detail::pbf::launch_confinement_jvp(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(cache.vorticities), cuda_detail::vector(cache.vorticity_normals), parameters.vorticity_confinement.data(), cuda_detail::vector(workspace.reconstructed_velocities), cuda_detail::vector(workspace.vorticities), cuda_detail::vector(workspace.vorticity_normals), parameter_tangent.vorticity_confinement.data(), cuda_detail::vector(workspace.confined_velocities));
        cuda_detail::pbf::launch_xsph_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.confined_velocities), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.confined_velocities), cuda_detail::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), parameter_tangent.xsph_viscosity.data(), cuda_detail::vector(next_state_tangent.velocities));
        domain.copy(workspace.current_positions, next_state_tangent.positions);
    }

    void Solver::vjp(const Domain& domain, const State& state, const Parameters& parameters, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
        const std::uint32_t count = domain.configuration.particle_count;
        domain.copy(next_state_adjoint.positions, workspace.current_positions);
        domain.clear(workspace.confined_velocities);
        cuda_detail::pbf::launch_xsph_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.confined_velocities), cuda_detail::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), cuda_detail::vector(next_state_adjoint.velocities), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.confined_velocities), parameter_adjoint.xsph_viscosity.data());

        domain.clear(workspace.reconstructed_velocities);
        domain.clear(workspace.vorticities);
        domain.clear(workspace.vorticity_normals);
        cuda_detail::pbf::launch_confinement_vjp(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(cache.vorticities), cuda_detail::vector(cache.vorticity_normals), parameters.vorticity_confinement.data(), cuda_detail::vector(workspace.confined_velocities), cuda_detail::vector(workspace.reconstructed_velocities), cuda_detail::vector(workspace.vorticities), cuda_detail::vector(workspace.vorticity_normals), parameter_adjoint.vorticity_confinement.data());
        cuda_detail::pbf::launch_normal_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.vorticities), cuda_detail::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), cuda_detail::vector(cache.vorticity_normals), cache.vorticity_normalizers.values.data(), cuda_detail::vector(workspace.vorticity_normals), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.vorticities));
        cuda_detail::pbf::launch_vorticity_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.reconstructed_velocities), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::vector(workspace.vorticities), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.reconstructed_velocities));
        cuda_detail::pbf::launch_reconstruct_vjp(domain.stream, count, 1.0F / domain.configuration.time_step, cuda_detail::vector(workspace.reconstructed_velocities), cuda_detail::vector(previous_state_adjoint.positions), cuda_detail::vector(workspace.current_positions));

        for (std::size_t segment = cache.checkpoints.size(); segment-- > 1uz;) {
            const StepCache::IterationCheckpoint& start = cache.checkpoints[segment - 1uz];
            const StepCache::IterationCheckpoint& end   = cache.checkpoints[segment];
            const std::uint32_t segment_length          = end.iteration - start.iteration;
            domain.copy(start.positions, workspace.position_history[0]);
            for (std::uint32_t local_iteration = 0u; local_iteration < segment_length; ++local_iteration) {
                IterationWorkspace& iteration = workspace.iteration_history[local_iteration];
                density.forward(domain, cache.predicted_positions, workspace.position_history[local_iteration], parameters, cache.neighborhood, iteration.densities);
                cuda_detail::pbf::launch_lambda_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.position_history[local_iteration]), cuda_detail::parameters(parameters), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), iteration.densities.values.data(), parameters.relaxation.data(), cuda_detail::vector(iteration.gradient_sums), iteration.denominators.values.data(), iteration.lambdas.values.data());
                cuda_detail::pbf::launch_correction_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.position_history[local_iteration]), cuda_detail::parameters(parameters), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), cuda_detail::vector(iteration.corrections));
                cuda_detail::pbf::launch_project_forward(domain.stream, count, collision_box(domain.configuration, state.step_index + 1u), cuda_detail::vector(workspace.position_history[local_iteration]), cuda_detail::vector(iteration.corrections), iteration.collision_masks.data(), cuda_detail::vector(workspace.position_history[local_iteration + 1u]));
            }
            for (std::uint32_t local_iteration = segment_length; local_iteration-- > 0u;) {
                IterationWorkspace& iteration = workspace.iteration_history[local_iteration];
                domain.clear(workspace.next_positions);
                domain.clear(workspace.corrections);
                domain.clear(workspace.lambdas);
                domain.clear(workspace.densities);
                cuda_detail::pbf::launch_project_vjp(domain.stream, count, iteration.collision_masks.data(), cuda_detail::vector(workspace.current_positions), cuda_detail::vector(workspace.next_positions), cuda_detail::vector(workspace.corrections));
                cuda_detail::pbf::launch_correction_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.position_history[local_iteration]), cuda_detail::parameters(parameters), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), cuda_detail::vector(workspace.corrections), cuda_detail::vector(workspace.next_positions), workspace.lambdas.values.data(), cuda_detail::parameter_adjoint(parameter_adjoint), parameter_adjoint.artificial_pressure_strength.data(), parameter_adjoint.artificial_pressure_exponent.data(), parameter_adjoint.artificial_pressure_radius.data());
                cuda_detail::pbf::launch_lambda_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.position_history[local_iteration]), cuda_detail::parameters(parameters), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), iteration.densities.values.data(), cuda_detail::vector(iteration.gradient_sums), iteration.denominators.values.data(), workspace.lambdas.values.data(), cuda_detail::vector(workspace.next_positions), workspace.densities.values.data(), cuda_detail::parameter_adjoint(parameter_adjoint), parameter_adjoint.relaxation.data());
                density.vjp(domain, cache.predicted_positions, workspace.position_history[local_iteration], parameters, cache.neighborhood, workspace.densities, workspace.next_positions, parameter_adjoint);
                std::swap(workspace.current_positions, workspace.next_positions);
            }
        }
        cuda_detail::pbf::launch_predict_vjp(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(workspace.current_positions), cuda_detail::vector(previous_state_adjoint.positions), cuda_detail::vector(previous_state_adjoint.velocities), cuda_detail::vector(control_adjoint.external_accelerations));
    }

    Solver::IterationWorkspace Solver::allocate_iteration_workspace(const Domain& domain) const {
        return {
            .densities       = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .gradient_sums   = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .denominators    = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .lambdas         = domain.allocate_scalar_field<float>(domain.configuration.particle_count),
            .corrections     = domain.allocate_vector_field<float>(domain.configuration.particle_count),
            .collision_masks = allocate_buffer<::cuda::device_buffer<std::uint32_t>>(domain),
        };
    }
} // namespace physica::fluids::liquid::pbf
