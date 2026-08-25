module;

#include "../domain/interop.h"
#include "../neighborhood/interop.h"
#include "kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.particle.pbf;

import std;

namespace physica::fluids::liquid::particle {
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

    PBF::PBF(DomainConfiguration domain_configuration, Configuration next_configuration, const ExecutionMode mode, const ::cuda::stream_ref stream) : configuration(std::move(next_configuration)), domain(std::move(domain_configuration), stream), neighborhood_search(domain), forward_iteration(allocate_iteration()) {
        if (mode == ExecutionMode::differentiable) {
            Differentiation workspace{
                .current_position_tangent       = domain.allocate_vector_field(domain.configuration.particle_count),
                .next_position_tangent          = domain.allocate_vector_field(domain.configuration.particle_count),
                .density_tangent                = domain.allocate_scalar_field(domain.configuration.particle_count),
                .gradient_sum_tangent           = domain.allocate_vector_field(domain.configuration.particle_count),
                .denominator_tangent            = domain.allocate_scalar_field(domain.configuration.particle_count),
                .lambda_tangent                 = domain.allocate_scalar_field(domain.configuration.particle_count),
                .correction_tangent             = domain.allocate_vector_field(domain.configuration.particle_count),
                .reconstructed_velocity_tangent = domain.allocate_vector_field(domain.configuration.particle_count),
                .vorticity_tangent              = domain.allocate_vector_field(domain.configuration.particle_count),
                .vorticity_magnitude_tangent    = domain.allocate_scalar_field(domain.configuration.particle_count),
                .vorticity_normal_tangent       = domain.allocate_vector_field(domain.configuration.particle_count),
                .confined_velocity_tangent      = domain.allocate_vector_field(domain.configuration.particle_count),
                .current_position_adjoint       = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .next_position_adjoint          = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .correction_adjoint             = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .lambda_adjoint                 = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count),
                .density_adjoint                = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count),
                .reconstructed_velocity_adjoint = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .vorticity_adjoint              = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .vorticity_magnitude_adjoint    = domain.allocate_scalar_adjoint_field(domain.configuration.particle_count),
                .vorticity_normal_adjoint       = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
                .confined_velocity_adjoint      = domain.allocate_vector_adjoint_field(domain.configuration.particle_count),
            };
            workspace.segment_position_history.reserve(configuration.checkpoint_interval + 1u);
            for (std::uint32_t index = 0u; index <= configuration.checkpoint_interval; ++index) workspace.segment_position_history.push_back(domain.allocate_vector_field(domain.configuration.particle_count));
            workspace.segment_iteration_history.reserve(configuration.checkpoint_interval);
            for (std::uint32_t index = 0u; index < configuration.checkpoint_interval; ++index) workspace.segment_iteration_history.push_back(allocate_iteration());
            differentiation.emplace(std::move(workspace));
        }
    }

    PBF::IterationPrimal PBF::allocate_iteration() const {
        return {
            .densities       = domain.allocate_scalar_field(domain.configuration.particle_count),
            .gradient_sums   = domain.allocate_vector_field(domain.configuration.particle_count),
            .denominators    = domain.allocate_scalar_field(domain.configuration.particle_count),
            .lambdas         = domain.allocate_scalar_field(domain.configuration.particle_count),
            .corrections     = domain.allocate_vector_field(domain.configuration.particle_count),
            .collision_masks = allocate_buffer<::cuda::device_buffer<std::uint32_t>>(domain),
        };
    }

    PBF::State PBF::allocate_state() const {
        return {.particles = domain.allocate_particle_state()};
    }
    Control PBF::allocate_control() const {
        return domain.allocate_control();
    }

    PBF::Parameters PBF::allocate_parameters() const {
        return {
            .particles                    = domain.allocate_particle_parameters(),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<float>>(domain),
        };
    }

    PBF::StepCache PBF::allocate_step_cache() const {
        StepCache cache{
            .neighborhood             = neighborhood_search.allocate(),
            .predicted_positions      = domain.allocate_vector_field(domain.configuration.particle_count),
            .corrected_positions      = domain.allocate_vector_field(domain.configuration.particle_count),
            .reconstructed_velocities = domain.allocate_vector_field(domain.configuration.particle_count),
            .vorticities              = domain.allocate_vector_field(domain.configuration.particle_count),
            .vorticity_magnitudes     = domain.allocate_scalar_field(domain.configuration.particle_count),
            .vorticity_normals        = domain.allocate_vector_field(domain.configuration.particle_count),
            .vorticity_normalizers    = domain.allocate_scalar_field(domain.configuration.particle_count),
            .confined_velocities      = domain.allocate_vector_field(domain.configuration.particle_count),
        };
        cache.checkpoints.push_back({.iteration = 0u, .positions = domain.allocate_vector_field(domain.configuration.particle_count)});
        for (std::uint32_t iteration = configuration.checkpoint_interval; iteration < configuration.pressure_iterations; iteration += configuration.checkpoint_interval) cache.checkpoints.push_back({.iteration = iteration, .positions = domain.allocate_vector_field(domain.configuration.particle_count)});
        if (configuration.pressure_iterations != 0u) cache.checkpoints.push_back({.iteration = configuration.pressure_iterations, .positions = domain.allocate_vector_field(domain.configuration.particle_count)});
        return cache;
    }

    PBF::StateTangent PBF::allocate_state_tangent() const {
        return {.particles = domain.allocate_particle_state_tangent()};
    }
    ControlTangent PBF::allocate_control_tangent() const {
        return domain.allocate_control_tangent();
    }

    PBF::ParameterTangent PBF::allocate_parameter_tangent() const {
        ParameterTangent tangent{
            .particles                    = domain.allocate_particle_parameter_tangent(),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<float>>(domain),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<float>>(domain),
        };
        ::cuda::fill_bytes(domain.stream, tangent.relaxation, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.artificial_pressure_strength, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.artificial_pressure_exponent, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.artificial_pressure_radius, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.xsph_viscosity, 0u);
        ::cuda::fill_bytes(domain.stream, tangent.vorticity_confinement, 0u);
        return tangent;
    }

    PBF::StateAdjoint PBF::allocate_state_adjoint() const {
        return {.particles = domain.allocate_particle_state_adjoint()};
    }
    ControlAdjoint PBF::allocate_control_adjoint() const {
        return domain.allocate_control_adjoint();
    }

    PBF::ParameterAdjoint PBF::allocate_parameter_adjoint() const {
        ParameterAdjoint adjoint{
            .particles                    = domain.allocate_particle_parameter_adjoint(),
            .relaxation                   = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .artificial_pressure_strength = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .artificial_pressure_exponent = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .artificial_pressure_radius   = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .xsph_viscosity               = allocate_buffer<::cuda::device_buffer<double>>(domain),
            .vorticity_confinement        = allocate_buffer<::cuda::device_buffer<double>>(domain),
        };
        ::cuda::fill_bytes(domain.stream, adjoint.relaxation, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.artificial_pressure_strength, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.artificial_pressure_exponent, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.artificial_pressure_radius, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.xsph_viscosity, 0u);
        ::cuda::fill_bytes(domain.stream, adjoint.vorticity_confinement, 0u);
        return adjoint;
    }

    void PBF::copy_state(const State& source, State& destination) const {
        domain.copy(source.particles, destination.particles);
    }
    void PBF::copy_state_tangent(const StateTangent& source, StateTangent& destination) const {
        domain.copy(source.particles, destination.particles);
    }
    void PBF::copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const {
        domain.copy(source.particles, destination.particles);
    }
    void PBF::accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination) const {
        domain.accumulate(source.particles, destination.particles);
    }

    void PBF::forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache) {
        const std::uint32_t count = domain.configuration.particle_count;
        cuda_detail::pbf::launch_predict_forward(domain.stream, count, domain.configuration.time_step, domain.configuration.gravity.x, domain.configuration.gravity.y, domain.configuration.gravity.z, cuda_detail::vector(state.particles.positions), cuda_detail::vector(state.particles.velocities), cuda_detail::vector(control.external_accelerations), cuda_detail::vector(cache.predicted_positions));
        neighborhood_search.build(state.particles.step_index + 1u, cache.predicted_positions, cache.neighborhood);
        domain.copy(cache.predicted_positions, cache.corrected_positions);
        domain.copy(cache.corrected_positions, cache.checkpoints[0].positions);
        std::size_t checkpoint = 1uz;
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            density::pbf_forward(domain, cache.predicted_positions, cache.corrected_positions, parameters.particles, cache.neighborhood, forward_iteration.densities);
            cuda_detail::pbf::launch_lambda_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::parameters(parameters.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), forward_iteration.densities.values.data(), parameters.relaxation.data(), cuda_detail::vector(forward_iteration.gradient_sums), forward_iteration.denominators.values.data(), forward_iteration.lambdas.values.data());
            cuda_detail::pbf::launch_correction_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::parameters(parameters.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), forward_iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), cuda_detail::vector(forward_iteration.corrections));
            cuda_detail::pbf::launch_project_forward(domain.stream, count, collision_box(domain.configuration, state.particles.step_index + 1u), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(forward_iteration.corrections), forward_iteration.collision_masks.data(), cuda_detail::vector(cache.corrected_positions));
            if (checkpoint < cache.checkpoints.size() && cache.checkpoints[checkpoint].iteration == iteration + 1u) {
                domain.copy(cache.corrected_positions, cache.checkpoints[checkpoint].positions);
                ++checkpoint;
            }
        }
        cuda_detail::pbf::launch_reconstruct_forward(domain.stream, count, 1.0F / domain.configuration.time_step, cuda_detail::vector(state.particles.positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.reconstructed_velocities));
        cuda_detail::pbf::launch_vorticity_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.reconstructed_velocities), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::vector(cache.vorticities));
        cuda_detail::pbf::launch_normal_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.vorticities), cuda_detail::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), cuda_detail::vector(cache.vorticity_normals), cache.vorticity_normalizers.values.data());
        cuda_detail::pbf::launch_confinement_forward(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(cache.reconstructed_velocities), cuda_detail::vector(cache.vorticities), cuda_detail::vector(cache.vorticity_normals), parameters.vorticity_confinement.data(), cuda_detail::vector(cache.confined_velocities));
        cuda_detail::pbf::launch_xsph_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.confined_velocities), cuda_detail::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), cuda_detail::vector(next_state.particles.velocities));
        domain.copy(cache.corrected_positions, next_state.particles.positions);
        next_state.particles.step_index = state.particles.step_index + 1u;
    }

    void PBF::jvp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent) {
        Differentiation& workspace = *differentiation;
        const std::uint32_t count  = domain.configuration.particle_count;
        cuda_detail::pbf::launch_predict_jvp(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(state_tangent.particles.positions), cuda_detail::vector(state_tangent.particles.velocities), cuda_detail::vector(control_tangent.external_accelerations), cuda_detail::vector(workspace.current_position_tangent));
        domain.copy(cache.predicted_positions, workspace.segment_position_history.front());
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            density::pbf_forward(domain, cache.predicted_positions, workspace.segment_position_history.front(), parameters.particles, cache.neighborhood, forward_iteration.densities);
            density::pbf_jvp(domain, cache.predicted_positions, workspace.segment_position_history.front(), workspace.current_position_tangent, parameters.particles, parameter_tangent.particles, cache.neighborhood, workspace.density_tangent);
            cuda_detail::pbf::launch_lambda_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.segment_position_history.front()), cuda_detail::parameters(parameters.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), forward_iteration.densities.values.data(), parameters.relaxation.data(), cuda_detail::vector(forward_iteration.gradient_sums), forward_iteration.denominators.values.data(), forward_iteration.lambdas.values.data());
            cuda_detail::pbf::launch_lambda_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.segment_position_history.front()), cuda_detail::vector(workspace.current_position_tangent), cuda_detail::parameters(parameters.particles), cuda_detail::parameter_tangent(parameter_tangent.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), forward_iteration.densities.values.data(), workspace.density_tangent.values.data(), parameters.relaxation.data(), parameter_tangent.relaxation.data(), cuda_detail::vector(workspace.gradient_sum_tangent), workspace.denominator_tangent.values.data(), workspace.lambda_tangent.values.data());
            cuda_detail::pbf::launch_correction_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.segment_position_history.front()), cuda_detail::parameters(parameters.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), forward_iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), cuda_detail::vector(forward_iteration.corrections));
            cuda_detail::pbf::launch_correction_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.segment_position_history.front()), cuda_detail::vector(workspace.current_position_tangent), cuda_detail::parameters(parameters.particles), cuda_detail::parameter_tangent(parameter_tangent.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), forward_iteration.lambdas.values.data(), workspace.lambda_tangent.values.data(), parameters.artificial_pressure_strength.data(), parameter_tangent.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameter_tangent.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), parameter_tangent.artificial_pressure_radius.data(), cuda_detail::vector(workspace.correction_tangent));
            cuda_detail::pbf::launch_project_forward(domain.stream, count, collision_box(domain.configuration, state.particles.step_index + 1u), cuda_detail::vector(workspace.segment_position_history.front()), cuda_detail::vector(forward_iteration.corrections), forward_iteration.collision_masks.data(), cuda_detail::vector(workspace.segment_position_history.front()));
            cuda_detail::pbf::launch_project_jvp(domain.stream, count, forward_iteration.collision_masks.data(), cuda_detail::vector(workspace.current_position_tangent), cuda_detail::vector(workspace.correction_tangent), cuda_detail::vector(workspace.next_position_tangent));
            std::swap(workspace.current_position_tangent, workspace.next_position_tangent);
        }
        cuda_detail::pbf::launch_reconstruct_jvp(domain.stream, count, 1.0F / domain.configuration.time_step, cuda_detail::vector(state_tangent.particles.positions), cuda_detail::vector(workspace.current_position_tangent), cuda_detail::vector(workspace.reconstructed_velocity_tangent));
        cuda_detail::pbf::launch_vorticity_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.reconstructed_velocities), cuda_detail::vector(workspace.current_position_tangent), cuda_detail::vector(workspace.reconstructed_velocity_tangent), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::vector(workspace.vorticity_tangent));
        cuda_detail::pbf::launch_normal_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.vorticities), cuda_detail::vector(workspace.current_position_tangent), cuda_detail::vector(workspace.vorticity_tangent), cuda_detail::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), cuda_detail::vector(cache.vorticity_normals), cache.vorticity_normalizers.values.data(), workspace.vorticity_magnitude_tangent.values.data(), cuda_detail::vector(workspace.vorticity_normal_tangent));
        cuda_detail::pbf::launch_confinement_jvp(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(cache.vorticities), cuda_detail::vector(cache.vorticity_normals), parameters.vorticity_confinement.data(), cuda_detail::vector(workspace.reconstructed_velocity_tangent), cuda_detail::vector(workspace.vorticity_tangent), cuda_detail::vector(workspace.vorticity_normal_tangent), parameter_tangent.vorticity_confinement.data(), cuda_detail::vector(workspace.confined_velocity_tangent));
        cuda_detail::pbf::launch_xsph_jvp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.confined_velocities), cuda_detail::vector(workspace.current_position_tangent), cuda_detail::vector(workspace.confined_velocity_tangent), cuda_detail::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), parameter_tangent.xsph_viscosity.data(), cuda_detail::vector(next_state_tangent.particles.velocities));
        domain.copy(workspace.current_position_tangent, next_state_tangent.particles.positions);
    }

    void PBF::vjp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint) {
        Differentiation& workspace = *differentiation;
        const std::uint32_t count  = domain.configuration.particle_count;
        domain.copy(next_state_adjoint.particles.positions, workspace.current_position_adjoint);
        domain.clear(workspace.confined_velocity_adjoint);
        cuda_detail::pbf::launch_xsph_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.confined_velocities), cuda_detail::neighborhood(cache.neighborhood), parameters.xsph_viscosity.data(), cuda_detail::adjoint_vector(next_state_adjoint.particles.velocities), cuda_detail::adjoint_vector(workspace.current_position_adjoint), cuda_detail::adjoint_vector(workspace.confined_velocity_adjoint), parameter_adjoint.xsph_viscosity.data());

        domain.clear(workspace.reconstructed_velocity_adjoint);
        domain.clear(workspace.vorticity_adjoint);
        domain.clear(workspace.vorticity_normal_adjoint);
        cuda_detail::pbf::launch_confinement_vjp(domain.stream, count, domain.configuration.time_step, cuda_detail::vector(cache.vorticities), cuda_detail::vector(cache.vorticity_normals), parameters.vorticity_confinement.data(), cuda_detail::adjoint_vector(workspace.confined_velocity_adjoint), cuda_detail::adjoint_vector(workspace.reconstructed_velocity_adjoint), cuda_detail::adjoint_vector(workspace.vorticity_adjoint), cuda_detail::adjoint_vector(workspace.vorticity_normal_adjoint), parameter_adjoint.vorticity_confinement.data());
        cuda_detail::pbf::launch_normal_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.vorticities), cuda_detail::neighborhood(cache.neighborhood), cache.vorticity_magnitudes.values.data(), cuda_detail::vector(cache.vorticity_normals), cache.vorticity_normalizers.values.data(), cuda_detail::adjoint_vector(workspace.vorticity_normal_adjoint), cuda_detail::adjoint_vector(workspace.current_position_adjoint), cuda_detail::adjoint_vector(workspace.vorticity_adjoint));
        cuda_detail::pbf::launch_vorticity_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(cache.corrected_positions), cuda_detail::vector(cache.reconstructed_velocities), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::adjoint_vector(workspace.vorticity_adjoint), cuda_detail::adjoint_vector(workspace.current_position_adjoint), cuda_detail::adjoint_vector(workspace.reconstructed_velocity_adjoint));
        cuda_detail::pbf::launch_reconstruct_vjp(domain.stream, count, 1.0F / domain.configuration.time_step, cuda_detail::adjoint_vector(workspace.reconstructed_velocity_adjoint), cuda_detail::adjoint_vector(previous_state_adjoint.particles.positions), cuda_detail::adjoint_vector(workspace.current_position_adjoint));

        for (std::size_t segment = cache.checkpoints.size(); segment-- > 1uz;) {
            const StepCache::IterationCheckpoint& start = cache.checkpoints[segment - 1uz];
            const StepCache::IterationCheckpoint& end   = cache.checkpoints[segment];
            const std::uint32_t segment_length          = end.iteration - start.iteration;
            domain.copy(start.positions, workspace.segment_position_history[0]);
            for (std::uint32_t local_iteration = 0u; local_iteration < segment_length; ++local_iteration) {
                IterationPrimal& iteration = workspace.segment_iteration_history[local_iteration];
                density::pbf_forward(domain, cache.predicted_positions, workspace.segment_position_history[local_iteration], parameters.particles, cache.neighborhood, iteration.densities);
                cuda_detail::pbf::launch_lambda_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.segment_position_history[local_iteration]), cuda_detail::parameters(parameters.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), iteration.densities.values.data(), parameters.relaxation.data(), cuda_detail::vector(iteration.gradient_sums), iteration.denominators.values.data(), iteration.lambdas.values.data());
                cuda_detail::pbf::launch_correction_forward(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.segment_position_history[local_iteration]), cuda_detail::parameters(parameters.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), cuda_detail::vector(iteration.corrections));
                cuda_detail::pbf::launch_project_forward(domain.stream, count, collision_box(domain.configuration, state.particles.step_index + 1u), cuda_detail::vector(workspace.segment_position_history[local_iteration]), cuda_detail::vector(iteration.corrections), iteration.collision_masks.data(), cuda_detail::vector(workspace.segment_position_history[local_iteration + 1u]));
            }
            for (std::uint32_t local_iteration = segment_length; local_iteration-- > 0u;) {
                IterationPrimal& iteration = workspace.segment_iteration_history[local_iteration];
                domain.clear(workspace.next_position_adjoint);
                domain.clear(workspace.correction_adjoint);
                domain.clear(workspace.lambda_adjoint);
                domain.clear(workspace.density_adjoint);
                cuda_detail::pbf::launch_project_vjp(domain.stream, count, iteration.collision_masks.data(), cuda_detail::adjoint_vector(workspace.current_position_adjoint), cuda_detail::adjoint_vector(workspace.next_position_adjoint), cuda_detail::adjoint_vector(workspace.correction_adjoint));
                cuda_detail::pbf::launch_correction_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.segment_position_history[local_iteration]), cuda_detail::parameters(parameters.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), iteration.lambdas.values.data(), parameters.artificial_pressure_strength.data(), parameters.artificial_pressure_exponent.data(), parameters.artificial_pressure_radius.data(), cuda_detail::adjoint_vector(workspace.correction_adjoint), cuda_detail::adjoint_vector(workspace.next_position_adjoint), workspace.lambda_adjoint.values.data(), cuda_detail::parameter_adjoint(parameter_adjoint.particles), parameter_adjoint.artificial_pressure_strength.data(), parameter_adjoint.artificial_pressure_exponent.data(), parameter_adjoint.artificial_pressure_radius.data());
                cuda_detail::pbf::launch_lambda_vjp(domain.stream, count, domain.configuration.support_radius, cuda_detail::vector(cache.predicted_positions), cuda_detail::vector(workspace.segment_position_history[local_iteration]), cuda_detail::parameters(parameters.particles), cuda_detail::neighborhood(cache.neighborhood), cuda_detail::boundary(domain.boundary, cache.neighborhood), iteration.densities.values.data(), cuda_detail::vector(iteration.gradient_sums), iteration.denominators.values.data(), workspace.lambda_adjoint.values.data(), cuda_detail::adjoint_vector(workspace.next_position_adjoint), workspace.density_adjoint.values.data(), cuda_detail::parameter_adjoint(parameter_adjoint.particles), parameter_adjoint.relaxation.data());
                density::pbf_vjp(domain, cache.predicted_positions, workspace.segment_position_history[local_iteration], parameters.particles, cache.neighborhood, workspace.density_adjoint, workspace.next_position_adjoint, parameter_adjoint.particles);
                std::swap(workspace.current_position_adjoint, workspace.next_position_adjoint);
            }
        }
        cuda_detail::pbf::launch_predict_vjp(domain.stream, count, domain.configuration.time_step, cuda_detail::adjoint_vector(workspace.current_position_adjoint), cuda_detail::adjoint_vector(previous_state_adjoint.particles.positions), cuda_detail::adjoint_vector(previous_state_adjoint.particles.velocities), cuda_detail::adjoint_vector(control_adjoint.external_accelerations));
    }
} // namespace physica::fluids::liquid::particle
