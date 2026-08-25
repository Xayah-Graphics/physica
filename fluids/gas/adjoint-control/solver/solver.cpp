module;

#include "../domain/interop.h"
#include "kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.adjoint_control.solver;

import std;

namespace physica::fluids::gas::adjoint_control {
    namespace {
        ::cuda::device_buffer<double> allocate_scalar_value(const Domain& domain) {
            return ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), 1u, ::cuda::no_init};
        }
    } // namespace

    Solver::Solver(const Domain& domain, SolverConfiguration next_configuration)
        : configuration(std::move(next_configuration)),
          raw_advected_velocity(domain.allocate_staggered_vector_field()),
          diffusion_first(domain.allocate_staggered_vector_field()),
          diffusion_second(domain.allocate_staggered_vector_field()),
          tangent{
              .controlled_velocity = domain.allocate_staggered_vector_field(),
              .raw_advected_velocity = domain.allocate_staggered_vector_field(),
              .advected_velocity = domain.allocate_staggered_vector_field(),
              .advected_density = domain.allocate_scalar_field(),
              .transported_density = domain.allocate_scalar_field(),
              .diffusion_first = domain.allocate_staggered_vector_field(),
              .diffusion_second = domain.allocate_staggered_vector_field(),
              .diffused_velocity = domain.allocate_staggered_vector_field(),
              .heat_force = domain.allocate_centered_vector_field(),
              .heated_velocity = domain.allocate_staggered_vector_field(),
              .constrained_velocity = domain.allocate_staggered_vector_field(),
              .pressure_rhs = domain.allocate_scalar_field(),
              .pressure = domain.allocate_scalar_field(),
              .projected_velocity = domain.allocate_staggered_vector_field(),
              .input_mass = allocate_scalar_value(domain),
              .advected_mass = allocate_scalar_value(domain),
          },
          adjoint{
              .transported_density = domain.allocate_scalar_adjoint_field(),
              .advected_density = domain.allocate_scalar_adjoint_field(),
              .projected_velocity = domain.allocate_staggered_vector_adjoint_field(),
              .pressure = domain.allocate_scalar_adjoint_field(),
              .pressure_rhs = domain.allocate_scalar_adjoint_field(),
              .constrained_velocity = domain.allocate_staggered_vector_adjoint_field(),
              .heated_velocity = domain.allocate_staggered_vector_adjoint_field(),
              .heat_force = domain.allocate_centered_vector_adjoint_field(),
              .diffused_velocity = domain.allocate_staggered_vector_adjoint_field(),
              .advected_velocity = domain.allocate_staggered_vector_adjoint_field(),
              .raw_advected_velocity = domain.allocate_staggered_vector_adjoint_field(),
              .controlled_velocity = domain.allocate_staggered_vector_adjoint_field(),
              .density_dot = allocate_scalar_value(domain),
          } {}

    State Solver::allocate_state(const Domain& domain) const { return {.density = domain.allocate_scalar_field(), .velocity = domain.allocate_staggered_vector_field()}; }
    StateTangent Solver::allocate_state_tangent(const Domain& domain) const { return {.density = domain.allocate_scalar_field(), .velocity = domain.allocate_staggered_vector_field()}; }
    StateAdjoint Solver::allocate_state_adjoint(const Domain& domain) const { return {.density = domain.allocate_scalar_adjoint_field(), .velocity = domain.allocate_staggered_vector_adjoint_field()}; }
    DenseControl Solver::allocate_control(const Domain& domain) const { return {.force = domain.allocate_centered_vector_field()}; }
    DenseControlTangent Solver::allocate_control_tangent(const Domain& domain) const { return {.force = domain.allocate_centered_vector_field()}; }
    DenseControlAdjoint Solver::allocate_control_adjoint(const Domain& domain) const { return {.force = domain.allocate_centered_vector_adjoint_field()}; }

    StepCache Solver::allocate_step_cache(const Domain& domain) const {
        return {
            .controlled_velocity = domain.allocate_staggered_vector_field(),
            .advected_velocity = domain.allocate_staggered_vector_field(),
            .advected_density = domain.allocate_scalar_field(),
            .transported_density = domain.allocate_scalar_field(),
            .diffused_velocity = domain.allocate_staggered_vector_field(),
            .heat_force = domain.allocate_centered_vector_field(),
            .heated_velocity = domain.allocate_staggered_vector_field(),
            .constrained_velocity = domain.allocate_staggered_vector_field(),
            .pressure_rhs = domain.allocate_scalar_field(),
            .pressure = domain.allocate_scalar_field(),
            .projected_velocity = domain.allocate_staggered_vector_field(),
            .input_mass = allocate_scalar_value(domain),
            .advected_mass = allocate_scalar_value(domain),
        };
    }

    void Solver::clear(const Domain& domain, State& state) const { domain.clear(state.density); domain.clear(state.velocity); }
    void Solver::clear(const Domain& domain, StateTangent& value) const { domain.clear(value.density); domain.clear(value.velocity); }
    void Solver::clear(const Domain& domain, StateAdjoint& value) const { domain.clear(value.density); domain.clear(value.velocity); }
    void Solver::clear(const Domain& domain, DenseControl& control) const { domain.clear(control.force); }
    void Solver::clear(const Domain& domain, DenseControlTangent& value) const { domain.clear(value.force); }
    void Solver::clear(const Domain& domain, DenseControlAdjoint& value) const { domain.clear(value.force); }

    void Solver::copy(const Domain& domain, const State& source, State& destination) const { domain.copy(source.density, destination.density); domain.copy(source.velocity, destination.velocity); }
    void Solver::copy(const Domain& domain, const StateTangent& source, StateTangent& destination) const { domain.copy(source.density, destination.density); domain.copy(source.velocity, destination.velocity); }
    void Solver::accumulate(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const { domain.accumulate(source.density, destination.density); domain.accumulate(source.velocity, destination.velocity); }

    void Solver::forward(const Domain& domain, const State& state, const DenseControl& control, State& output, StepCache& cache) {
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData velocity_boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        const float density_retention = std::exp(-configuration.density_dissipation * domain.configuration.time_step);
        cuda_detail::integrate_velocity_forward(domain.stream, grid, cuda_detail::staggered(state.velocity), cuda_detail::centered(control.force), cuda_detail::staggered(cache.controlled_velocity));
        cuda_detail::advect_velocity_forward(domain.stream, grid, cuda_detail::staggered(cache.controlled_velocity), velocity_boundary, cuda_detail::staggered(raw_advected_velocity));
        cuda_detail::constrain_velocity_forward(domain.stream, grid, cuda_detail::staggered(raw_advected_velocity), velocity_boundary, cuda_detail::staggered(cache.advected_velocity));
        cuda_detail::advect_scalar_forward(domain.stream, grid, cuda_detail::scalar(state.density), cuda_detail::staggered(cache.controlled_velocity), cuda_detail::scalar_boundary(domain.configuration.density_boundary), velocity_boundary, cuda_detail::scalar(cache.advected_density));
        cuda_detail::mass_forward(domain.stream, grid, density_retention, cuda_detail::scalar(state.density), cuda_detail::scalar(cache.advected_density), cache.input_mass.data(), cache.advected_mass.data(), cuda_detail::scalar(cache.transported_density));
        cuda_detail::diffusion_forward(domain.stream, grid, configuration.diffusion_iterations, configuration.viscosity, velocity_boundary, cuda_detail::staggered(cache.advected_velocity), cuda_detail::staggered(diffusion_first), cuda_detail::staggered(diffusion_second), cuda_detail::staggered(cache.diffused_velocity));
        cuda_detail::heat_forward(domain.stream, grid, configuration.density_buoyancy, cuda_detail::scalar(cache.transported_density), cuda_detail::centered(cache.heat_force));
        cuda_detail::integrate_velocity_forward(domain.stream, grid, cuda_detail::staggered(cache.diffused_velocity), cuda_detail::centered(cache.heat_force), cuda_detail::staggered(cache.heated_velocity));
        cuda_detail::constrain_velocity_forward(domain.stream, grid, cuda_detail::staggered(cache.heated_velocity), velocity_boundary, cuda_detail::staggered(cache.constrained_velocity));
        domain.clear(cache.pressure);
        cuda_detail::pressure_rhs_forward(domain.stream, grid, domain.pressure_anchor, cuda_detail::staggered(cache.constrained_velocity), cuda_detail::scalar(cache.pressure_rhs));
        cuda_detail::pressure_forward(domain.stream, grid, configuration.pressure_iterations, domain.pressure_anchor, cuda_detail::scalar_boundary(domain.configuration.pressure_boundary), cuda_detail::scalar(cache.pressure_rhs), cuda_detail::scalar(cache.pressure));
        cuda_detail::project_velocity_forward(domain.stream, grid, cuda_detail::staggered(cache.constrained_velocity), cuda_detail::scalar(cache.pressure), cuda_detail::staggered(cache.projected_velocity));
        domain.copy(cache.transported_density, output.density);
        domain.copy(cache.projected_velocity, output.velocity);
    }

    void Solver::jvp(const Domain& domain, const State& state, const DenseControl&, const StepCache& cache, const StateTangent& state_tangent, const DenseControlTangent& control_tangent, StateTangent& output_tangent) {
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData velocity_boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        cuda_detail::VelocityBoundaryData tangent_boundary = velocity_boundary;
        for (float& value : tangent_boundary.values) value = 0.0F;
        const float density_retention = std::exp(-configuration.density_dissipation * domain.configuration.time_step);
        cuda_detail::integrate_velocity_forward(domain.stream, grid, cuda_detail::staggered(state_tangent.velocity), cuda_detail::centered(control_tangent.force), cuda_detail::staggered(tangent.controlled_velocity));
        cuda_detail::advect_velocity_jvp(domain.stream, grid, cuda_detail::staggered(cache.controlled_velocity), cuda_detail::staggered(tangent.controlled_velocity), velocity_boundary, cuda_detail::staggered(tangent.raw_advected_velocity));
        cuda_detail::constrain_velocity_forward(domain.stream, grid, cuda_detail::staggered(tangent.raw_advected_velocity), tangent_boundary, cuda_detail::staggered(tangent.advected_velocity));
        cuda_detail::advect_scalar_jvp(domain.stream, grid, cuda_detail::scalar(state.density), cuda_detail::scalar(state_tangent.density), cuda_detail::staggered(cache.controlled_velocity), cuda_detail::staggered(tangent.controlled_velocity), cuda_detail::scalar_boundary(domain.configuration.density_boundary), velocity_boundary, cuda_detail::scalar(tangent.advected_density));
        cuda_detail::mass_jvp(domain.stream, grid, density_retention, cuda_detail::scalar(state.density), cuda_detail::scalar(cache.advected_density), cuda_detail::scalar(state_tangent.density), cuda_detail::scalar(tangent.advected_density), cache.input_mass.data(), cache.advected_mass.data(), tangent.input_mass.data(), tangent.advected_mass.data(), cuda_detail::scalar(tangent.transported_density));
        cuda_detail::diffusion_forward(domain.stream, grid, configuration.diffusion_iterations, configuration.viscosity, tangent_boundary, cuda_detail::staggered(tangent.advected_velocity), cuda_detail::staggered(tangent.diffusion_first), cuda_detail::staggered(tangent.diffusion_second), cuda_detail::staggered(tangent.diffused_velocity));
        cuda_detail::heat_jvp(domain.stream, grid, configuration.density_buoyancy, cuda_detail::scalar(tangent.transported_density), cuda_detail::centered(tangent.heat_force));
        cuda_detail::integrate_velocity_forward(domain.stream, grid, cuda_detail::staggered(tangent.diffused_velocity), cuda_detail::centered(tangent.heat_force), cuda_detail::staggered(tangent.heated_velocity));
        cuda_detail::constrain_velocity_forward(domain.stream, grid, cuda_detail::staggered(tangent.heated_velocity), tangent_boundary, cuda_detail::staggered(tangent.constrained_velocity));
        domain.clear(tangent.pressure);
        cuda_detail::pressure_rhs_forward(domain.stream, grid, domain.pressure_anchor, cuda_detail::staggered(tangent.constrained_velocity), cuda_detail::scalar(tangent.pressure_rhs));
        ScalarBoundary pressure_boundary = domain.configuration.pressure_boundary;
        pressure_boundary.x_min.value = 0.0F; pressure_boundary.x_max.value = 0.0F;
        pressure_boundary.y_min.value = 0.0F; pressure_boundary.y_max.value = 0.0F;
        pressure_boundary.z_min.value = 0.0F; pressure_boundary.z_max.value = 0.0F;
        cuda_detail::pressure_forward(domain.stream, grid, configuration.pressure_iterations, domain.pressure_anchor, cuda_detail::scalar_boundary(pressure_boundary), cuda_detail::scalar(tangent.pressure_rhs), cuda_detail::scalar(tangent.pressure));
        cuda_detail::project_velocity_forward(domain.stream, grid, cuda_detail::staggered(tangent.constrained_velocity), cuda_detail::scalar(tangent.pressure), cuda_detail::staggered(tangent.projected_velocity));
        domain.copy(tangent.transported_density, output_tangent.density);
        domain.copy(tangent.projected_velocity, output_tangent.velocity);
    }

    void Solver::vjp(const Domain& domain, const State& state, const DenseControl&, const StepCache& cache, const StateAdjoint& output_adjoint, StateAdjoint& state_adjoint, DenseControlAdjoint& control_adjoint) {
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData velocity_boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        const float density_retention = std::exp(-configuration.density_dissipation * domain.configuration.time_step);
        clear(domain, state_adjoint); clear(domain, control_adjoint);
        domain.clear(adjoint.transported_density); domain.clear(adjoint.advected_density);
        domain.clear(adjoint.projected_velocity); domain.clear(adjoint.pressure); domain.clear(adjoint.pressure_rhs); domain.clear(adjoint.constrained_velocity);
        domain.clear(adjoint.heated_velocity); domain.clear(adjoint.heat_force); domain.clear(adjoint.diffused_velocity);
        domain.clear(adjoint.advected_velocity); domain.clear(adjoint.raw_advected_velocity); domain.clear(adjoint.controlled_velocity);
        domain.copy(output_adjoint.density, adjoint.transported_density);
        domain.copy(output_adjoint.velocity, adjoint.projected_velocity);
        cuda_detail::project_velocity_vjp(domain.stream, grid, cuda_detail::staggered_adjoint(adjoint.projected_velocity), cuda_detail::staggered_adjoint(adjoint.constrained_velocity), cuda_detail::scalar_adjoint(adjoint.pressure));
        cuda_detail::pressure_vjp(domain.stream, grid, configuration.pressure_iterations, domain.pressure_anchor, cuda_detail::scalar_boundary(domain.configuration.pressure_boundary), cuda_detail::scalar_adjoint(adjoint.pressure), cuda_detail::scalar_adjoint(adjoint.pressure_rhs));
        cuda_detail::pressure_rhs_vjp(domain.stream, grid, domain.pressure_anchor, cuda_detail::scalar_adjoint(adjoint.pressure_rhs), cuda_detail::staggered_adjoint(adjoint.constrained_velocity));
        cuda_detail::constrain_velocity_vjp(domain.stream, grid, cuda_detail::staggered_adjoint(adjoint.constrained_velocity), velocity_boundary, cuda_detail::staggered_adjoint(adjoint.heated_velocity));
        cuda_detail::integrate_velocity_vjp(domain.stream, grid, cuda_detail::staggered_adjoint(adjoint.heated_velocity), cuda_detail::staggered_adjoint(adjoint.diffused_velocity), cuda_detail::centered_adjoint(adjoint.heat_force));
        cuda_detail::heat_vjp(domain.stream, grid, configuration.density_buoyancy, cuda_detail::centered_adjoint(adjoint.heat_force), cuda_detail::scalar_adjoint(adjoint.transported_density));
        cuda_detail::diffusion_vjp(domain.stream, grid, configuration.diffusion_iterations, configuration.viscosity, velocity_boundary, cuda_detail::staggered_adjoint(adjoint.diffused_velocity), cuda_detail::staggered_adjoint(adjoint.projected_velocity), cuda_detail::staggered_adjoint(adjoint.raw_advected_velocity), cuda_detail::staggered_adjoint(adjoint.advected_velocity));
        domain.clear(adjoint.raw_advected_velocity);
        cuda_detail::constrain_velocity_vjp(domain.stream, grid, cuda_detail::staggered_adjoint(adjoint.advected_velocity), velocity_boundary, cuda_detail::staggered_adjoint(adjoint.raw_advected_velocity));
        cuda_detail::advect_velocity_vjp(domain.stream, grid, cuda_detail::staggered(cache.controlled_velocity), velocity_boundary, cuda_detail::staggered_adjoint(adjoint.raw_advected_velocity), cuda_detail::staggered_adjoint(adjoint.controlled_velocity));
        cuda_detail::mass_vjp(domain.stream, grid, density_retention, cuda_detail::scalar(cache.advected_density), cache.input_mass.data(), cache.advected_mass.data(), cuda_detail::scalar_adjoint(adjoint.transported_density), adjoint.density_dot.data(), cuda_detail::scalar_adjoint(state_adjoint.density), cuda_detail::scalar_adjoint(adjoint.advected_density));
        cuda_detail::advect_scalar_vjp(domain.stream, grid, cuda_detail::scalar(state.density), cuda_detail::staggered(cache.controlled_velocity), cuda_detail::scalar_boundary(domain.configuration.density_boundary), velocity_boundary, cuda_detail::scalar_adjoint(adjoint.advected_density), cuda_detail::scalar_adjoint(state_adjoint.density), cuda_detail::staggered_adjoint(adjoint.controlled_velocity));
        cuda_detail::integrate_velocity_vjp(domain.stream, grid, cuda_detail::staggered_adjoint(adjoint.controlled_velocity), cuda_detail::staggered_adjoint(state_adjoint.velocity), cuda_detail::centered_adjoint(control_adjoint.force));
    }
} // namespace physica::fluids::gas::adjoint_control
