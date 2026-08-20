module;

#include "../domain/interop.h"
#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

module physica.fluids.gas.smoke.transport;

import std;

namespace physica::fluids::gas::smoke {
    SemiLagrangianRK2::SemiLagrangianRK2(const Domain& domain, Configuration, const ExecutionMode mode)
        : raw_advected_velocity(domain.allocate_staggered_vector_field()), differentiation{} {
        if (mode == ExecutionMode::differentiable) {
            differentiation.emplace(Differentiation{
                .sourced_density_tangent       = domain.allocate_scalar_field(),
                .sourced_temperature_tangent   = domain.allocate_scalar_field(),
                .forced_velocity_tangent       = domain.allocate_staggered_vector_field(),
                .raw_advected_velocity_tangent = domain.allocate_staggered_vector_field(),
                .advected_velocity_tangent     = domain.allocate_staggered_vector_field(),
                .sourced_density_adjoint       = domain.allocate_scalar_adjoint_field(),
                .sourced_temperature_adjoint   = domain.allocate_scalar_adjoint_field(),
                .projected_velocity_adjoint    = domain.allocate_staggered_vector_adjoint_field(),
                .advected_velocity_adjoint     = domain.allocate_staggered_vector_adjoint_field(),
                .raw_advected_velocity_adjoint = domain.allocate_staggered_vector_adjoint_field(),
                .forced_velocity_adjoint       = domain.allocate_staggered_vector_adjoint_field(),
            });
        }
    }

    SemiLagrangianRK2::Cache SemiLagrangianRK2::allocate_cache(const Domain& domain) const {
        return {
            .sourced_density     = domain.allocate_scalar_field(),
            .sourced_temperature = domain.allocate_scalar_field(),
            .forced_velocity     = domain.allocate_staggered_vector_field(),
            .advected_velocity   = domain.allocate_staggered_vector_field(),
        };
    }

    void SemiLagrangianRK2::source_forward(const Domain& domain, const ScalarField& density, const ScalarField& temperature, const ScalarField& density_source, const ScalarField& temperature_source, Cache& cache) const {
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        cuda_detail::source_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(density), cuda_detail::scalar(density_source), cuda_detail::scalar(cache.sourced_density));
        cuda_detail::source_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(temperature), cuda_detail::scalar(temperature_source), cuda_detail::scalar(cache.sourced_temperature));
    }

    void SemiLagrangianRK2::velocity_forward(const Domain& domain, const StaggeredVectorField& velocity, const CenteredVectorField& force, Cache& cache) {
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        cuda_detail::integrate_velocity_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(velocity), cuda_detail::centered(force), cuda_detail::staggered(cache.forced_velocity));
        cuda_detail::advect_velocity_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(cache.forced_velocity), boundary, cuda_detail::staggered(raw_advected_velocity));
        cuda_detail::constrain_velocity_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(domain.collider_velocity), cuda_detail::staggered(raw_advected_velocity), boundary, cuda_detail::staggered(cache.advected_velocity));
    }

    void SemiLagrangianRK2::scalar_forward(const Domain& domain, const Cache& cache, const StaggeredVectorField& projected_velocity, ScalarField& density, ScalarField& temperature) const {
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData velocity_boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        cuda_detail::advect_scalar_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(domain.collider_density), cuda_detail::scalar(cache.sourced_density), cuda_detail::staggered(projected_velocity), cuda_detail::scalar_boundary(domain.configuration.density_boundary), velocity_boundary, cuda_detail::scalar(density));
        cuda_detail::advect_scalar_forward(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(domain.collider_temperature), cuda_detail::scalar(cache.sourced_temperature), cuda_detail::staggered(projected_velocity), cuda_detail::scalar_boundary(domain.configuration.temperature_boundary), velocity_boundary, cuda_detail::scalar(temperature));
    }

    void SemiLagrangianRK2::source_jvp(const Domain& domain, const ScalarField& density_tangent, const ScalarField& temperature_tangent, const ScalarField& density_source_tangent, const ScalarField& temperature_source_tangent) {
        Differentiation& workspace = *differentiation;
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        cuda_detail::source_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(density_tangent), cuda_detail::scalar(density_source_tangent), cuda_detail::scalar(workspace.sourced_density_tangent));
        cuda_detail::source_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(temperature_tangent), cuda_detail::scalar(temperature_source_tangent), cuda_detail::scalar(workspace.sourced_temperature_tangent));
    }

    void SemiLagrangianRK2::velocity_jvp(const Domain& domain, const Cache& cache, const StaggeredVectorField& velocity_tangent, const CenteredVectorField& force_tangent) {
        Differentiation& workspace = *differentiation;
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        cuda_detail::integrate_velocity_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(velocity_tangent), cuda_detail::centered(force_tangent), cuda_detail::staggered(workspace.forced_velocity_tangent));
        cuda_detail::advect_velocity_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(cache.forced_velocity), cuda_detail::staggered(workspace.forced_velocity_tangent), boundary, cuda_detail::staggered(workspace.raw_advected_velocity_tangent));
        cuda_detail::constrain_velocity_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(workspace.raw_advected_velocity_tangent), boundary, cuda_detail::staggered(workspace.advected_velocity_tangent));
    }

    void SemiLagrangianRK2::scalar_jvp(const Domain& domain, const Cache& cache, const StaggeredVectorField& projected_velocity, const StaggeredVectorField& projected_velocity_tangent, ScalarField& density_tangent, ScalarField& temperature_tangent) const {
        const Differentiation& workspace = *differentiation;
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData velocity_boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        cuda_detail::advect_scalar_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(cache.sourced_density), cuda_detail::scalar(workspace.sourced_density_tangent), cuda_detail::staggered(projected_velocity), cuda_detail::staggered(projected_velocity_tangent), cuda_detail::scalar_boundary(domain.configuration.density_boundary), velocity_boundary, cuda_detail::scalar(density_tangent));
        cuda_detail::advect_scalar_jvp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(cache.sourced_temperature), cuda_detail::scalar(workspace.sourced_temperature_tangent), cuda_detail::staggered(projected_velocity), cuda_detail::staggered(projected_velocity_tangent), cuda_detail::scalar_boundary(domain.configuration.temperature_boundary), velocity_boundary, cuda_detail::scalar(temperature_tangent));
    }

    void SemiLagrangianRK2::scalar_vjp(const Domain& domain, const Cache& cache, const StaggeredVectorField& projected_velocity, const ScalarAdjointField& density_adjoint, const ScalarAdjointField& temperature_adjoint, const StaggeredVectorAdjointField& velocity_adjoint) {
        Differentiation& workspace = *differentiation;
        domain.clear(workspace.sourced_density_adjoint);
        domain.clear(workspace.sourced_temperature_adjoint);
        domain.copy(velocity_adjoint, workspace.projected_velocity_adjoint);
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData velocity_boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        cuda_detail::advect_scalar_vjp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(cache.sourced_density), cuda_detail::staggered(projected_velocity), cuda_detail::scalar_boundary(domain.configuration.density_boundary), velocity_boundary, cuda_detail::scalar_adjoint(density_adjoint), cuda_detail::scalar_adjoint(workspace.sourced_density_adjoint), cuda_detail::staggered_adjoint(workspace.projected_velocity_adjoint));
        cuda_detail::advect_scalar_vjp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar(cache.sourced_temperature), cuda_detail::staggered(projected_velocity), cuda_detail::scalar_boundary(domain.configuration.temperature_boundary), velocity_boundary, cuda_detail::scalar_adjoint(temperature_adjoint), cuda_detail::scalar_adjoint(workspace.sourced_temperature_adjoint), cuda_detail::staggered_adjoint(workspace.projected_velocity_adjoint));
    }

    void SemiLagrangianRK2::velocity_vjp(const Domain& domain, const Cache& cache, StaggeredVectorAdjointField& velocity_adjoint, CenteredVectorAdjointField& force_adjoint) {
        Differentiation& workspace = *differentiation;
        domain.clear(workspace.raw_advected_velocity_adjoint);
        domain.clear(workspace.forced_velocity_adjoint);
        domain.clear(force_adjoint);
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        const cuda_detail::VelocityBoundaryData boundary = cuda_detail::velocity_boundary(domain.configuration.velocity_boundary);
        cuda_detail::constrain_velocity_vjp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered_adjoint(workspace.advected_velocity_adjoint), boundary, cuda_detail::staggered_adjoint(workspace.raw_advected_velocity_adjoint));
        cuda_detail::advect_velocity_vjp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered(cache.forced_velocity), boundary, cuda_detail::staggered_adjoint(workspace.raw_advected_velocity_adjoint), cuda_detail::staggered_adjoint(workspace.forced_velocity_adjoint));
        cuda_detail::integrate_velocity_vjp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::staggered_adjoint(workspace.forced_velocity_adjoint), cuda_detail::staggered_adjoint(velocity_adjoint), cuda_detail::centered_adjoint(force_adjoint));
    }

    void SemiLagrangianRK2::source_vjp(const Domain& domain, ScalarAdjointField& density_adjoint, ScalarAdjointField& temperature_adjoint, ScalarAdjointField& density_source_adjoint, ScalarAdjointField& temperature_source_adjoint) const {
        const Differentiation& workspace = *differentiation;
        const cuda_detail::Grid grid = cuda_detail::grid(domain.configuration);
        cuda_detail::source_vjp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar_adjoint(workspace.sourced_density_adjoint), cuda_detail::scalar_adjoint(density_adjoint), cuda_detail::scalar_adjoint(density_source_adjoint));
        cuda_detail::source_vjp(domain.stream, grid, domain.cell_mask.data(), cuda_detail::scalar_adjoint(workspace.sourced_temperature_adjoint), cuda_detail::scalar_adjoint(temperature_adjoint), cuda_detail::scalar_adjoint(temperature_source_adjoint));
    }
} // namespace physica::fluids::gas::smoke
