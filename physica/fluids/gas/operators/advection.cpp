module;

#include "advection-kernels.h"
#include <fluids/gas/interop.h>
#include <physica/cuda.h>

module physica.fluids.gas.operators.advection;

import physica.fluids.gas.operators.pointwise;

namespace physica::fluids::gas::operators {
    SemiLagrangianRK2::SemiLagrangianRK2(Configuration) {}

    SemiLagrangianRK2::Workspace SemiLagrangianRK2::allocate_workspace(const Domain& domain) const {
        return {.raw_velocity = domain.grid.allocate_mac_field<float>()};
    }

    SemiLagrangianRK2::TangentWorkspace SemiLagrangianRK2::allocate_tangent_workspace(const Domain& domain) const {
        return {.raw_velocity = domain.grid.allocate_mac_field<float>()};
    }

    SemiLagrangianRK2::AdjointWorkspace SemiLagrangianRK2::allocate_adjoint_workspace(const Domain& domain) const {
        return {.raw_velocity = domain.grid.allocate_mac_field<double>()};
    }

    void SemiLagrangianRK2::velocity_forward(const Domain& domain, const simulation::VectorField<float>& velocity, simulation::VectorField<float>& output, Workspace& workspace) const {
        kernels::advect_velocity_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::view(workspace.raw_velocity));
        constrain_velocity_forward(domain, workspace.raw_velocity, output);
    }

    void SemiLagrangianRK2::velocity_jvp(const Domain& domain, const simulation::VectorField<float>& velocity, const simulation::VectorField<float>& velocity_tangent, simulation::VectorField<float>& output_tangent, TangentWorkspace& workspace) const {
        kernels::advect_velocity_jvp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity), simulation::view(velocity_tangent), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::view(workspace.raw_velocity));
        constrain_velocity_jvp(domain, workspace.raw_velocity, output_tangent);
    }

    void SemiLagrangianRK2::velocity_vjp(const Domain& domain, const simulation::VectorField<float>& velocity, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const {
        domain.grid.clear(workspace.raw_velocity);
        constrain_velocity_vjp(domain, output_adjoint, workspace.raw_velocity);
        kernels::advect_velocity_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::view(workspace.raw_velocity), simulation::view(velocity_adjoint));
    }

    void SemiLagrangianRK2::scalar_forward(const Domain& domain, const simulation::ScalarField<float>& source, const simulation::VectorField<float>& velocity, const ScalarBoundary& boundary, const simulation::ScalarField<float>& collider_value, simulation::ScalarField<float>& output) const {
        kernels::advect_scalar_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(collider_value), simulation::scalar_view(source), simulation::view(velocity), device::scalar_boundary(boundary), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::scalar_view(output));
    }

    void SemiLagrangianRK2::scalar_jvp(const Domain& domain, const simulation::ScalarField<float>& source, const simulation::ScalarField<float>& source_tangent, const simulation::VectorField<float>& velocity, const simulation::VectorField<float>& velocity_tangent, const ScalarBoundary& boundary, simulation::ScalarField<float>& output_tangent) const {
        kernels::advect_scalar_jvp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(source), simulation::scalar_view(source_tangent), simulation::view(velocity), simulation::view(velocity_tangent), device::scalar_boundary(boundary), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::scalar_view(output_tangent));
    }

    void SemiLagrangianRK2::scalar_vjp(const Domain& domain, const simulation::ScalarField<float>& source, const simulation::VectorField<float>& velocity, const ScalarBoundary& boundary, const simulation::ScalarField<double>& output_adjoint, simulation::ScalarField<double>& source_adjoint, simulation::VectorField<double>& velocity_adjoint) const {
        kernels::advect_scalar_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(source), simulation::view(velocity), device::scalar_boundary(boundary), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::scalar_view(output_adjoint), simulation::scalar_view(source_adjoint), simulation::view(velocity_adjoint));
    }
} // namespace physica::fluids::gas::operators
