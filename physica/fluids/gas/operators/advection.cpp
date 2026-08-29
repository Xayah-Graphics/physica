module;

#include <physica/fluids/gas/interop.h>
#include "advection-kernels.h"
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

    void SemiLagrangianRK2::velocity_forward(const Domain& domain, const VectorField<float>& velocity, VectorField<float>& output, Workspace& workspace) const {
        kernels::advect_velocity_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity), device::velocity_boundary(domain.configuration.velocity_boundary), field::view(workspace.raw_velocity));
        constrain_velocity_forward(domain, workspace.raw_velocity, output);
    }

    void SemiLagrangianRK2::velocity_jvp(const Domain& domain, const VectorField<float>& velocity, const VectorField<float>& velocity_tangent, VectorField<float>& output_tangent, TangentWorkspace& workspace) const {
        kernels::advect_velocity_jvp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity), field::view(velocity_tangent), device::velocity_boundary(domain.configuration.velocity_boundary), field::view(workspace.raw_velocity));
        constrain_velocity_jvp(domain, workspace.raw_velocity, output_tangent);
    }

    void SemiLagrangianRK2::velocity_vjp(const Domain& domain, const VectorField<float>& velocity, const VectorField<double>& output_adjoint, VectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const {
        domain.grid.clear(workspace.raw_velocity);
        constrain_velocity_vjp(domain, output_adjoint, workspace.raw_velocity);
        kernels::advect_velocity_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity), device::velocity_boundary(domain.configuration.velocity_boundary), field::view(workspace.raw_velocity), field::view(velocity_adjoint));
    }

    void SemiLagrangianRK2::scalar_forward(const Domain& domain, const ScalarField<float>& source, const VectorField<float>& velocity, const ScalarBoundary& boundary, const ScalarField<float>& collider_value, ScalarField<float>& output) const {
        kernels::advect_scalar_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(collider_value), field::scalar_view(source), field::view(velocity), device::scalar_boundary(boundary), device::velocity_boundary(domain.configuration.velocity_boundary), field::scalar_view(output));
    }

    void SemiLagrangianRK2::scalar_jvp(const Domain& domain, const ScalarField<float>& source, const ScalarField<float>& source_tangent, const VectorField<float>& velocity, const VectorField<float>& velocity_tangent, const ScalarBoundary& boundary, ScalarField<float>& output_tangent) const {
        kernels::advect_scalar_jvp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(source), field::scalar_view(source_tangent), field::view(velocity), field::view(velocity_tangent), device::scalar_boundary(boundary), device::velocity_boundary(domain.configuration.velocity_boundary), field::scalar_view(output_tangent));
    }

    void SemiLagrangianRK2::scalar_vjp(const Domain& domain, const ScalarField<float>& source, const VectorField<float>& velocity, const ScalarBoundary& boundary, const ScalarField<double>& output_adjoint, ScalarField<double>& source_adjoint, VectorField<double>& velocity_adjoint) const {
        kernels::advect_scalar_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(source), field::view(velocity), device::scalar_boundary(boundary), device::velocity_boundary(domain.configuration.velocity_boundary), field::scalar_view(output_adjoint), field::scalar_view(source_adjoint), field::view(velocity_adjoint));
    }
} // namespace physica::fluids::gas::operators
