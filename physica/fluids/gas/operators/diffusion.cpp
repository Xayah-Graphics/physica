module;

#include <physica/fluids/gas/interop.h>
#include "diffusion-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.diffusion;

import std;

namespace physica::fluids::gas::operators {
    IdentityVelocityDiffusion::IdentityVelocityDiffusion(Configuration) {}

    IdentityVelocityDiffusion::Workspace IdentityVelocityDiffusion::allocate_workspace(const Domain&) const {
        return {};
    }
    IdentityVelocityDiffusion::TangentWorkspace IdentityVelocityDiffusion::allocate_tangent_workspace(const Domain&) const {
        return {};
    }
    IdentityVelocityDiffusion::AdjointWorkspace IdentityVelocityDiffusion::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void IdentityVelocityDiffusion::forward(const Domain& domain, const VectorField<float>& source, VectorField<float>& output, Workspace&) const {
        domain.grid.copy(source, output);
    }
    void IdentityVelocityDiffusion::jvp(const Domain& domain, const VectorField<float>& source_tangent, VectorField<float>& output_tangent, TangentWorkspace&) const {
        domain.grid.copy(source_tangent, output_tangent);
    }
    void IdentityVelocityDiffusion::vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& source_adjoint, AdjointWorkspace&) const {
        kernels::identity_velocity_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), field::view(output_adjoint), field::view(source_adjoint));
    }

    ImplicitVelocityDiffusion::ImplicitVelocityDiffusion(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ImplicitVelocityDiffusion::Workspace ImplicitVelocityDiffusion::allocate_workspace(const Domain& domain) const {
        return {.first = domain.grid.allocate_mac_field<float>(), .second = domain.grid.allocate_mac_field<float>()};
    }

    ImplicitVelocityDiffusion::TangentWorkspace ImplicitVelocityDiffusion::allocate_tangent_workspace(const Domain& domain) const {
        return {.first = domain.grid.allocate_mac_field<float>(), .second = domain.grid.allocate_mac_field<float>()};
    }

    ImplicitVelocityDiffusion::AdjointWorkspace ImplicitVelocityDiffusion::allocate_adjoint_workspace(const Domain& domain) const {
        return {.first = domain.grid.allocate_mac_field<double>(), .second = domain.grid.allocate_mac_field<double>()};
    }

    void ImplicitVelocityDiffusion::forward(const Domain& domain, const VectorField<float>& source, VectorField<float>& output, Workspace& workspace) const {
        kernels::diffusion_forward(domain.grid.fields.stream, device::discretization(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), device::velocity_boundary(domain.configuration.velocity_boundary), field::view(source), field::view(workspace.first), field::view(workspace.second), field::view(output));
    }

    void ImplicitVelocityDiffusion::jvp(const Domain& domain, const VectorField<float>& source_tangent, VectorField<float>& output_tangent, TangentWorkspace& workspace) const {
        kernels::diffusion_forward(domain.grid.fields.stream, device::discretization(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), device::velocity_boundary(homogeneous(domain.configuration.velocity_boundary)), field::view(source_tangent), field::view(workspace.first), field::view(workspace.second), field::view(output_tangent));
    }

    void ImplicitVelocityDiffusion::vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& source_adjoint, AdjointWorkspace& workspace) const {
        kernels::diffusion_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), device::velocity_boundary(domain.configuration.velocity_boundary), field::view(output_adjoint), field::view(workspace.first), field::view(workspace.second), field::view(source_adjoint));
    }
} // namespace physica::fluids::gas::operators
