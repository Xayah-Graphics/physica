module;

#include "diffusion-kernels.h"
#include <fluids/gas/interop.h>
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

    void IdentityVelocityDiffusion::forward(const Domain& domain, const simulation::VectorField<float>& source, simulation::VectorField<float>& output, Workspace&) const {
        domain.grid.copy(source, output);
    }
    void IdentityVelocityDiffusion::jvp(const Domain& domain, const simulation::VectorField<float>& source_tangent, simulation::VectorField<float>& output_tangent, TangentWorkspace&) const {
        domain.grid.copy(source_tangent, output_tangent);
    }
    void IdentityVelocityDiffusion::vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& source_adjoint, AdjointWorkspace&) const {
        kernels::identity_velocity_vjp(domain.grid.stream, device::discretization(domain.configuration), simulation::view(output_adjoint), simulation::view(source_adjoint));
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

    void ImplicitVelocityDiffusion::forward(const Domain& domain, const simulation::VectorField<float>& source, simulation::VectorField<float>& output, Workspace& workspace) const {
        kernels::diffusion_forward(domain.grid.stream, device::discretization(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::view(source), simulation::view(workspace.first), simulation::view(workspace.second), simulation::view(output));
    }

    void ImplicitVelocityDiffusion::jvp(const Domain& domain, const simulation::VectorField<float>& source_tangent, simulation::VectorField<float>& output_tangent, TangentWorkspace& workspace) const {
        kernels::diffusion_forward(domain.grid.stream, device::discretization(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), device::velocity_boundary(homogeneous(domain.configuration.velocity_boundary)), simulation::view(source_tangent), simulation::view(workspace.first), simulation::view(workspace.second), simulation::view(output_tangent));
    }

    void ImplicitVelocityDiffusion::vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& source_adjoint, AdjointWorkspace& workspace) const {
        kernels::diffusion_vjp(domain.grid.stream, device::discretization(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::view(output_adjoint), simulation::view(workspace.first), simulation::view(workspace.second), simulation::view(source_adjoint));
    }
} // namespace physica::fluids::gas::operators
