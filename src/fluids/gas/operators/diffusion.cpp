module;

#include "../detail/cuda/interop.h"
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

    void IdentityVelocityDiffusion::forward(const Domain& domain, const StaggeredVectorField<float>& source, StaggeredVectorField<float>& output, Workspace&) const {
        domain.copy(source, output);
    }
    void IdentityVelocityDiffusion::jvp(const Domain& domain, const StaggeredVectorField<float>& source_tangent, StaggeredVectorField<float>& output_tangent, TangentWorkspace&) const {
        domain.copy(source_tangent, output_tangent);
    }
    void IdentityVelocityDiffusion::vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& source_adjoint, AdjointWorkspace&) const {
        cuda_backend::identity_velocity_vjp(domain.stream, detail::cuda::grid(domain.configuration), detail::cuda::staggered_adjoint(output_adjoint), detail::cuda::staggered_adjoint(source_adjoint));
    }

    ImplicitVelocityDiffusion::ImplicitVelocityDiffusion(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ImplicitVelocityDiffusion::Workspace ImplicitVelocityDiffusion::allocate_workspace(const Domain& domain) const {
        return {.first = domain.allocate_staggered_vector_field<float>(), .second = domain.allocate_staggered_vector_field<float>()};
    }

    ImplicitVelocityDiffusion::TangentWorkspace ImplicitVelocityDiffusion::allocate_tangent_workspace(const Domain& domain) const {
        return {.first = domain.allocate_staggered_vector_field<float>(), .second = domain.allocate_staggered_vector_field<float>()};
    }

    ImplicitVelocityDiffusion::AdjointWorkspace ImplicitVelocityDiffusion::allocate_adjoint_workspace(const Domain& domain) const {
        return {.first = domain.allocate_staggered_vector_field<double>(), .second = domain.allocate_staggered_vector_field<double>()};
    }

    void ImplicitVelocityDiffusion::forward(const Domain& domain, const StaggeredVectorField<float>& source, StaggeredVectorField<float>& output, Workspace& workspace) const {
        cuda_backend::diffusion_forward(domain.stream, detail::cuda::grid(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::staggered(source), detail::cuda::staggered(workspace.first), detail::cuda::staggered(workspace.second), detail::cuda::staggered(output));
    }

    void ImplicitVelocityDiffusion::jvp(const Domain& domain, const StaggeredVectorField<float>& source_tangent, StaggeredVectorField<float>& output_tangent, TangentWorkspace& workspace) const {
        cuda_backend::diffusion_forward(domain.stream, detail::cuda::grid(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), detail::cuda::velocity_boundary(homogeneous(domain.configuration.velocity_boundary)), detail::cuda::staggered(source_tangent), detail::cuda::staggered(workspace.first), detail::cuda::staggered(workspace.second), detail::cuda::staggered(output_tangent));
    }

    void ImplicitVelocityDiffusion::vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& source_adjoint, AdjointWorkspace& workspace) const {
        cuda_backend::diffusion_vjp(domain.stream, detail::cuda::grid(domain.configuration), configuration.iterations, configuration.viscosity, domain.collider_ids.values.data(), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::staggered_adjoint(output_adjoint), detail::cuda::staggered_adjoint(workspace.first), detail::cuda::staggered_adjoint(workspace.second), detail::cuda::staggered_adjoint(source_adjoint));
    }
} // namespace physica::fluids::gas::operators
