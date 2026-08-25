module;

#include "../detail/cuda/interop.h"
#include "advection-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.advection;

import physica.fluids.gas.operators.pointwise;

namespace physica::fluids::gas::operators {
    SemiLagrangianRK2::SemiLagrangianRK2(Configuration) {}

    SemiLagrangianRK2::Workspace SemiLagrangianRK2::allocate_workspace(const Domain& domain) const {
        return {.raw_velocity = domain.allocate_staggered_vector_field<float>()};
    }

    SemiLagrangianRK2::TangentWorkspace SemiLagrangianRK2::allocate_tangent_workspace(const Domain& domain) const {
        return {.raw_velocity = domain.allocate_staggered_vector_field<float>()};
    }

    SemiLagrangianRK2::AdjointWorkspace SemiLagrangianRK2::allocate_adjoint_workspace(const Domain& domain) const {
        return {.raw_velocity = domain.allocate_staggered_vector_field<double>()};
    }

    void SemiLagrangianRK2::velocity_forward(const Domain& domain, const StaggeredVectorField<float>& velocity, StaggeredVectorField<float>& output, Workspace& workspace) const {
        cuda_backend::advect_velocity_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::staggered(workspace.raw_velocity));
        constrain_velocity_forward(domain, workspace.raw_velocity, output);
    }

    void SemiLagrangianRK2::velocity_jvp(const Domain& domain, const StaggeredVectorField<float>& velocity, const StaggeredVectorField<float>& velocity_tangent, StaggeredVectorField<float>& output_tangent, TangentWorkspace& workspace) const {
        cuda_backend::advect_velocity_jvp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity), detail::cuda::staggered(velocity_tangent), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::staggered(workspace.raw_velocity));
        constrain_velocity_jvp(domain, workspace.raw_velocity, output_tangent);
    }

    void SemiLagrangianRK2::velocity_vjp(const Domain& domain, const StaggeredVectorField<float>& velocity, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const {
        domain.clear(workspace.raw_velocity);
        constrain_velocity_vjp(domain, output_adjoint, workspace.raw_velocity);
        cuda_backend::advect_velocity_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::staggered_adjoint(workspace.raw_velocity), detail::cuda::staggered_adjoint(velocity_adjoint));
    }

    void SemiLagrangianRK2::scalar_forward(const Domain& domain, const CellField<float>& source, const StaggeredVectorField<float>& velocity, const ScalarBoundary& boundary, const CellField<float>& collider_value, CellField<float>& output) const {
        cuda_backend::advect_scalar_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar(collider_value), detail::cuda::scalar(source), detail::cuda::staggered(velocity), detail::cuda::scalar_boundary(boundary), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::scalar(output));
    }

    void SemiLagrangianRK2::scalar_jvp(const Domain& domain, const CellField<float>& source, const CellField<float>& source_tangent, const StaggeredVectorField<float>& velocity, const StaggeredVectorField<float>& velocity_tangent, const ScalarBoundary& boundary, CellField<float>& output_tangent) const {
        cuda_backend::advect_scalar_jvp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar(source), detail::cuda::scalar(source_tangent), detail::cuda::staggered(velocity), detail::cuda::staggered(velocity_tangent), detail::cuda::scalar_boundary(boundary), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::scalar(output_tangent));
    }

    void SemiLagrangianRK2::scalar_vjp(const Domain& domain, const CellField<float>& source, const StaggeredVectorField<float>& velocity, const ScalarBoundary& boundary, const CellField<double>& output_adjoint, CellField<double>& source_adjoint, StaggeredVectorField<double>& velocity_adjoint) const {
        cuda_backend::advect_scalar_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar(source), detail::cuda::staggered(velocity), detail::cuda::scalar_boundary(boundary), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::scalar_adjoint(output_adjoint), detail::cuda::scalar_adjoint(source_adjoint), detail::cuda::staggered_adjoint(velocity_adjoint));
    }
} // namespace physica::fluids::gas::operators
