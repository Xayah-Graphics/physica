module;

#include "../detail/cuda/interop.h"
#include "pointwise-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.pointwise;

namespace physica::fluids::gas::operators {
    void add_source_forward(const Domain& domain, const CellField<float>& state, const CellField<float>& source, CellField<float>& output) {
        cuda_backend::source_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar(state), detail::cuda::scalar(source), detail::cuda::scalar(output));
    }

    void add_source_jvp(const Domain& domain, const CellField<float>& state_tangent, const CellField<float>& source_tangent, CellField<float>& output_tangent) {
        cuda_backend::source_jvp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar(state_tangent), detail::cuda::scalar(source_tangent), detail::cuda::scalar(output_tangent));
    }

    void add_source_vjp(const Domain& domain, const CellField<double>& output_adjoint, CellField<double>& state_adjoint, CellField<double>& source_adjoint) {
        cuda_backend::source_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::scalar_adjoint(output_adjoint), detail::cuda::scalar_adjoint(state_adjoint), detail::cuda::scalar_adjoint(source_adjoint));
    }

    void integrate_velocity_forward(const Domain& domain, const StaggeredVectorField<float>& velocity, const CenteredVectorField<float>& acceleration, StaggeredVectorField<float>& output) {
        cuda_backend::integrate_velocity_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity), detail::cuda::centered(acceleration), detail::cuda::staggered(output));
    }

    void integrate_velocity_jvp(const Domain& domain, const StaggeredVectorField<float>& velocity_tangent, const CenteredVectorField<float>& acceleration_tangent, StaggeredVectorField<float>& output_tangent) {
        cuda_backend::integrate_velocity_jvp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity_tangent), detail::cuda::centered(acceleration_tangent), detail::cuda::staggered(output_tangent));
    }

    void integrate_velocity_vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& velocity_adjoint, CenteredVectorField<double>& acceleration_adjoint) {
        cuda_backend::integrate_velocity_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered_adjoint(output_adjoint), detail::cuda::staggered_adjoint(velocity_adjoint), detail::cuda::centered_adjoint(acceleration_adjoint));
    }

    void constrain_velocity_forward(const Domain& domain, const StaggeredVectorField<float>& velocity, StaggeredVectorField<float>& output) {
        cuda_backend::constrain_velocity_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(domain.collider_velocity), detail::cuda::staggered(velocity), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::staggered(output));
    }

    void constrain_velocity_jvp(const Domain& domain, const StaggeredVectorField<float>& velocity_tangent, StaggeredVectorField<float>& output_tangent) {
        cuda_backend::constrain_velocity_jvp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity_tangent), detail::cuda::velocity_boundary(homogeneous(domain.configuration.velocity_boundary)), detail::cuda::staggered(output_tangent));
    }

    void constrain_velocity_vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& velocity_adjoint) {
        cuda_backend::constrain_velocity_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered_adjoint(output_adjoint), detail::cuda::velocity_boundary(domain.configuration.velocity_boundary), detail::cuda::staggered_adjoint(velocity_adjoint));
    }
} // namespace physica::fluids::gas::operators
