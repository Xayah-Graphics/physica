module;

#include "../detail/cuda/interop.h"
#include "projection-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.projection;

import std;

namespace physica::fluids::gas::operators {
    namespace projection_detail {
        void pressure_rhs_forward(const Domain& domain, const std::uint32_t pressure_anchor, const StaggeredVectorField<float>& velocity, CellField<float>& rhs) {
            cuda_backend::pressure_rhs_forward(domain.stream, detail::cuda::grid(domain.configuration), pressure_anchor, domain.collider_ids.values.data(), detail::cuda::staggered(velocity), detail::cuda::scalar(rhs));
        }

        void pressure_rhs_vjp(const Domain& domain, const std::uint32_t pressure_anchor, const CellField<double>& rhs_adjoint, StaggeredVectorField<double>& velocity_adjoint) {
            cuda_backend::pressure_rhs_vjp(domain.stream, detail::cuda::grid(domain.configuration), pressure_anchor, domain.collider_ids.values.data(), detail::cuda::scalar_adjoint(rhs_adjoint), detail::cuda::staggered_adjoint(velocity_adjoint));
        }

        void project_velocity_forward(const Domain& domain, const StaggeredVectorField<float>& velocity, const CellField<float>& pressure, StaggeredVectorField<float>& output) {
            cuda_backend::project_velocity_forward(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered(velocity), detail::cuda::scalar(pressure), detail::cuda::staggered(output));
        }

        void project_velocity_vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& velocity_adjoint, CellField<double>& pressure_adjoint) {
            cuda_backend::project_velocity_vjp(domain.stream, detail::cuda::grid(domain.configuration), domain.collider_ids.values.data(), detail::cuda::staggered_adjoint(output_adjoint), detail::cuda::staggered_adjoint(velocity_adjoint), detail::cuda::scalar_adjoint(pressure_adjoint));
        }
    } // namespace projection_detail

    RedBlackGaussSeidel::RedBlackGaussSeidel(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    RedBlackGaussSeidel::Workspace RedBlackGaussSeidel::allocate_workspace(const Domain&) const {
        return {};
    }

    RedBlackGaussSeidel::AdjointWorkspace RedBlackGaussSeidel::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void RedBlackGaussSeidel::forward(const Domain& domain, const ScalarBoundary& boundary, const std::uint32_t pressure_anchor, const CellField<float>& rhs, CellField<float>& pressure, Workspace&) const {
        cuda_backend::red_black_gauss_seidel_forward(domain.stream, detail::cuda::grid(domain.configuration), configuration.iterations, pressure_anchor, domain.collider_ids.values.data(), detail::cuda::scalar_boundary(boundary), detail::cuda::scalar(rhs), detail::cuda::scalar(pressure));
    }

    void RedBlackGaussSeidel::vjp(const Domain& domain, const ScalarBoundary& boundary, const std::uint32_t pressure_anchor, CellField<double>& pressure_adjoint, CellField<double>& rhs_adjoint, AdjointWorkspace&) const {
        cuda_backend::red_black_gauss_seidel_vjp(domain.stream, detail::cuda::grid(domain.configuration), configuration.iterations, pressure_anchor, domain.collider_ids.values.data(), detail::cuda::scalar_boundary(boundary), detail::cuda::scalar_adjoint(pressure_adjoint), detail::cuda::scalar_adjoint(rhs_adjoint));
    }
} // namespace physica::fluids::gas::operators
