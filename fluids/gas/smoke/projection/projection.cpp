module;

#include "../domain/interop.h"
#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

module physica.fluids.gas.smoke.projection;

import std;

namespace physica::fluids::gas::smoke {
    namespace projection_detail {
        void pressure_rhs_forward(const Domain& domain, const StaggeredVectorField& velocity, ScalarField& rhs) {
            cuda_detail::pressure_rhs_forward(domain.stream, cuda_detail::grid(domain.configuration), domain.pressure_anchor, domain.cell_mask.data(), cuda_detail::staggered(velocity), cuda_detail::scalar(rhs));
        }

        void pressure_rhs_vjp(const Domain& domain, const ScalarAdjointField& rhs_adjoint, StaggeredVectorAdjointField& velocity_adjoint) {
            cuda_detail::pressure_rhs_vjp(domain.stream, cuda_detail::grid(domain.configuration), domain.pressure_anchor, domain.cell_mask.data(), cuda_detail::scalar_adjoint(rhs_adjoint), cuda_detail::staggered_adjoint(velocity_adjoint));
        }

        void project_velocity_forward(const Domain& domain, const StaggeredVectorField& velocity, const ScalarField& pressure, StaggeredVectorField& output) {
            cuda_detail::project_velocity_forward(domain.stream, cuda_detail::grid(domain.configuration), domain.cell_mask.data(), cuda_detail::staggered(velocity), cuda_detail::scalar(pressure), cuda_detail::staggered(output));
        }

        void project_velocity_vjp(const Domain& domain, const StaggeredVectorAdjointField& output_adjoint, StaggeredVectorAdjointField& velocity_adjoint, ScalarAdjointField& pressure_adjoint) {
            cuda_detail::project_velocity_vjp(domain.stream, cuda_detail::grid(domain.configuration), domain.cell_mask.data(), cuda_detail::staggered_adjoint(output_adjoint), cuda_detail::staggered_adjoint(velocity_adjoint), cuda_detail::scalar_adjoint(pressure_adjoint));
        }
    } // namespace projection_detail

    RedBlackGaussSeidel::RedBlackGaussSeidel(Configuration next_configuration)
        : configuration(std::move(next_configuration)) {}

    void RedBlackGaussSeidel::forward(const Domain& domain, const ScalarBoundary& boundary, const ScalarField& rhs, ScalarField& pressure) const {
        cuda_detail::red_black_gauss_seidel_forward(domain.stream, cuda_detail::grid(domain.configuration), configuration.iterations, domain.pressure_anchor, domain.cell_mask.data(), cuda_detail::scalar_boundary(boundary), cuda_detail::scalar(rhs), cuda_detail::scalar(pressure));
    }

    void RedBlackGaussSeidel::vjp(const Domain& domain, const ScalarBoundary& boundary, ScalarAdjointField& pressure_adjoint, ScalarAdjointField& rhs_adjoint) const {
        cuda_detail::red_black_gauss_seidel_vjp(domain.stream, cuda_detail::grid(domain.configuration), configuration.iterations, domain.pressure_anchor, domain.cell_mask.data(), cuda_detail::scalar_boundary(boundary), cuda_detail::scalar_adjoint(pressure_adjoint), cuda_detail::scalar_adjoint(rhs_adjoint));
    }
} // namespace physica::fluids::gas::smoke
