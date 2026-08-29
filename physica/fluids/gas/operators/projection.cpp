module;

#include <physica/fluids/gas/interop.h>
#include "projection-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.projection;

import std;

namespace physica::fluids::gas::operators {
    namespace projection {
        void pressure_rhs_forward(const Domain& domain, const std::uint32_t pressure_anchor, const VectorField<float>& velocity, ScalarField<float>& rhs) {
            kernels::pressure_rhs_forward(domain.grid.fields.stream, device::discretization(domain.configuration), pressure_anchor, domain.collider_ids.values.data(), field::view(velocity), field::scalar_view(rhs));
        }

        void pressure_rhs_vjp(const Domain& domain, const std::uint32_t pressure_anchor, const ScalarField<double>& rhs_adjoint, VectorField<double>& velocity_adjoint) {
            kernels::pressure_rhs_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), pressure_anchor, domain.collider_ids.values.data(), field::scalar_view(rhs_adjoint), field::view(velocity_adjoint));
        }

        void project_velocity_forward(const Domain& domain, const VectorField<float>& velocity, const ScalarField<float>& pressure, VectorField<float>& output) {
            kernels::project_velocity_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity), field::scalar_view(pressure), field::view(output));
        }

        void project_velocity_vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& velocity_adjoint, ScalarField<double>& pressure_adjoint) {
            kernels::project_velocity_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(output_adjoint), field::view(velocity_adjoint), field::scalar_view(pressure_adjoint));
        }
    } // namespace projection

    RedBlackGaussSeidel::RedBlackGaussSeidel(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    RedBlackGaussSeidel::Workspace RedBlackGaussSeidel::allocate_workspace(const Domain&) const {
        return {};
    }

    RedBlackGaussSeidel::AdjointWorkspace RedBlackGaussSeidel::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void RedBlackGaussSeidel::forward(const Domain& domain, const ScalarBoundary& boundary, const std::uint32_t pressure_anchor, const ScalarField<float>& rhs, ScalarField<float>& pressure, Workspace&) const {
        kernels::red_black_gauss_seidel_forward(domain.grid.fields.stream, device::discretization(domain.configuration), configuration.iterations, pressure_anchor, domain.collider_ids.values.data(), device::scalar_boundary(boundary), field::scalar_view(rhs), field::scalar_view(pressure));
    }

    void RedBlackGaussSeidel::vjp(const Domain& domain, const ScalarBoundary& boundary, const std::uint32_t pressure_anchor, ScalarField<double>& pressure_adjoint, ScalarField<double>& rhs_adjoint, AdjointWorkspace&) const {
        kernels::red_black_gauss_seidel_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), configuration.iterations, pressure_anchor, domain.collider_ids.values.data(), device::scalar_boundary(boundary), field::scalar_view(pressure_adjoint), field::scalar_view(rhs_adjoint));
    }
} // namespace physica::fluids::gas::operators
