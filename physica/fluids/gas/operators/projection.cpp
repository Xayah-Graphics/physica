module;

#include <fluids/gas/interop.h>
#include "projection-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.projection;

import std;

namespace physica::fluids::gas::operators {
    namespace projection {
        void pressure_rhs_forward(const Domain& domain, const std::uint32_t pressure_anchor, const simulation::VectorField<float>& velocity, simulation::ScalarField<float>& rhs) {
            kernels::pressure_rhs_forward(domain.grid.stream, device::discretization(domain.configuration), pressure_anchor, domain.collider_ids.values.data(), simulation::view(velocity), simulation::scalar_view(rhs));
        }

        void pressure_rhs_vjp(const Domain& domain, const std::uint32_t pressure_anchor, const simulation::ScalarField<double>& rhs_adjoint, simulation::VectorField<double>& velocity_adjoint) {
            kernels::pressure_rhs_vjp(domain.grid.stream, device::discretization(domain.configuration), pressure_anchor, domain.collider_ids.values.data(), simulation::scalar_view(rhs_adjoint), simulation::view(velocity_adjoint));
        }

        void project_velocity_forward(const Domain& domain, const simulation::VectorField<float>& velocity, const simulation::ScalarField<float>& pressure, simulation::VectorField<float>& output) {
            kernels::project_velocity_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity), simulation::scalar_view(pressure), simulation::view(output));
        }

        void project_velocity_vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::ScalarField<double>& pressure_adjoint) {
            kernels::project_velocity_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(output_adjoint), simulation::view(velocity_adjoint), simulation::scalar_view(pressure_adjoint));
        }
    } // namespace projection

    RedBlackGaussSeidel::RedBlackGaussSeidel(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    RedBlackGaussSeidel::Workspace RedBlackGaussSeidel::allocate_workspace(const Domain&) const {
        return {};
    }

    RedBlackGaussSeidel::AdjointWorkspace RedBlackGaussSeidel::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void RedBlackGaussSeidel::forward(const Domain& domain, const ScalarBoundary& boundary, const std::uint32_t pressure_anchor, const simulation::ScalarField<float>& rhs, simulation::ScalarField<float>& pressure, Workspace&) const {
        kernels::red_black_gauss_seidel_forward(domain.grid.stream, device::discretization(domain.configuration), configuration.iterations, pressure_anchor, domain.collider_ids.values.data(), device::scalar_boundary(boundary), simulation::scalar_view(rhs), simulation::scalar_view(pressure));
    }

    void RedBlackGaussSeidel::vjp(const Domain& domain, const ScalarBoundary& boundary, const std::uint32_t pressure_anchor, simulation::ScalarField<double>& pressure_adjoint, simulation::ScalarField<double>& rhs_adjoint, AdjointWorkspace&) const {
        kernels::red_black_gauss_seidel_vjp(domain.grid.stream, device::discretization(domain.configuration), configuration.iterations, pressure_anchor, domain.collider_ids.values.data(), device::scalar_boundary(boundary), simulation::scalar_view(pressure_adjoint), simulation::scalar_view(rhs_adjoint));
    }
} // namespace physica::fluids::gas::operators
