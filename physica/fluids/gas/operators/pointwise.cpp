module;

#include <fluids/gas/interop.h>
#include "pointwise-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.pointwise;

namespace physica::fluids::gas::operators {
    void add_source_forward(const Domain& domain, const simulation::ScalarField<float>& state, const simulation::ScalarField<float>& source, simulation::ScalarField<float>& output) {
        kernels::source_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(state), simulation::scalar_view(source), simulation::scalar_view(output));
    }

    void add_source_jvp(const Domain& domain, const simulation::ScalarField<float>& state_tangent, const simulation::ScalarField<float>& source_tangent, simulation::ScalarField<float>& output_tangent) {
        kernels::source_jvp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(state_tangent), simulation::scalar_view(source_tangent), simulation::scalar_view(output_tangent));
    }

    void add_source_vjp(const Domain& domain, const simulation::ScalarField<double>& output_adjoint, simulation::ScalarField<double>& state_adjoint, simulation::ScalarField<double>& source_adjoint) {
        kernels::source_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::scalar_view(output_adjoint), simulation::scalar_view(state_adjoint), simulation::scalar_view(source_adjoint));
    }

    void integrate_velocity_forward(const Domain& domain, const simulation::VectorField<float>& velocity, const simulation::VectorField<float>& acceleration, simulation::VectorField<float>& output) {
        kernels::integrate_velocity_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity), simulation::view(acceleration), simulation::view(output));
    }

    void integrate_velocity_jvp(const Domain& domain, const simulation::VectorField<float>& velocity_tangent, const simulation::VectorField<float>& acceleration_tangent, simulation::VectorField<float>& output_tangent) {
        kernels::integrate_velocity_jvp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity_tangent), simulation::view(acceleration_tangent), simulation::view(output_tangent));
    }

    void integrate_velocity_vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& acceleration_adjoint) {
        kernels::integrate_velocity_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(output_adjoint), simulation::view(velocity_adjoint), simulation::view(acceleration_adjoint));
    }

    void constrain_velocity_forward(const Domain& domain, const simulation::VectorField<float>& velocity, simulation::VectorField<float>& output) {
        kernels::constrain_velocity_forward(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(domain.collider_velocity), simulation::view(velocity), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::view(output));
    }

    void constrain_velocity_jvp(const Domain& domain, const simulation::VectorField<float>& velocity_tangent, simulation::VectorField<float>& output_tangent) {
        kernels::constrain_velocity_jvp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(velocity_tangent), device::velocity_boundary(homogeneous(domain.configuration.velocity_boundary)), simulation::view(output_tangent));
    }

    void constrain_velocity_vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& velocity_adjoint) {
        kernels::constrain_velocity_vjp(domain.grid.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), simulation::view(output_adjoint), device::velocity_boundary(domain.configuration.velocity_boundary), simulation::view(velocity_adjoint));
    }
} // namespace physica::fluids::gas::operators
