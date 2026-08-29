module;

#include <physica/fluids/gas/interop.h>
#include "pointwise-kernels.h"
#include <physica/cuda.h>

module physica.fluids.gas.operators.pointwise;

namespace physica::fluids::gas::operators {
    void add_source_forward(const Domain& domain, const ScalarField<float>& state, const ScalarField<float>& source, ScalarField<float>& output) {
        kernels::source_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(state), field::scalar_view(source), field::scalar_view(output));
    }

    void add_source_jvp(const Domain& domain, const ScalarField<float>& state_tangent, const ScalarField<float>& source_tangent, ScalarField<float>& output_tangent) {
        kernels::source_jvp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(state_tangent), field::scalar_view(source_tangent), field::scalar_view(output_tangent));
    }

    void add_source_vjp(const Domain& domain, const ScalarField<double>& output_adjoint, ScalarField<double>& state_adjoint, ScalarField<double>& source_adjoint) {
        kernels::source_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::scalar_view(output_adjoint), field::scalar_view(state_adjoint), field::scalar_view(source_adjoint));
    }

    void integrate_velocity_forward(const Domain& domain, const VectorField<float>& velocity, const VectorField<float>& acceleration, VectorField<float>& output) {
        kernels::integrate_velocity_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity), field::view(acceleration), field::view(output));
    }

    void integrate_velocity_jvp(const Domain& domain, const VectorField<float>& velocity_tangent, const VectorField<float>& acceleration_tangent, VectorField<float>& output_tangent) {
        kernels::integrate_velocity_jvp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity_tangent), field::view(acceleration_tangent), field::view(output_tangent));
    }

    void integrate_velocity_vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& acceleration_adjoint) {
        kernels::integrate_velocity_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(output_adjoint), field::view(velocity_adjoint), field::view(acceleration_adjoint));
    }

    void constrain_velocity_forward(const Domain& domain, const VectorField<float>& velocity, VectorField<float>& output) {
        kernels::constrain_velocity_forward(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(domain.collider_velocity), field::view(velocity), device::velocity_boundary(domain.configuration.velocity_boundary), field::view(output));
    }

    void constrain_velocity_jvp(const Domain& domain, const VectorField<float>& velocity_tangent, VectorField<float>& output_tangent) {
        kernels::constrain_velocity_jvp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(velocity_tangent), device::velocity_boundary(homogeneous(domain.configuration.velocity_boundary)), field::view(output_tangent));
    }

    void constrain_velocity_vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& velocity_adjoint) {
        kernels::constrain_velocity_vjp(domain.grid.fields.stream, device::discretization(domain.configuration), domain.collider_ids.values.data(), field::view(output_adjoint), device::velocity_boundary(domain.configuration.velocity_boundary), field::view(velocity_adjoint));
    }
} // namespace physica::fluids::gas::operators
