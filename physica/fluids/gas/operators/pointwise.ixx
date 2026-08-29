export module physica.fluids.gas.operators.pointwise;

import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    void add_source_forward(const Domain& domain, const ScalarField<float>& state, const ScalarField<float>& source, ScalarField<float>& output);
    void add_source_jvp(const Domain& domain, const ScalarField<float>& state_tangent, const ScalarField<float>& source_tangent, ScalarField<float>& output_tangent);
    void add_source_vjp(const Domain& domain, const ScalarField<double>& output_adjoint, ScalarField<double>& state_adjoint, ScalarField<double>& source_adjoint);

    void integrate_velocity_forward(const Domain& domain, const VectorField<float>& velocity, const VectorField<float>& acceleration, VectorField<float>& output);
    void integrate_velocity_jvp(const Domain& domain, const VectorField<float>& velocity_tangent, const VectorField<float>& acceleration_tangent, VectorField<float>& output_tangent);
    void integrate_velocity_vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& acceleration_adjoint);

    void constrain_velocity_forward(const Domain& domain, const VectorField<float>& velocity, VectorField<float>& output);
    void constrain_velocity_jvp(const Domain& domain, const VectorField<float>& velocity_tangent, VectorField<float>& output_tangent);
    void constrain_velocity_vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& velocity_adjoint);
} // namespace physica::fluids::gas::operators
