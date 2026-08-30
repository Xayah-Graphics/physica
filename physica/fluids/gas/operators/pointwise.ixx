export module physica.fluids.gas.operators.pointwise;

import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    void add_source_forward(const Domain& domain, const simulation::ScalarField<float>& state, const simulation::ScalarField<float>& source, simulation::ScalarField<float>& output);
    void add_source_jvp(const Domain& domain, const simulation::ScalarField<float>& state_tangent, const simulation::ScalarField<float>& source_tangent, simulation::ScalarField<float>& output_tangent);
    void add_source_vjp(const Domain& domain, const simulation::ScalarField<double>& output_adjoint, simulation::ScalarField<double>& state_adjoint, simulation::ScalarField<double>& source_adjoint);

    void integrate_velocity_forward(const Domain& domain, const simulation::VectorField<float>& velocity, const simulation::VectorField<float>& acceleration, simulation::VectorField<float>& output);
    void integrate_velocity_jvp(const Domain& domain, const simulation::VectorField<float>& velocity_tangent, const simulation::VectorField<float>& acceleration_tangent, simulation::VectorField<float>& output_tangent);
    void integrate_velocity_vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& acceleration_adjoint);

    void constrain_velocity_forward(const Domain& domain, const simulation::VectorField<float>& velocity, simulation::VectorField<float>& output);
    void constrain_velocity_jvp(const Domain& domain, const simulation::VectorField<float>& velocity_tangent, simulation::VectorField<float>& output_tangent);
    void constrain_velocity_vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& velocity_adjoint);
} // namespace physica::fluids::gas::operators
