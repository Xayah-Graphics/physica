export module physica.fluids.gas.operators.pointwise;

import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    void add_source_forward(const Domain& domain, const CellField<float>& state, const CellField<float>& source, CellField<float>& output);
    void add_source_jvp(const Domain& domain, const CellField<float>& state_tangent, const CellField<float>& source_tangent, CellField<float>& output_tangent);
    void add_source_vjp(const Domain& domain, const CellField<double>& output_adjoint, CellField<double>& state_adjoint, CellField<double>& source_adjoint);

    void integrate_velocity_forward(const Domain& domain, const StaggeredVectorField<float>& velocity, const CenteredVectorField<float>& acceleration, StaggeredVectorField<float>& output);
    void integrate_velocity_jvp(const Domain& domain, const StaggeredVectorField<float>& velocity_tangent, const CenteredVectorField<float>& acceleration_tangent, StaggeredVectorField<float>& output_tangent);
    void integrate_velocity_vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& velocity_adjoint, CenteredVectorField<double>& acceleration_adjoint);

    void constrain_velocity_forward(const Domain& domain, const StaggeredVectorField<float>& velocity, StaggeredVectorField<float>& output);
    void constrain_velocity_jvp(const Domain& domain, const StaggeredVectorField<float>& velocity_tangent, StaggeredVectorField<float>& output_tangent);
    void constrain_velocity_vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& velocity_adjoint);
} // namespace physica::fluids::gas::operators
