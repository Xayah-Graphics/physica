module;

#include <physica/cuda.h>

export module physica.deformables.cloth.operators.semi_implicit_euler;

import physica.deformables.cloth.domain;

export namespace physica::deformables::cloth::operators {
    struct SemiImplicitEuler final {
        struct Configuration final {
            float time_step;
        };

        struct Cache final {};
        struct Workspace final {};
        struct TangentWorkspace final {};
        struct AdjointWorkspace final {};

        explicit SemiImplicitEuler(Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const ScalarField<float>& masses, const VectorField<float>& forces, VectorField<float>& integrated_positions, VectorField<float>& integrated_velocities, Cache& cache, Workspace& workspace) const;
        void jvp(const Domain& domain, const ScalarField<float>& masses, const VectorField<float>& forces, const Cache& cache, const VectorField<float>& position_tangent, const VectorField<float>& velocity_tangent, const ScalarField<float>& mass_tangent, const VectorField<float>& force_tangent, VectorField<float>& integrated_position_tangent, VectorField<float>& integrated_velocity_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const ScalarField<float>& masses, const VectorField<float>& forces, const Cache& cache, const VectorField<double>& integrated_position_adjoint, const VectorField<double>& integrated_velocity_adjoint, VectorField<double>& position_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& force_adjoint, ScalarField<double>& mass_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };
} // namespace physica::deformables::cloth::operators
