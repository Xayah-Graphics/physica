module;

#include <physica/cuda.h>

export module physica.deformables.cloth.integrators.semi_implicit_euler;

import physica.deformables.cloth.model;

export namespace physica::deformables::cloth::integrators {
    struct SemiImplicitEuler final {
        struct Configuration final {
            float time_step;
        };

        struct Cache final {};
        struct Workspace final {};
        struct TangentWorkspace final {};
        struct AdjointWorkspace final {};

        const Configuration configuration;

        explicit SemiImplicitEuler(Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Model<float>& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model<float>& model) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Model<float>& model) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Model<float>& model) const;

        void forward(const Model<float>& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::ScalarField<float>& masses, const simulation::VectorField<float>& forces, simulation::VectorField<float>& integrated_positions, simulation::VectorField<float>& integrated_velocities, Cache& cache, Workspace& workspace) const;
        void jvp(const Model<float>& model, const simulation::ScalarField<float>& masses, const simulation::VectorField<float>& forces, const Cache& cache, const simulation::VectorField<float>& position_tangent, const simulation::VectorField<float>& velocity_tangent, const simulation::ScalarField<float>& mass_tangent, const simulation::VectorField<float>& force_tangent, simulation::VectorField<float>& integrated_position_tangent, simulation::VectorField<float>& integrated_velocity_tangent, TangentWorkspace& workspace) const;
        void vjp(const Model<float>& model, const simulation::ScalarField<float>& masses, const simulation::VectorField<float>& forces, const Cache& cache, const simulation::VectorField<double>& integrated_position_adjoint, const simulation::VectorField<double>& integrated_velocity_adjoint, simulation::VectorField<double>& position_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& force_adjoint, simulation::ScalarField<double>& mass_adjoint, AdjointWorkspace& workspace) const;

    };
} // namespace physica::deformables::cloth::integrators
