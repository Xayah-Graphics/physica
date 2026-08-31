module;

#include <physica/cuda.h>

export module physica.deformables.cloth.constraints.fixed_position;

import std;
import physica.deformables.cloth.model;

export namespace physica::deformables::cloth::constraints {
    struct FixedPositionConstraint final {
        struct Anchor final {
            std::uint32_t particle;
            Vector3<float> position;
        };

        struct Configuration final {
            std::vector<Anchor> anchors;
        };

        struct Cache final {};
        struct Workspace final {};
        struct TangentWorkspace final {};
        struct AdjointWorkspace final {};

        FixedPositionConstraint(const Model<float>& model, Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Model<float>& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model<float>& model) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Model<float>& model) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Model<float>& model) const;

        void forward(const Model<float>& model, const simulation::VectorField<float>& previous_positions, const simulation::VectorField<float>& previous_velocities, const simulation::VectorField<float>& integrated_positions, const simulation::VectorField<float>& integrated_velocities, const simulation::ScalarField<float>& masses, float time_step, simulation::VectorField<float>& constrained_positions, simulation::VectorField<float>& constrained_velocities, Cache& cache, Workspace& workspace) const;
        void jvp(const Model<float>& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>& constrained_positions, const simulation::VectorField<float>& constrained_velocities, const Cache& cache, const simulation::VectorField<float>& position_tangent, const simulation::VectorField<float>& velocity_tangent, simulation::VectorField<float>& constrained_position_tangent, simulation::VectorField<float>& constrained_velocity_tangent, TangentWorkspace& workspace) const;
        void vjp(const Model<float>& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>& constrained_positions, const simulation::VectorField<float>& constrained_velocities, const Cache& cache, const simulation::VectorField<double>& constrained_position_adjoint, const simulation::VectorField<double>& constrained_velocity_adjoint, simulation::VectorField<double>& position_adjoint, simulation::VectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const;

    private:
        simulation::ScalarField<std::uint32_t> anchor_mask;
        simulation::VectorField<float> anchor_positions;
    };
} // namespace physica::deformables::cloth::constraints
