module;

#include <physica/cuda.h>

export module physica.deformables.cloth.operators.fixed_position;

import std;
import physica.deformables.cloth.domain;

export namespace physica::deformables::cloth::operators {
    struct FixedPositionConstraint final {
        struct Anchor final {
            std::uint32_t particle;
            Vector3 position;
        };

        struct Configuration final {
            std::vector<Anchor> anchors;
        };

        struct Cache final {};
        struct Workspace final {};
        struct TangentWorkspace final {};
        struct AdjointWorkspace final {};

        FixedPositionConstraint(const Domain& domain, Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, VectorField<float>& constrained_positions, VectorField<float>& constrained_velocities, Cache& cache, Workspace& workspace) const;
        void jvp(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>& constrained_positions, const VectorField<float>& constrained_velocities, const Cache& cache, const VectorField<float>& position_tangent, const VectorField<float>& velocity_tangent, VectorField<float>& constrained_position_tangent, VectorField<float>& constrained_velocity_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>& constrained_positions, const VectorField<float>& constrained_velocities, const Cache& cache, const VectorField<double>& constrained_position_adjoint, const VectorField<double>& constrained_velocity_adjoint, VectorField<double>& position_adjoint, VectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const;

    private:
        ScalarField<std::uint32_t> anchor_mask;
        VectorField<float> anchor_positions;
    };
} // namespace physica::deformables::cloth::operators
