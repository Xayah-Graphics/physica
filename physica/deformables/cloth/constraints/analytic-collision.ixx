module;

#include <physica/cuda.h>

export module physica.deformables.cloth.constraints.analytic_collision;

import std;
import physica.deformables.cloth.model;

export namespace physica::deformables::cloth::constraints {
    struct AnalyticCollisionConstraint final {
        struct Plane final {
            Vector3<float> normal;
            float offset;
            float restitution;
            float friction;
        };

        struct Sphere final {
            Vector3<float> center;
            float radius;
            float restitution;
            float friction;
        };

        struct Configuration final {
            float thickness;
            std::vector<Plane> planes;
            std::vector<Sphere> spheres;
        };

        struct Cache final {};
        struct Workspace final {};

        AnalyticCollisionConstraint(const Model<float>& model, Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Model<float>& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model<float>& model) const;

        void forward(const Model<float>& model, const simulation::VectorField<float>& previous_positions, const simulation::VectorField<float>& previous_velocities, const simulation::VectorField<float>& integrated_positions, const simulation::VectorField<float>& integrated_velocities, const simulation::ScalarField<float>& masses, float time_step, simulation::VectorField<float>& constrained_positions, simulation::VectorField<float>& constrained_velocities, Cache& cache, Workspace& workspace) const;

    private:
        const Configuration configuration;
    };
} // namespace physica::deformables::cloth::constraints
