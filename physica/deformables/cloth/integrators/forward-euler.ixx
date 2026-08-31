module;

#include <physica/cuda.h>

export module physica.deformables.cloth.integrators.forward_euler;

import physica.deformables.cloth.model;

export namespace physica::deformables::cloth::integrators {
    struct ForwardEuler final {
        struct Configuration final {
            float time_step;
        };

        struct Cache final {};
        struct Workspace final {};

        const Configuration configuration;

        explicit ForwardEuler(Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Model<float>& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model<float>& model) const;

        void forward(const Model<float>& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::ScalarField<float>& masses, const simulation::VectorField<float>& forces, simulation::VectorField<float>& integrated_positions, simulation::VectorField<float>& integrated_velocities, Cache& cache, Workspace& workspace) const;

    };
} // namespace physica::deformables::cloth::integrators
