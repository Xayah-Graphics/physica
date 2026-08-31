module;

#include <physica/cuda.h>

export module physica.deformables.cloth.constraints.provot_strain_limit;

import std;
import physica.deformables.cloth.coloring;
import physica.deformables.cloth.model;

export namespace physica::deformables::cloth::constraints {
    struct ProvotStrainLimitConstraint final {
        struct FixedVertex final {
            std::uint32_t particle;
            Vector3<float> position;
        };

        struct Configuration final {
            float maximum_stretch_ratio;
            std::uint32_t iteration_count;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Cache final {};
        struct Workspace final {};

        ProvotStrainLimitConstraint(const Model<float>& model, Configuration configuration);

        ProvotStrainLimitConstraint(const ProvotStrainLimitConstraint&)            = delete;
        ProvotStrainLimitConstraint& operator=(const ProvotStrainLimitConstraint&) = delete;
        ProvotStrainLimitConstraint(ProvotStrainLimitConstraint&&)                 = delete;
        ProvotStrainLimitConstraint& operator=(ProvotStrainLimitConstraint&&)      = delete;

        [[nodiscard]] Cache allocate_cache(const Model<float>& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model<float>& model) const;

        void forward(const Model<float>& model, const simulation::VectorField<float>& previous_positions, const simulation::VectorField<float>& previous_velocities, const simulation::VectorField<float>& integrated_positions, const simulation::VectorField<float>& integrated_velocities, const simulation::ScalarField<float>& masses, float time_step, simulation::VectorField<float>& projected_positions, simulation::VectorField<float>& reconstructed_velocities, Cache& cache, Workspace& workspace) const;

    private:
        const std::uint32_t iteration_count;
        const EdgeColoring coloring;
        simulation::ScalarField<std::uint32_t> colored_edges;
        simulation::ScalarField<float> maximum_lengths;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
    };
} // namespace physica::deformables::cloth::constraints
