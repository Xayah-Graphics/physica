module;

#include <physica/cuda.h>

export module physica.fluids.liquid.solvers.pic.transfer;

import std;
import physica.fluids.liquid.solvers.pic.model;
import physica.fluids.liquid.solvers.pic.particle_step;

export namespace physica::fluids::liquid::solvers::pic {
    struct FlipTransfer final {
        struct Configuration final {
            float ratio{0.95F};
        };

        struct State final {};
        struct Workspace final {};

        explicit FlipTransfer(Configuration configuration);

        [[nodiscard]] State allocate_state(const Model& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model& model) const;
        void clear_state(const Model& model, State& state) const;
        void copy_state(const Model& model, const State& source, State& destination) const;
        void particle_to_grid(const Model& model, float time, std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const State& state, simulation::VectorField<float>& momentum, simulation::VectorField<float>& mass) const;
        void grid_to_particle(const Model& model, float time, std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& input_velocities, const State& input_state, const simulation::VectorField<float>& old_grid_velocity, const simulation::VectorField<float>& new_grid_velocity, simulation::VectorField<float>& output_velocities, State& output_state) const;
        void compact_and_seed(const Model& model, float time, std::uint32_t source_particle_count, const ParticleStep::Maintenance& maintenance, const simulation::VectorField<float>& compacted_positions, const simulation::VectorField<float>& grid_velocity, const State& source, State& destination, const simulation::ScalarField<std::uint32_t>& keep_flags, const simulation::ScalarField<std::uint32_t>& destinations, Workspace& workspace) const;
        void commit_compaction(State& state, Workspace& workspace) const;

        const Configuration configuration;
    };

    struct ApicTransfer final {
        struct Configuration final {
            float affine_ratio{1.0F};
        };

        struct State final {
            simulation::Matrix3Field<float> affine;
        };

        struct Workspace final {
            simulation::Matrix3Field<float> compacted_affine;
        };

        explicit ApicTransfer(Configuration configuration);

        [[nodiscard]] State allocate_state(const Model& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model& model) const;
        void clear_state(const Model& model, State& state) const;
        void copy_state(const Model& model, const State& source, State& destination) const;
        void particle_to_grid(const Model& model, float time, std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const State& state, simulation::VectorField<float>& momentum, simulation::VectorField<float>& mass) const;
        void grid_to_particle(const Model& model, float time, std::uint32_t particle_count, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& input_velocities, const State& input_state, const simulation::VectorField<float>& old_grid_velocity, const simulation::VectorField<float>& new_grid_velocity, simulation::VectorField<float>& output_velocities, State& output_state) const;
        void compact_and_seed(const Model& model, float time, std::uint32_t source_particle_count, const ParticleStep::Maintenance& maintenance, const simulation::VectorField<float>& compacted_positions, const simulation::VectorField<float>& grid_velocity, const State& source, State& destination, const simulation::ScalarField<std::uint32_t>& keep_flags, const simulation::ScalarField<std::uint32_t>& destinations, Workspace& workspace) const;
        void commit_compaction(State& state, Workspace& workspace) const;

        const Configuration configuration;
    };
} // namespace physica::fluids::liquid::solvers::pic
