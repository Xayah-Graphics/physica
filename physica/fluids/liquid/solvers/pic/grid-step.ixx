module;

#include <physica/cuda.h>

export module physica.fluids.liquid.solvers.pic.grid_step;

import std;
import physica.fluids.liquid.solvers.pic.model;

export namespace physica::fluids::liquid::solvers::pic {
    struct GridStep final {
        struct Configuration final {
            Vector3<float> acceleration{0.0F, -9.81F, 0.0F};
            float level_set_radius_cells{0.75F};
            std::uint32_t extrapolation_layers{4u};
        };

        struct Diagnostics final {
            float l2{};
            float maximum{};
        };

        struct State final {
            simulation::VectorField<float> velocity_before_projection;
            simulation::VectorField<float> velocity;
            simulation::ScalarField<std::uint32_t> cell_types;
            simulation::ScalarField<float> level_set;
            simulation::ScalarField<float> divergence;
            simulation::ScalarField<float> pressure;
        };

        struct Workspace final {
            simulation::VectorField<float> face_mass;
            simulation::VectorField<std::uint32_t> valid_faces;
            simulation::VectorField<float> projection_input_velocity;
            simulation::VectorField<float> velocity_scratch;
            simulation::VectorField<std::uint32_t> valid_faces_scratch;
            simulation::ScalarField<std::uint32_t> particle_counts;
            ::cuda::device_buffer<float> divergence_metrics;
        };

        explicit GridStep(Configuration configuration);

        [[nodiscard]] State allocate_state(const Model& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model& model) const;
        void begin_transfer(const Model& model, State& state, Workspace& workspace) const;
        void classify_and_normalize(const Model& model, float time, std::uint32_t particle_count, const simulation::VectorField<float>& positions, State& state, Workspace& workspace) const;
        void extrapolate_before_projection(const Model& model, float time, State& state, Workspace& workspace) const;
        void apply_force_and_constrain(const Model& model, float time, float time_step, const simulation::ScalarField<std::uint32_t>& cell_types, simulation::VectorField<float>& velocity) const;
        void prepare_after_projection(const Model& model, float time, State& state, Workspace& workspace) const;
        [[nodiscard]] Diagnostics divergence(const Model& model, float time, const simulation::ScalarField<std::uint32_t>& cell_types, const simulation::VectorField<float>& velocity, simulation::ScalarField<float>& values, Workspace& workspace) const;

        const Configuration configuration;
    };
} // namespace physica::fluids::liquid::solvers::pic
