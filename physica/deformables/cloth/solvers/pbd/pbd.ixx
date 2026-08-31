module;

#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.pbd;

import std;
import physica.deformables.cloth.coloring;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::pbd {
    struct Solver final {
        struct FixedVertex final {
            std::uint32_t particle;
            Vector3<float> position;
        };

        struct Configuration final {
            float time_step;
            std::uint32_t iteration_count;
            Vector3<float> gravity;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Parameters final {
            simulation::ScalarField<float> masses;
        };

        struct StepCache final {};
        struct Workspace final {};

        Solver(const Model<float>& model, Configuration configuration);

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State<float> allocate_state(const Model<float>& model) const;
        [[nodiscard]] Control<float> allocate_control(const Model<float>& model) const;
        [[nodiscard]] Parameters allocate_parameters(const Model<float>& model) const;
        [[nodiscard]] StepCache allocate_step_cache(const Model<float>& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model<float>& model) const;

        void forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters& parameters, State<float>& next_state, StepCache& cache, Workspace& workspace) const;

    private:
        const float time_step;
        const std::uint32_t iteration_count;
        const Vector3<float> gravity;
        const EdgeColoring coloring;
        simulation::ScalarField<std::uint32_t> colored_edges;
        simulation::ScalarField<float> rest_lengths;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
    };
} // namespace physica::deformables::cloth::solvers::pbd
