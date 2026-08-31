module;

#include <cudss.h>
#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.fast_mass_spring;

import std;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::fast_mass_spring {
    struct Solver final {
        struct FixedVertex final {
            std::uint32_t particle;
            Vector3<float> position;
        };

        struct Configuration final {
            float time_step;
            std::uint32_t global_iteration_count;
            Vector3<float> gravity;
            float spring_stiffness;
            std::vector<float> masses;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Parameters final {};

        struct StepCache final {
            simulation::VectorField<float> predicted_positions;
            simulation::VectorField<float> projected_springs;
        };

        struct Workspace final {};

        Solver(const Model<float>& model, Configuration configuration);
        ~Solver();

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
        void destroy_cudss() noexcept;

        const float time_step;
        const std::uint32_t global_iteration_count;
        const Vector3<float> gravity;
        const float spring_stiffness;
        const std::uint32_t free_particle_count;
        simulation::ScalarField<float> masses;
        simulation::ScalarField<float> rest_lengths;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
        simulation::ScalarField<std::uint32_t> free_particles;
        simulation::ScalarField<std::int32_t> matrix_row_offsets;
        simulation::ScalarField<std::int32_t> matrix_column_indices;
        simulation::ScalarField<float> matrix_values;
        mutable simulation::ScalarField<float> right_hand_sides;
        mutable simulation::ScalarField<float> solutions;
        cudssHandle_t cudss_handle;
        cudssConfig_t cudss_configuration;
        cudssData_t cudss_data;
        cudssMatrix_t system_matrix;
        std::array<cudssMatrix_t, 3uz> right_hand_side_matrices;
        std::array<cudssMatrix_t, 3uz> solution_matrices;
    };
} // namespace physica::deformables::cloth::solvers::fast_mass_spring
