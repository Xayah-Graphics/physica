module;

#include <cudss.h>
#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.projective_dynamics;

import std;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::projective_dynamics {
    struct Solver final {
        struct FixedVertex final {
            std::uint32_t particle;
            Vector3<float> position;
        };

        struct Configuration final {
            float time_step;
            std::uint32_t global_iteration_count;
            Vector3<float> gravity;
            float membrane_stiffness;
            float bending_stiffness;
            std::vector<float> masses;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Parameters final {};

        struct StepCache final {
            simulation::VectorField<float> predicted_positions;
            simulation::VectorField<float> projected_frame_first_columns;
            simulation::VectorField<float> projected_frame_second_columns;
            simulation::VectorField<float> bending_directions;
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
        struct HostSystem final {
            std::vector<float> material_inverse_00;
            std::vector<float> material_inverse_01;
            std::vector<float> material_inverse_10;
            std::vector<float> material_inverse_11;
            std::vector<float> membrane_weights;
            std::vector<float> bending_rest_lengths;
            std::vector<std::uint32_t> fixed_vertex_mask;
            std::vector<Vector3<float>> fixed_positions;
            std::vector<std::uint32_t> free_particles;
            std::vector<Vector3<float>> fixed_right_hand_sides;
            std::vector<std::int32_t> matrix_row_offsets;
            std::vector<std::int32_t> matrix_column_indices;
            std::vector<float> matrix_values;
        };

        [[nodiscard]] static HostSystem build_system(const Model<float>& model, const Configuration& configuration);
        Solver(const Model<float>& model, const Configuration& configuration, HostSystem system);
        void destroy_cudss() noexcept;

        const float time_step;
        const std::uint32_t global_iteration_count;
        const Vector3<float> gravity;
        const float bending_stiffness;
        const std::uint32_t free_particle_count;
        simulation::ScalarField<float> masses;
        simulation::ScalarField<float> material_inverse_00;
        simulation::ScalarField<float> material_inverse_01;
        simulation::ScalarField<float> material_inverse_10;
        simulation::ScalarField<float> material_inverse_11;
        simulation::ScalarField<float> membrane_weights;
        simulation::ScalarField<float> bending_rest_lengths;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
        simulation::ScalarField<std::uint32_t> free_particles;
        simulation::VectorField<float> fixed_right_hand_sides;
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
} // namespace physica::deformables::cloth::solvers::projective_dynamics
