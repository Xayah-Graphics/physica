module;

#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.baraff_witkin;

import std;
import physica.deformables.cloth.coloring;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.solvers.block_pcg;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::baraff_witkin {
    struct Solver final {
        struct FixedVertex final {
            std::uint32_t particle;
            Vector3<float> position;
        };

        struct Configuration final {
            float time_step;
            std::uint32_t pcg_iteration_count;
            Vector3<float> gravity;
            float stretch_u_target;
            float stretch_v_target;
            float stretch_u_stiffness;
            float stretch_v_stiffness;
            float shear_stiffness;
            float bend_u_stiffness;
            float bend_v_stiffness;
            float stretch_u_damping;
            float stretch_v_damping;
            float shear_damping;
            float bend_u_damping;
            float bend_v_damping;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Parameters final {
            simulation::ScalarField<float> masses;
        };

        struct StepCache final {
            simulation::VectorField<float> triangle_conditions;
            simulation::ScalarField<float> bending_angles;
            simulation::VectorField<float> forces;
            block_pcg::BlockCsrMatrix force_position_derivative;
            block_pcg::BlockCsrMatrix force_velocity_derivative;
            block_pcg::BlockCsrMatrix system;
            simulation::VectorField<float> right_hand_side;
            simulation::VectorField<float> constraint_velocity_change;
            simulation::VectorField<float> velocity_increment;
        };

        struct Workspace final {
            simulation::VectorField<float> matrix_times_constraint_velocity_change;
            simulation::VectorField<float> reduced_right_hand_side;
            simulation::VectorField<float> free_velocity_change;
            block_pcg::Solver::Workspace pcg;
        };

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
        struct Pattern final {
            std::vector<std::uint32_t> row_offsets;
            std::vector<std::uint32_t> column_indices;
        };

        struct HingeColoring final {
            std::vector<std::uint32_t> offsets;
            std::vector<std::uint32_t> hinges;
        };

        [[nodiscard]] static Pattern build_pattern(const Model<float>& model);
        [[nodiscard]] static HingeColoring build_hinge_coloring(const Model<float>& model);
        [[nodiscard]] static std::uint32_t find_block(const Pattern& pattern, std::uint32_t row, std::uint32_t column);
        [[nodiscard]] static MaterialCoordinate<float> material_coordinate(const Model<float>& model, std::uint32_t triangle, std::uint32_t particle);
        [[nodiscard]] static float signed_dihedral(const std::vector<Vector3<float>>& positions, const Hinge& hinge);

        const float time_step;
        const Vector3<float> gravity;
        const float stretch_u_target;
        const float stretch_v_target;
        const float stretch_u_stiffness;
        const float stretch_v_stiffness;
        const float shear_stiffness;
        const float stretch_u_damping;
        const float stretch_v_damping;
        const float shear_damping;
        const Pattern pattern;
        const TriangleColoring triangle_coloring;
        const HingeColoring hinge_coloring;
        block_pcg::Solver block_solver;
        simulation::VectorField<float> triangle_u_coefficients;
        simulation::VectorField<float> triangle_v_coefficients;
        simulation::ScalarField<float> triangle_areas;
        simulation::ScalarField<float> hinge_rest_angles;
        simulation::ScalarField<float> hinge_stiffnesses;
        simulation::ScalarField<float> hinge_dampings;
        simulation::ScalarField<std::uint32_t> colored_triangles;
        simulation::ScalarField<std::uint32_t> colored_hinges;
        simulation::ScalarField<std::uint32_t> triangle_block_indices;
        simulation::ScalarField<std::uint32_t> hinge_block_indices;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
    };
} // namespace physica::deformables::cloth::solvers::baraff_witkin
