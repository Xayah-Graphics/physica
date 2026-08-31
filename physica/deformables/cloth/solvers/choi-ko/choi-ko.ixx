module;

#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.choi_ko;

import std;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.solvers.block_pcg;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::choi_ko {
    struct State final {
        simulation::VectorField<float> positions;
        simulation::VectorField<float> velocities;
        simulation::VectorField<float> previous_positions;
        simulation::VectorField<float> previous_velocities;

        State(const ::cuda::stream_ref stream, std::size_t particle_count);

        State(const State&)            = delete;
        State& operator=(const State&) = delete;
        State(State&&)                 = default;
        State& operator=(State&&)      = default;
    };

    struct Solver final {
        struct FixedVertex final {
            std::uint32_t particle;
            Vector3<float> position;
        };

        struct Configuration final {
            float time_step;
            std::uint32_t pcg_iteration_count;
            Vector3<float> gravity;
            float stretch_u_stiffness;
            float stretch_v_stiffness;
            float diagonal_u_stiffness;
            float diagonal_v_stiffness;
            float bend_u_stiffness;
            float bend_v_stiffness;
            float imperfection_stiffness;
            float stretch_u_damping;
            float stretch_v_damping;
            float diagonal_u_damping;
            float diagonal_v_damping;
            float bending_damping;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Parameters final {
            simulation::ScalarField<float> masses;
        };

        struct StepCache final {
            simulation::ScalarField<float> triangle_conditions;
            simulation::ScalarField<float> hinge_curvatures;
            simulation::ScalarField<float> hinge_curvature_first_derivatives;
            simulation::ScalarField<float> hinge_curvature_second_derivatives;
            simulation::ScalarField<float> hinge_responses;
            simulation::ScalarField<float> hinge_response_derivatives;
            simulation::VectorField<float> forces;
            block_pcg::BlockCsrMatrix symmetric_force_position_derivative;
            block_pcg::BlockCsrMatrix force_velocity_derivative;
            block_pcg::BlockCsrMatrix system;
            simulation::VectorField<float> right_hand_side;
            simulation::VectorField<float> prescribed_displacement;
            simulation::VectorField<float> solution;
            simulation::VectorField<float> bdf2_displacement;
        };

        struct Workspace final {
            simulation::VectorField<float> local_forces;
            simulation::ScalarField<float> local_symmetric_force_position_derivatives;
            simulation::ScalarField<float> local_force_velocity_derivatives;
            simulation::VectorField<float> system_times_prescribed_displacement;
            simulation::VectorField<float> reduced_right_hand_side;
            block_pcg::Solver::Workspace pcg;
        };

        Solver(const Model<float>& model, Configuration configuration);

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State allocate_state(const Model<float>& model) const;
        [[nodiscard]] Control<float> allocate_control(const Model<float>& model) const;
        [[nodiscard]] Parameters allocate_parameters(const Model<float>& model) const;
        [[nodiscard]] StepCache allocate_step_cache(const Model<float>& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model<float>& model) const;

        void forward(const Model<float>& model, const State& state, const Control<float>& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const;

    private:
        struct Pattern final {
            std::vector<std::uint32_t> row_offsets;
            std::vector<std::uint32_t> column_indices;
            std::vector<std::uint32_t> force_contribution_offsets;
            std::vector<std::uint32_t> force_contribution_indices;
            std::vector<std::uint32_t> block_contribution_offsets;
            std::vector<std::uint32_t> block_contribution_indices;
        };

        [[nodiscard]] static Pattern build_pattern(const Model<float>& model);
        [[nodiscard]] static std::uint32_t find_block(const Pattern& pattern, std::uint32_t row, std::uint32_t column);
        [[nodiscard]] static MaterialCoordinate<float> material_coordinate(const Model<float>& model, std::uint32_t triangle, std::uint32_t particle);

        const float time_step;
        const Vector3<float> gravity;
        const float stretch_u_stiffness;
        const float stretch_v_stiffness;
        const float diagonal_u_stiffness;
        const float diagonal_v_stiffness;
        const float imperfection_stiffness;
        const float stretch_u_damping;
        const float stretch_v_damping;
        const float diagonal_u_damping;
        const float diagonal_v_damping;
        const float bending_damping;
        const Pattern pattern;
        block_pcg::Solver block_solver;
        simulation::VectorField<float> triangle_direction_coefficients;
        simulation::ScalarField<float> triangle_areas;
        simulation::ScalarField<float> hinge_rest_spans;
        simulation::ScalarField<float> hinge_area_sums;
        simulation::ScalarField<float> hinge_stiffnesses;
        simulation::ScalarField<std::uint32_t> force_contribution_offsets;
        simulation::ScalarField<std::uint32_t> force_contribution_indices;
        simulation::ScalarField<std::uint32_t> block_contribution_offsets;
        simulation::ScalarField<std::uint32_t> block_contribution_indices;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
    };
} // namespace physica::deformables::cloth::solvers::choi_ko
