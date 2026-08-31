module;

#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.discrete_shells;

import std;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.solvers.block_pcg;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::discrete_shells {
    struct State final {
        simulation::VectorField<float> positions;
        simulation::VectorField<float> velocities;
        simulation::VectorField<float> accelerations;

        State(::cuda::stream_ref stream, std::size_t particle_count);

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
            std::uint32_t newton_iteration_count;
            std::uint32_t pcg_iteration_count;
            std::uint32_t line_search_candidate_count;
            Vector3<float> gravity;
            float length_stiffness;
            float area_stiffness;
            float bending_stiffness;
            float bending_damping;
            float hessian_positive_margin;
            float armijo_coefficient;
            float line_search_contraction;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Parameters final {
            simulation::ScalarField<float> masses;
        };

        struct StepCache final {
            simulation::VectorField<float> position_predictor;
            simulation::VectorField<float> velocity_predictor;

            simulation::ScalarField<float> edge_length_conditions;
            simulation::ScalarField<float> edge_energies;
            simulation::VectorField<float> edge_energy_gradients;
            simulation::ScalarField<float> edge_energy_hessians;

            simulation::ScalarField<float> triangle_area_conditions;
            simulation::ScalarField<float> triangle_energies;
            simulation::VectorField<float> triangle_energy_gradients;
            simulation::ScalarField<float> triangle_energy_hessians;

            simulation::ScalarField<float> previous_hinge_angles;
            simulation::ScalarField<float> hinge_angles;
            simulation::ScalarField<float> hinge_angle_deltas;
            simulation::ScalarField<float> hinge_angle_rates;
            simulation::ScalarField<float> hinge_energies;
            simulation::ScalarField<float> hinge_damping_potentials;
            simulation::VectorField<float> hinge_angle_gradients;
            simulation::ScalarField<float> hinge_angle_hessians;
            simulation::VectorField<float> hinge_energy_gradients;
            simulation::ScalarField<float> hinge_energy_hessians;
            simulation::VectorField<float> hinge_damping_residuals;
            simulation::ScalarField<float> hinge_damping_jacobians;

            simulation::VectorField<float> energy_gradient;
            simulation::VectorField<float> damping_residual;
            simulation::VectorField<float> residual;
            block_pcg::BlockCsrMatrix energy_hessian;
            block_pcg::BlockCsrMatrix damping_jacobian;
            block_pcg::BlockCsrMatrix unregularized_system;
            simulation::ScalarField<double> minimum_gershgorin_bound;
            simulation::ScalarField<float> regularization_shift;

            simulation::ScalarField<double> incremental_potential;
            simulation::ScalarField<double> directional_derivative;
            simulation::ScalarField<double> line_search_potentials;
            simulation::ScalarField<float> accepted_step_size;
            simulation::ScalarField<std::uint32_t> accepted_candidate;
        };

        struct Workspace final {
            block_pcg::BlockCsrMatrix system;
            simulation::VectorField<float> right_hand_side;
            simulation::VectorField<float> newton_direction;
            simulation::ScalarField<float> gershgorin_lower_bounds;
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
            std::vector<std::uint32_t> energy_hessian_contribution_offsets;
            std::vector<std::uint32_t> energy_hessian_contributions;
            std::vector<std::uint32_t> damping_jacobian_contribution_offsets;
            std::vector<std::uint32_t> damping_jacobian_contributions;
        };

        struct HostData final {
            Pattern pattern;
            std::vector<float> edge_rest_lengths;
            std::vector<float> triangle_rest_areas;
            std::vector<float> hinge_rest_angles;
            std::vector<float> hinge_weights;
            std::vector<std::uint32_t> fixed_vertex_mask;
            std::vector<Vector3<float>> fixed_positions;
            std::vector<float> line_search_steps;
        };

        [[nodiscard]] static HostData build_host_data(const Model<float>& model, const Configuration& configuration);
        [[nodiscard]] static std::uint32_t find_block(const Pattern& pattern, std::uint32_t row, std::uint32_t column);
        [[nodiscard]] static float signed_dihedral(std::span<const Vector3<float>> positions, const Hinge& hinge);
        Solver(const Model<float>& model, const Configuration& configuration, HostData host_data);
        void evaluate_system(const Model<float>& model, const Control<float>& control, const Parameters& parameters, const simulation::VectorField<float>& positions, StepCache& cache, Workspace& workspace) const;
        void evaluate_potential(const Model<float>& model, const Control<float>& control, const Parameters& parameters, const simulation::VectorField<float>& positions, StepCache& cache) const;

        const float time_step;
        const std::uint32_t newton_iteration_count;
        const std::uint32_t line_search_candidate_count;
        const Vector3<float> gravity;
        const float length_stiffness;
        const float area_stiffness;
        const float bending_stiffness;
        const float bending_damping;
        const float hessian_positive_margin;
        const float armijo_coefficient;
        const Pattern pattern;
        block_pcg::Solver block_solver;
        simulation::ScalarField<float> edge_rest_lengths;
        simulation::ScalarField<float> triangle_rest_areas;
        simulation::ScalarField<float> hinge_rest_angles;
        simulation::ScalarField<float> hinge_weights;
        simulation::ScalarField<std::uint32_t> energy_hessian_contribution_offsets;
        simulation::ScalarField<std::uint32_t> energy_hessian_contributions;
        simulation::ScalarField<std::uint32_t> damping_jacobian_contribution_offsets;
        simulation::ScalarField<std::uint32_t> damping_jacobian_contributions;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
        simulation::ScalarField<float> line_search_steps;
    };
} // namespace physica::deformables::cloth::solvers::discrete_shells
