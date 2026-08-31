module;

#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.corotated_fem;

import std;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.solvers.block_pcg;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::corotated_fem {
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
            float young_modulus;
            float poisson_ratio;
            float thickness;
            float hessian_positive_margin;
            float armijo_coefficient;
            float line_search_contraction;
            float rank_safety_fraction;
            std::vector<float> masses;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Parameters final {};

        struct StepCache final {
            simulation::VectorField<float> predicted_positions;
            simulation::VectorField<float> deformation_gradient_first_columns;
            simulation::VectorField<float> deformation_gradient_second_columns;
            simulation::VectorField<float> biot_strains;
            simulation::ScalarField<float> triangle_energies;
            simulation::VectorField<float> triangle_gradients;
            simulation::ScalarField<float> triangle_hessians;
            simulation::VectorField<float> gradient;
            block_pcg::BlockCsrMatrix hessian;
            simulation::ScalarField<float> triangle_rank_step_limits;
            simulation::ScalarField<float> maximum_rank_safe_step;
            simulation::ScalarField<float> regularization_shift;
            simulation::ScalarField<double> minimum_gershgorin_bound;
            simulation::ScalarField<float> accepted_step_size;
            simulation::ScalarField<std::uint32_t> accepted_candidate;
            simulation::ScalarField<double> incremental_potential;
            simulation::ScalarField<double> directional_derivative;
            simulation::ScalarField<double> line_search_potentials;
        };

        struct Workspace final {
            block_pcg::BlockCsrMatrix regularized_hessian;
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
            std::vector<std::uint32_t> block_contribution_offsets;
            std::vector<std::uint32_t> block_contributions;
        };

        struct HostData final {
            Pattern pattern;
            std::vector<Vector3<float>> material_u_gradients;
            std::vector<Vector3<float>> material_v_gradients;
            std::vector<float> triangle_weights;
            std::vector<std::uint32_t> fixed_vertex_mask;
            std::vector<Vector3<float>> fixed_positions;
            std::vector<float> line_search_steps;
        };

        [[nodiscard]] static HostData build_host_data(const Model<float>& model, const Configuration& configuration);
        Solver(const Model<float>& model, const Configuration& configuration, HostData host_data);
        void evaluate_system(const Model<float>& model, const Parameters& parameters, const simulation::VectorField<float>& positions, StepCache& cache, Workspace& workspace) const;

        const float time_step;
        const std::uint32_t newton_iteration_count;
        const std::uint32_t line_search_candidate_count;
        const Vector3<float> gravity;
        const float lame_lambda;
        const float lame_mu;
        const float hessian_positive_margin;
        const float armijo_coefficient;
        const float rank_safety_fraction;
        const Pattern pattern;
        block_pcg::Solver block_solver;
        simulation::ScalarField<float> masses;
        simulation::VectorField<float> material_u_gradients;
        simulation::VectorField<float> material_v_gradients;
        simulation::ScalarField<float> triangle_weights;
        simulation::ScalarField<std::uint32_t> block_contribution_offsets;
        simulation::ScalarField<std::uint32_t> block_contributions;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
        simulation::ScalarField<float> line_search_steps;
    };
} // namespace physica::deformables::cloth::solvers::corotated_fem
