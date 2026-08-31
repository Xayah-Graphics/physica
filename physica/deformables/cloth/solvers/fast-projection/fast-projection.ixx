module;

#include <cublas_v2.h>
#include <physica/cuda.h>

export module physica.deformables.cloth.solvers.fast_projection;

import std;
import physica.deformables.cloth.model;
export import physica.deformables.cloth.state;

export namespace physica::deformables::cloth::solvers::fast_projection {
    struct Solver final {
        struct FixedVertex final {
            std::uint32_t particle;
            Vector3<float> position;
        };

        struct Configuration final {
            float time_step;
            std::uint32_t outer_iteration_count;
            std::uint32_t pcg_iteration_count;
            Vector3<float> gravity;
            std::vector<FixedVertex> fixed_vertices;
        };

        struct Parameters final {
            simulation::ScalarField<float> masses;
        };

        struct StepCache final {
            simulation::ScalarField<float> constraint_values;
            simulation::VectorField<float> jacobian_directions;
            simulation::ScalarField<float> jacobi_inverse_diagonal;
            simulation::ScalarField<float> lambdas;
        };

        struct Workspace final {
            simulation::ScalarField<float> residual;
            simulation::ScalarField<float> preconditioned_residual;
            simulation::ScalarField<float> search_direction;
            simulation::ScalarField<float> matrix_product;
            simulation::VectorField<float> vertex_product;
            simulation::ScalarField<float> rho;
            simulation::ScalarField<float> next_rho;
            simulation::ScalarField<float> matrix_denominator;
            simulation::ScalarField<float> alpha;
            simulation::ScalarField<float> beta;
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
        struct CublasDeleter final {
            void operator()(cublasContext* handle) const noexcept;
        };

        const float time_step;
        const std::uint32_t outer_iteration_count;
        const std::uint32_t pcg_iteration_count;
        const Vector3<float> gravity;
        std::unique_ptr<cublasContext, CublasDeleter> cublas;
        simulation::ScalarField<float> rest_lengths;
        simulation::ScalarField<std::uint32_t> fixed_vertex_mask;
        simulation::VectorField<float> fixed_positions;
    };
} // namespace physica::deformables::cloth::solvers::fast_projection
