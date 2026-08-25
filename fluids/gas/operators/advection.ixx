module;

#include <physica/cuda.h>

export module physica.fluids.gas.operators.advection;

import std;
import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    struct SemiLagrangianRK2 final {
        struct Configuration final {};

        struct Workspace final {
            StaggeredVectorField<float> raw_velocity;
        };

        struct TangentWorkspace final {
            StaggeredVectorField<float> raw_velocity;
        };

        struct AdjointWorkspace final {
            StaggeredVectorField<double> raw_velocity;
        };

        explicit SemiLagrangianRK2(Configuration configuration);

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void velocity_forward(const Domain& domain, const StaggeredVectorField<float>& velocity, StaggeredVectorField<float>& output, Workspace& workspace) const;
        void velocity_jvp(const Domain& domain, const StaggeredVectorField<float>& velocity, const StaggeredVectorField<float>& velocity_tangent, StaggeredVectorField<float>& output_tangent, TangentWorkspace& workspace) const;
        void velocity_vjp(const Domain& domain, const StaggeredVectorField<float>& velocity, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const;
        void scalar_forward(const Domain& domain, const CellField<float>& source, const StaggeredVectorField<float>& velocity, const ScalarBoundary& boundary, const CellField<float>& collider_value, CellField<float>& output) const;
        void scalar_jvp(const Domain& domain, const CellField<float>& source, const CellField<float>& source_tangent, const StaggeredVectorField<float>& velocity, const StaggeredVectorField<float>& velocity_tangent, const ScalarBoundary& boundary, CellField<float>& output_tangent) const;
        void scalar_vjp(const Domain& domain, const CellField<float>& source, const StaggeredVectorField<float>& velocity, const ScalarBoundary& boundary, const CellField<double>& output_adjoint, CellField<double>& source_adjoint, StaggeredVectorField<double>& velocity_adjoint) const;
    };

    template <class Algorithm>
    concept AdvectionAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm algorithm, const Domain& domain, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace, const CellField<float>& scalar, CellField<float>& scalar_output, const CellField<double>& scalar_adjoint, CellField<double>& scalar_adjoint_output, const StaggeredVectorField<float>& velocity, StaggeredVectorField<float>& velocity_output, const StaggeredVectorField<double>& velocity_adjoint, StaggeredVectorField<double>& velocity_adjoint_output, const ScalarBoundary& boundary) {
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.velocity_forward(domain, velocity, velocity_output, workspace);
        algorithm.velocity_jvp(domain, velocity, velocity, velocity_output, tangent_workspace);
        algorithm.velocity_vjp(domain, velocity, velocity_adjoint, velocity_adjoint_output, adjoint_workspace);
        algorithm.scalar_forward(domain, scalar, velocity, boundary, scalar, scalar_output);
        algorithm.scalar_jvp(domain, scalar, scalar, velocity, velocity, boundary, scalar_output);
        algorithm.scalar_vjp(domain, scalar, velocity, boundary, scalar_adjoint, scalar_adjoint_output, velocity_adjoint_output);
    };
} // namespace physica::fluids::gas::operators
