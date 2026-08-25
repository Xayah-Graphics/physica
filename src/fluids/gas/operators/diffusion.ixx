module;

#include <physica/cuda.h>

export module physica.fluids.gas.operators.diffusion;

import std;
import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    struct IdentityVelocityDiffusion final {
        struct Configuration final {};
        struct Workspace final {};
        struct TangentWorkspace final {};
        struct AdjointWorkspace final {};

        explicit IdentityVelocityDiffusion(Configuration configuration);

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const StaggeredVectorField<float>& source, StaggeredVectorField<float>& output, Workspace& workspace) const;
        void jvp(const Domain& domain, const StaggeredVectorField<float>& source_tangent, StaggeredVectorField<float>& output_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& source_adjoint, AdjointWorkspace& workspace) const;
    };

    struct ImplicitVelocityDiffusion final {
        struct Configuration final {
            std::uint32_t iterations{20u};
            float viscosity{};
        };

        struct Workspace final {
            StaggeredVectorField<float> first;
            StaggeredVectorField<float> second;
        };

        struct TangentWorkspace final {
            StaggeredVectorField<float> first;
            StaggeredVectorField<float> second;
        };

        struct AdjointWorkspace final {
            StaggeredVectorField<double> first;
            StaggeredVectorField<double> second;
        };

        explicit ImplicitVelocityDiffusion(Configuration configuration);

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const StaggeredVectorField<float>& source, StaggeredVectorField<float>& output, Workspace& workspace) const;
        void jvp(const Domain& domain, const StaggeredVectorField<float>& source_tangent, StaggeredVectorField<float>& output_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const StaggeredVectorField<double>& output_adjoint, StaggeredVectorField<double>& source_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };

    template <class Algorithm>
    concept DiffusionAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace, const StaggeredVectorField<float>& velocity, StaggeredVectorField<float>& velocity_output, const StaggeredVectorField<double>& velocity_adjoint, StaggeredVectorField<double>& velocity_adjoint_output) {
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, velocity, velocity_output, workspace);
        algorithm.jvp(domain, velocity, velocity_output, tangent_workspace);
        algorithm.vjp(domain, velocity_adjoint, velocity_adjoint_output, adjoint_workspace);
    };
} // namespace physica::fluids::gas::operators
