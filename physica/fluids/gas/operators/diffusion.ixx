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

        void forward(const Domain& domain, const VectorField<float>& source, VectorField<float>& output, Workspace& workspace) const;
        void jvp(const Domain& domain, const VectorField<float>& source_tangent, VectorField<float>& output_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& source_adjoint, AdjointWorkspace& workspace) const;
    };

    struct ImplicitVelocityDiffusion final {
        struct Configuration final {
            std::uint32_t iterations{20u};
            float viscosity{};
        };

        struct Workspace final {
            VectorField<float> first;
            VectorField<float> second;
        };

        struct TangentWorkspace final {
            VectorField<float> first;
            VectorField<float> second;
        };

        struct AdjointWorkspace final {
            VectorField<double> first;
            VectorField<double> second;
        };

        explicit ImplicitVelocityDiffusion(Configuration configuration);

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const VectorField<float>& source, VectorField<float>& output, Workspace& workspace) const;
        void jvp(const Domain& domain, const VectorField<float>& source_tangent, VectorField<float>& output_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const VectorField<double>& output_adjoint, VectorField<double>& source_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };

    template <class Algorithm>
    concept DiffusionAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace, const VectorField<float>& velocity, VectorField<float>& velocity_output, const VectorField<double>& velocity_adjoint, VectorField<double>& velocity_adjoint_output) {
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, velocity, velocity_output, workspace);
        algorithm.jvp(domain, velocity, velocity_output, tangent_workspace);
        algorithm.vjp(domain, velocity_adjoint, velocity_adjoint_output, adjoint_workspace);
    };
} // namespace physica::fluids::gas::operators
