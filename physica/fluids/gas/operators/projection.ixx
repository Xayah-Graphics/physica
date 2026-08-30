module;

#include <physica/cuda.h>

export module physica.fluids.gas.operators.projection;

import std;
import physica.fluids.gas.domain;

namespace physica::fluids::gas::operators::projection {
    void pressure_rhs_forward(const Domain& domain, std::uint32_t pressure_anchor, const simulation::VectorField<float>& velocity, simulation::ScalarField<float>& rhs);
    void pressure_rhs_vjp(const Domain& domain, std::uint32_t pressure_anchor, const simulation::ScalarField<double>& rhs_adjoint, simulation::VectorField<double>& velocity_adjoint);
    void project_velocity_forward(const Domain& domain, const simulation::VectorField<float>& velocity, const simulation::ScalarField<float>& pressure, simulation::VectorField<float>& output);
    void project_velocity_vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::ScalarField<double>& pressure_adjoint);
} // namespace physica::fluids::gas::operators::projection

export namespace physica::fluids::gas::operators {
    enum class PressureGauge : std::uint32_t {
        first_fluid_cell,
        none,
    };

    struct RedBlackGaussSeidel final {
        struct Configuration final {
            std::uint32_t iterations{80u};
        };

        struct Workspace final {};
        struct AdjointWorkspace final {};

        explicit RedBlackGaussSeidel(Configuration configuration);

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;
        void forward(const Domain& domain, const ScalarBoundary& boundary, std::uint32_t pressure_anchor, const simulation::ScalarField<float>& rhs, simulation::ScalarField<float>& pressure, Workspace& workspace) const;
        void vjp(const Domain& domain, const ScalarBoundary& boundary, std::uint32_t pressure_anchor, simulation::ScalarField<double>& pressure_adjoint, simulation::ScalarField<double>& rhs_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };

    template <class Algorithm>
    concept PressureAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, const ScalarBoundary& boundary, typename Algorithm::Workspace& workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace, const simulation::ScalarField<float>& rhs, simulation::ScalarField<float>& pressure, simulation::ScalarField<double>& pressure_adjoint, simulation::ScalarField<double>& rhs_adjoint) {
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, boundary, std::uint32_t{}, rhs, pressure, workspace);
        algorithm.vjp(domain, boundary, std::uint32_t{}, pressure_adjoint, rhs_adjoint, adjoint_workspace);
    };

    template <PressureAlgorithm Pressure>
    struct MacProjection final {
        struct Configuration final {
            ScalarBoundary boundary{};
            PressureGauge gauge{PressureGauge::first_fluid_cell};
            typename Pressure::Configuration pressure{};
        };

        struct Workspace final {
            simulation::ScalarField<float> rhs;
            simulation::ScalarField<float> pressure;
            typename Pressure::Workspace pressure_solver;
        };

        struct TangentWorkspace final {
            simulation::ScalarField<float> rhs;
            simulation::ScalarField<float> pressure;
            typename Pressure::Workspace pressure_solver;
        };

        struct AdjointWorkspace final {
            simulation::ScalarField<double> rhs;
            simulation::ScalarField<double> pressure;
            typename Pressure::AdjointWorkspace pressure_solver;
        };

        MacProjection(const Domain& domain, Configuration configuration) : pressure(std::move(configuration.pressure)), boundary(std::move(configuration.boundary)), pressure_anchor(configuration.gauge == PressureGauge::none || std::ranges::any_of(std::array{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max}, [](const ScalarBoundaryFace& face) { return face.mode == ScalarBoundaryMode::fixed_value; }) ? static_cast<std::uint32_t>(domain.grid.cell_count) : domain.first_fluid_cell) {}

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const {
            return {.rhs = domain.grid.allocate_cell_field<float>(), .pressure = domain.grid.allocate_cell_field<float>(), .pressure_solver = pressure.allocate_workspace(domain)};
        }

        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const {
            return {.rhs = domain.grid.allocate_cell_field<float>(), .pressure = domain.grid.allocate_cell_field<float>(), .pressure_solver = pressure.allocate_workspace(domain)};
        }

        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const {
            return {.rhs = domain.grid.allocate_cell_field<double>(), .pressure = domain.grid.allocate_cell_field<double>(), .pressure_solver = pressure.allocate_adjoint_workspace(domain)};
        }

        void forward(const Domain& domain, const simulation::VectorField<float>& velocity, simulation::VectorField<float>& output, Workspace& workspace) const {
            domain.grid.clear(workspace.pressure);
            projection::pressure_rhs_forward(domain, pressure_anchor, velocity, workspace.rhs);
            pressure.forward(domain, boundary, pressure_anchor, workspace.rhs, workspace.pressure, workspace.pressure_solver);
            projection::project_velocity_forward(domain, velocity, workspace.pressure, output);
        }

        void jvp(const Domain& domain, const simulation::VectorField<float>& velocity_tangent, simulation::VectorField<float>& output_tangent, TangentWorkspace& workspace) const {
            domain.grid.clear(workspace.pressure);
            projection::pressure_rhs_forward(domain, pressure_anchor, velocity_tangent, workspace.rhs);
            pressure.forward(domain, homogeneous(boundary), pressure_anchor, workspace.rhs, workspace.pressure, workspace.pressure_solver);
            projection::project_velocity_forward(domain, velocity_tangent, workspace.pressure, output_tangent);
        }

        void vjp(const Domain& domain, const simulation::VectorField<double>& output_adjoint, simulation::VectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const {
            domain.grid.clear(workspace.pressure);
            domain.grid.clear(workspace.rhs);
            projection::project_velocity_vjp(domain, output_adjoint, velocity_adjoint, workspace.pressure);
            pressure.vjp(domain, boundary, pressure_anchor, workspace.pressure, workspace.rhs, workspace.pressure_solver);
            projection::pressure_rhs_vjp(domain, pressure_anchor, workspace.rhs, velocity_adjoint);
        }

    private:
        Pressure pressure;
        const ScalarBoundary boundary;
        const std::uint32_t pressure_anchor;
    };

    template <class Algorithm>
    concept ProjectionAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Domain& domain, typename Algorithm::Workspace& workspace, typename Algorithm::TangentWorkspace& tangent_workspace, typename Algorithm::AdjointWorkspace& adjoint_workspace, const simulation::VectorField<float>& velocity, simulation::VectorField<float>& velocity_output, const simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& velocity_adjoint_output) {
        { algorithm.allocate_workspace(domain) } -> std::same_as<typename Algorithm::Workspace>;
        { algorithm.allocate_tangent_workspace(domain) } -> std::same_as<typename Algorithm::TangentWorkspace>;
        { algorithm.allocate_adjoint_workspace(domain) } -> std::same_as<typename Algorithm::AdjointWorkspace>;
        algorithm.forward(domain, velocity, velocity_output, workspace);
        algorithm.jvp(domain, velocity, velocity_output, tangent_workspace);
        algorithm.vjp(domain, velocity_adjoint, velocity_adjoint_output, adjoint_workspace);
    };
} // namespace physica::fluids::gas::operators
