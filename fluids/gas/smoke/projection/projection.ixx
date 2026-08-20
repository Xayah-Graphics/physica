module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

export module physica.fluids.gas.smoke.projection;

import std;
import physica.fluids.gas.smoke.domain;

namespace physica::fluids::gas::smoke::projection_detail {
    void pressure_rhs_forward(const Domain& domain, const StaggeredVectorField& velocity, ScalarField& rhs);
    void pressure_rhs_vjp(const Domain& domain, const ScalarAdjointField& rhs_adjoint, StaggeredVectorAdjointField& velocity_adjoint);
    void project_velocity_forward(const Domain& domain, const StaggeredVectorField& velocity, const ScalarField& pressure, StaggeredVectorField& output);
    void project_velocity_vjp(const Domain& domain, const StaggeredVectorAdjointField& output_adjoint, StaggeredVectorAdjointField& velocity_adjoint, ScalarAdjointField& pressure_adjoint);
} // namespace physica::fluids::gas::smoke::projection_detail

export namespace physica::fluids::gas::smoke {
    struct RedBlackGaussSeidel final {
        struct Configuration final {
            std::uint32_t iterations = 80u;
        };

        const Configuration configuration;

        explicit RedBlackGaussSeidel(Configuration configuration);

        void forward(const Domain& domain, const ScalarBoundary& boundary, const ScalarField& rhs, ScalarField& pressure) const;
        void vjp(const Domain& domain, const ScalarBoundary& boundary, ScalarAdjointField& pressure_adjoint, ScalarAdjointField& rhs_adjoint) const;
    };

    template<class Algorithm>
    concept PressureAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm algorithm, const Domain& domain, const ScalarBoundary& boundary, const ScalarField& rhs, ScalarField& pressure, ScalarAdjointField& pressure_adjoint, ScalarAdjointField& rhs_adjoint) {
        algorithm.forward(domain, boundary, rhs, pressure);
        algorithm.vjp(domain, boundary, pressure_adjoint, rhs_adjoint);
    };

    template<PressureAlgorithm Pressure>
    struct MacProjection final {
        struct Configuration final {
            typename Pressure::Configuration pressure;
        };

        struct Differentiation final {
            ScalarField pressure_tangent;
            ScalarField rhs_tangent;
            ScalarAdjointField pressure_adjoint;
            ScalarAdjointField rhs_adjoint;
        };

        Pressure pressure_algorithm;
        ScalarField pressure;
        ScalarField rhs;
        std::optional<Differentiation> differentiation;

        MacProjection(const Domain& domain, Configuration configuration, const ExecutionMode mode)
            : pressure_algorithm(std::move(configuration.pressure)), pressure(domain.allocate_scalar_field()), rhs(domain.allocate_scalar_field()), differentiation{} {
            if (mode == ExecutionMode::differentiable) {
                differentiation.emplace(Differentiation{
                    .pressure_tangent = domain.allocate_scalar_field(),
                    .rhs_tangent      = domain.allocate_scalar_field(),
                    .pressure_adjoint = domain.allocate_scalar_adjoint_field(),
                    .rhs_adjoint      = domain.allocate_scalar_adjoint_field(),
                });
            }
        }

        void forward(const Domain& domain, const StaggeredVectorField& velocity, StaggeredVectorField& output) {
            domain.clear(pressure);
            projection_detail::pressure_rhs_forward(domain, velocity, rhs);
            pressure_algorithm.forward(domain, domain.configuration.pressure_boundary, rhs, pressure);
            projection_detail::project_velocity_forward(domain, velocity, pressure, output);
        }

        void jvp(const Domain& domain, const StaggeredVectorField& velocity_tangent, StaggeredVectorField& output_tangent) {
            Differentiation& workspace = *differentiation;
            ScalarBoundary boundary = domain.configuration.pressure_boundary;
            boundary.x_min.value = 0.0F;
            boundary.x_max.value = 0.0F;
            boundary.y_min.value = 0.0F;
            boundary.y_max.value = 0.0F;
            boundary.z_min.value = 0.0F;
            boundary.z_max.value = 0.0F;
            domain.clear(workspace.pressure_tangent);
            projection_detail::pressure_rhs_forward(domain, velocity_tangent, workspace.rhs_tangent);
            pressure_algorithm.forward(domain, boundary, workspace.rhs_tangent, workspace.pressure_tangent);
            projection_detail::project_velocity_forward(domain, velocity_tangent, workspace.pressure_tangent, output_tangent);
        }

        void vjp(const Domain& domain, const StaggeredVectorAdjointField& output_adjoint, StaggeredVectorAdjointField& velocity_adjoint) {
            Differentiation& workspace = *differentiation;
            domain.clear(workspace.pressure_adjoint);
            domain.clear(workspace.rhs_adjoint);
            domain.clear(velocity_adjoint);
            projection_detail::project_velocity_vjp(domain, output_adjoint, velocity_adjoint, workspace.pressure_adjoint);
            pressure_algorithm.vjp(domain, domain.configuration.pressure_boundary, workspace.pressure_adjoint, workspace.rhs_adjoint);
            projection_detail::pressure_rhs_vjp(domain, workspace.rhs_adjoint, velocity_adjoint);
        }
    };

    template<class Algorithm>
    concept ProjectionAlgorithm = std::constructible_from<Algorithm, const Domain&, typename Algorithm::Configuration, ExecutionMode> && requires(Algorithm algorithm, const Domain& domain, const StaggeredVectorField& field, StaggeredVectorField& output, const StaggeredVectorAdjointField& adjoint, StaggeredVectorAdjointField& adjoint_output) {
        algorithm.forward(domain, field, output);
        algorithm.jvp(domain, field, output);
        algorithm.vjp(domain, adjoint, adjoint_output);
    };
} // namespace physica::fluids::gas::smoke
