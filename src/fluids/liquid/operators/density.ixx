module;

#include "../detail/cuda/interop.h"
#include "density-kernels.h"
#include <physica/cuda.h>

export module physica.fluids.liquid.operators.density;

import physica.fluids.liquid.domain;
import physica.fluids.liquid.operators.neighborhood;

export namespace physica::fluids::liquid::operators {
    struct CubicSplineDensity final {
        struct Configuration final {};

        explicit CubicSplineDensity(Configuration) {}

        template <class Parameters>
        void forward(const Domain& domain, const VectorField<float>& topology_positions, const VectorField<float>& positions, const Parameters& parameters, const Neighborhood& neighborhood, ScalarField<float>& densities) const {
            cuda_detail::cubic_density_forward(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data());
        }

        template <class Parameters, class ParameterTangent>
        void jvp(const Domain& domain, const VectorField<float>& topology_positions, const VectorField<float>& positions, const VectorField<float>& position_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField<float>& density_tangent) const {
            cuda_detail::cubic_density_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::vector(position_tangent), cuda_detail::parameters(parameters), cuda_detail::parameter_tangent(parameter_tangent), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), density_tangent.values.data());
        }

        template <class Parameters, class ParameterAdjoint>
        void vjp(const Domain& domain, const VectorField<float>& topology_positions, const VectorField<float>& positions, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<double>& density_adjoint, VectorField<double>& position_adjoint, ParameterAdjoint& parameter_adjoint) const {
            cuda_detail::cubic_density_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), density_adjoint.values.data(), cuda_detail::vector(position_adjoint), cuda_detail::parameter_adjoint(parameter_adjoint));
        }
    };

    struct Poly6Density final {
        struct Configuration final {};

        explicit Poly6Density(Configuration) {}

        template <class Parameters>
        void forward(const Domain& domain, const VectorField<float>& topology_positions, const VectorField<float>& positions, const Parameters& parameters, const Neighborhood& neighborhood, ScalarField<float>& densities) const {
            cuda_detail::poly6_density_forward(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data());
        }

        template <class Parameters, class ParameterTangent>
        void jvp(const Domain& domain, const VectorField<float>& topology_positions, const VectorField<float>& positions, const VectorField<float>& position_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField<float>& density_tangent) const {
            cuda_detail::poly6_density_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::vector(position_tangent), cuda_detail::parameters(parameters), cuda_detail::parameter_tangent(parameter_tangent), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), density_tangent.values.data());
        }

        template <class Parameters, class ParameterAdjoint>
        void vjp(const Domain& domain, const VectorField<float>& topology_positions, const VectorField<float>& positions, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<double>& density_adjoint, VectorField<double>& position_adjoint, ParameterAdjoint& parameter_adjoint) const {
            cuda_detail::poly6_density_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), density_adjoint.values.data(), cuda_detail::vector(position_adjoint), cuda_detail::parameter_adjoint(parameter_adjoint));
        }
    };

} // namespace physica::fluids::liquid::operators
