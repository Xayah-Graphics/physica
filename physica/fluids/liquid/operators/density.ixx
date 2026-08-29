module;

#include <physica/fluids/liquid/interop.h>
#include "density-kernels.h"
#include <physica/cuda.h>

export module physica.fluids.liquid.operators.density;

import physica.fluids.liquid.meshfree;
import physica.fluids.liquid.operators.neighborhood;

export namespace physica::fluids::liquid::operators {
    struct CubicSplineDensity final {
        struct Configuration final {};

        explicit CubicSplineDensity(Configuration) {}

        template <class Parameters>
        void forward(const meshfree::Model& model, const VectorField<float>& topology_positions, const VectorField<float>& positions, const Parameters& parameters, const Neighborhood& neighborhood, ScalarField<float>& densities) const {
            kernels::density::cubic_density_forward(model.fields.stream, model.configuration.particle_count, model.configuration.support_radius, field::view(topology_positions), field::view(positions), device::particle_parameters(parameters), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data());
        }

        template <class Parameters, class ParameterTangent>
        void jvp(const meshfree::Model& model, const VectorField<float>& topology_positions, const VectorField<float>& positions, const VectorField<float>& position_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField<float>& density_tangent) const {
            kernels::density::cubic_density_jvp(model.fields.stream, model.configuration.particle_count, model.configuration.support_radius, field::view(topology_positions), field::view(positions), field::view(position_tangent), device::particle_parameters(parameters), device::particle_parameter_tangent(parameter_tangent), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), density_tangent.values.data());
        }

        template <class Parameters, class ParameterAdjoint>
        void vjp(const meshfree::Model& model, const VectorField<float>& topology_positions, const VectorField<float>& positions, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<double>& density_adjoint, VectorField<double>& position_adjoint, ParameterAdjoint& parameter_adjoint) const {
            kernels::density::cubic_density_vjp(model.fields.stream, model.configuration.particle_count, model.configuration.support_radius, field::view(topology_positions), field::view(positions), device::particle_parameters(parameters), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), density_adjoint.values.data(), field::view(position_adjoint), device::particle_parameter_adjoint(parameter_adjoint));
        }
    };

    struct Poly6Density final {
        struct Configuration final {};

        explicit Poly6Density(Configuration) {}

        template <class Parameters>
        void forward(const meshfree::Model& model, const VectorField<float>& topology_positions, const VectorField<float>& positions, const Parameters& parameters, const Neighborhood& neighborhood, ScalarField<float>& densities) const {
            kernels::density::poly6_density_forward(model.fields.stream, model.configuration.particle_count, model.configuration.support_radius, field::view(topology_positions), field::view(positions), device::particle_parameters(parameters), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), densities.values.data());
        }

        template <class Parameters, class ParameterTangent>
        void jvp(const meshfree::Model& model, const VectorField<float>& topology_positions, const VectorField<float>& positions, const VectorField<float>& position_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField<float>& density_tangent) const {
            kernels::density::poly6_density_jvp(model.fields.stream, model.configuration.particle_count, model.configuration.support_radius, field::view(topology_positions), field::view(positions), field::view(position_tangent), device::particle_parameters(parameters), device::particle_parameter_tangent(parameter_tangent), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), density_tangent.values.data());
        }

        template <class Parameters, class ParameterAdjoint>
        void vjp(const meshfree::Model& model, const VectorField<float>& topology_positions, const VectorField<float>& positions, const Parameters& parameters, const Neighborhood& neighborhood, const ScalarField<double>& density_adjoint, VectorField<double>& position_adjoint, ParameterAdjoint& parameter_adjoint) const {
            kernels::density::poly6_density_vjp(model.fields.stream, model.configuration.particle_count, model.configuration.support_radius, field::view(topology_positions), field::view(positions), device::particle_parameters(parameters), device::neighborhood(neighborhood), device::boundary(model.boundary, neighborhood), density_adjoint.values.data(), field::view(position_adjoint), device::particle_parameter_adjoint(parameter_adjoint));
        }
    };

} // namespace physica::fluids::liquid::operators
