module;

#include "../domain/interop.h"
#include "../neighborhood/interop.h"
#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

module physica.fluids.liquid.particle.density;

namespace physica::fluids::liquid::particle::density {
    void sph_forward(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities) {
        cuda_detail::density_forward(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, false, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data());
    }

    void sph_jvp(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent) {
        cuda_detail::density_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, false, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::vector(position_tangent), cuda_detail::parameters(parameters), cuda_detail::parameter_tangent(parameter_tangent), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), density_tangent.values.data());
    }

    void sph_vjp(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
        cuda_detail::density_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, false, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), density_adjoint.values.data(), cuda_detail::adjoint_vector(position_adjoint), cuda_detail::parameter_adjoint(parameter_adjoint));
    }

    void pbf_forward(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities) {
        cuda_detail::density_forward(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, true, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), densities.values.data());
    }

    void pbf_jvp(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent) {
        cuda_detail::density_jvp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, true, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::vector(position_tangent), cuda_detail::parameters(parameters), cuda_detail::parameter_tangent(parameter_tangent), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), density_tangent.values.data());
    }

    void pbf_vjp(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
        cuda_detail::density_vjp(domain.stream, domain.configuration.particle_count, domain.configuration.support_radius, true, cuda_detail::vector(topology_positions), cuda_detail::vector(positions), cuda_detail::parameters(parameters), cuda_detail::neighborhood(neighborhood), cuda_detail::boundary(domain.boundary, neighborhood), density_adjoint.values.data(), cuda_detail::adjoint_vector(position_adjoint), cuda_detail::parameter_adjoint(parameter_adjoint));
    }
} // namespace physica::fluids::liquid::particle::density
