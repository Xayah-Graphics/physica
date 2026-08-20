export module physica.fluids.liquid.particle.density;

import physica.fluids.liquid.particle.domain;
import physica.fluids.liquid.particle.neighborhood;

export namespace physica::fluids::liquid::particle::density {
    void sph_forward(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities);
    void sph_jvp(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent);
    void sph_vjp(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint);
    void pbf_forward(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities);
    void pbf_jvp(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent);
    void pbf_vjp(const Domain& domain, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint);
} // namespace physica::fluids::liquid::particle::density
