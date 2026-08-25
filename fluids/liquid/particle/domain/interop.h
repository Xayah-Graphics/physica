#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_DOMAIN_INTEROP_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_DOMAIN_INTEROP_H

#include "device.h"

namespace physica::fluids::liquid::particle::cuda_detail {
    template <class Field>
    ConstVectorView<float> vector(const Field& field) {
        return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
    }

    template <class Field>
    VectorView<float> vector(Field& field) {
        return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
    }

    template <class Field>
    ConstVectorView<double> adjoint_vector(const Field& field) {
        return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
    }

    template <class Field>
    VectorView<double> adjoint_vector(Field& field) {
        return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
    }

    template <class Parameters>
    ParticleParameterView parameters(const Parameters& value) {
        return {.masses = value.masses.data(), .rest_densities = value.rest_densities.data(), .viscosities = value.viscosities.data(), .surface_tensions = value.surface_tensions.data()};
    }

    template <class Parameters>
    ParticleParameterTangentView parameter_tangent(const Parameters& value) {
        return {.masses = value.masses.data(), .rest_densities = value.rest_densities.data(), .viscosities = value.viscosities.data(), .surface_tensions = value.surface_tensions.data()};
    }

    template <class Parameters>
    ParticleParameterAdjointView parameter_adjoint(Parameters& value) {
        return {.masses = value.masses.data(), .rest_densities = value.rest_densities.data(), .viscosities = value.viscosities.data(), .surface_tensions = value.surface_tensions.data()};
    }
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
