#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_DOMAIN_DEVICE_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_DOMAIN_DEVICE_H

#include <cstdint>

namespace physica::fluids::liquid::particle::cuda_detail {
    struct Float3 final {
        float x;
        float y;
        float z;
    };

    struct Double3 final {
        double x;
        double y;
        double z;
    };

    template<class Scalar>
    struct ConstVectorView;

    template<class Scalar>
    struct VectorView final {
        Scalar* x;
        Scalar* y;
        Scalar* z;

        operator ConstVectorView<Scalar>() const {
            return {x, y, z};
        }
    };

    template<class Scalar>
    struct ConstVectorView final {
        const Scalar* x;
        const Scalar* y;
        const Scalar* z;
    };

    struct Box final {
        float minimum_x;
        float minimum_y;
        float minimum_z;
        float maximum_x;
        float maximum_y;
        float maximum_z;
        float velocity_x;
        float velocity_y;
        float velocity_z;
        std::uint32_t no_slip;
    };

    struct ParticleParameterView final {
        const float* masses;
        const float* rest_densities;
        const float* viscosities;
        const float* surface_tensions;
    };

    struct ParticleParameterTangentView final {
        const float* masses;
        const float* rest_densities;
        const float* viscosities;
        const float* surface_tensions;
    };

    struct ParticleParameterAdjointView final {
        double* masses;
        double* rest_densities;
        double* viscosities;
        double* surface_tensions;
    };
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
