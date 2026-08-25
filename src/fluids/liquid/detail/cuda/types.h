#ifndef PHYSICA_FLUIDS_LIQUID_DETAIL_CUDA_TYPES_H
#define PHYSICA_FLUIDS_LIQUID_DETAIL_CUDA_TYPES_H

#include <cstdint>

namespace physica::fluids::liquid::cuda_detail {
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

    template <class Scalar>
    struct ConstVectorView;

    template <class Scalar>
    struct VectorView final {
        Scalar* x;
        Scalar* y;
        Scalar* z;

        operator ConstVectorView<Scalar>() const {
            return {x, y, z};
        }
    };

    template <class Scalar>
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

    struct NeighborhoodView final {
        const std::uint64_t* sorted_keys;
        const std::uint32_t* sorted_particle_indices;
        const std::uint32_t* cell_offsets;
        const std::uint64_t* sorted_boundary_keys;
        const std::uint32_t* sorted_boundary_indices;
        const std::uint32_t* boundary_cell_offsets;
        std::uint32_t particle_count;
        std::uint32_t boundary_count;
        std::uint32_t cells_x;
        std::uint32_t cells_y;
        std::uint32_t cells_z;
        float origin_x;
        float origin_y;
        float origin_z;
        float cell_size;
    };

    struct BoundaryView final {
        const float* position_x;
        const float* position_y;
        const float* position_z;
        const float* velocity_x;
        const float* velocity_y;
        const float* velocity_z;
        const float* volumes;
        std::uint32_t count;
        float time;
    };
} // namespace physica::fluids::liquid::cuda_detail

#endif
