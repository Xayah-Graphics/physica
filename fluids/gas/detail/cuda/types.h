#ifndef PHYSICA_FLUIDS_GAS_DETAIL_CUDA_TYPES_H
#define PHYSICA_FLUIDS_GAS_DETAIL_CUDA_TYPES_H

#include <cstdint>
#include <type_traits>

namespace physica::fluids::gas::detail::cuda {
    struct Vector final {
        float x;
        float y;
        float z;
    };

    struct Grid final {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        std::uint32_t dimensions;
        float cell_size;
        float time_step;
    };

    struct ScalarBoundaryData final {
        std::uint32_t modes[6];
        float values[6];
    };

    struct VelocityBoundaryData final {
        std::uint32_t modes[6];
        float values[18];
    };

    template <class Value>
    struct ScalarView final {
        Value* values;

        constexpr ScalarView() = default;
        constexpr explicit ScalarView(Value* next_values) : values(next_values) {}

        template <class Other>
            requires std::is_convertible_v<Other*, Value*>
        constexpr ScalarView(const ScalarView<Other> other) : values(other.values) {}
    };

    template <class Value>
    struct CenteredVectorView final {
        Value* x;
        Value* y;
        Value* z;

        constexpr CenteredVectorView() = default;
        constexpr CenteredVectorView(Value* next_x, Value* next_y, Value* next_z) : x(next_x), y(next_y), z(next_z) {}

        template <class Other>
            requires std::is_convertible_v<Other*, Value*>
        constexpr CenteredVectorView(const CenteredVectorView<Other> other) : x(other.x), y(other.y), z(other.z) {}
    };

    template <class Value>
    struct StaggeredVectorView final {
        Value* x;
        Value* y;
        Value* z;

        constexpr StaggeredVectorView() = default;
        constexpr StaggeredVectorView(Value* next_x, Value* next_y, Value* next_z) : x(next_x), y(next_y), z(next_z) {}

        template <class Other>
            requires std::is_convertible_v<Other*, Value*>
        constexpr StaggeredVectorView(const StaggeredVectorView<Other> other) : x(other.x), y(other.y), z(other.z) {}
    };
} // namespace physica::fluids::gas::detail::cuda

#endif
