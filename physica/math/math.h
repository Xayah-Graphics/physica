#ifndef PHYSICA_MATH_H
#define PHYSICA_MATH_H

#include <array>
#include <cstddef>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <type_traits>

#if defined(__CUDACC__)
#define PHYSICA_HOST_DEVICE __host__ __device__
#else
#define PHYSICA_HOST_DEVICE
#endif

namespace physica {
    template <class Scalar>
    struct Vector2 final {
        Scalar x{};
        Scalar y{};

        [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Scalar& operator[](const std::size_t index) noexcept {
            if (index == 0U) return x;
            return y;
        }

        [[nodiscard]] PHYSICA_HOST_DEVICE constexpr const Scalar& operator[](const std::size_t index) const noexcept {
            if (index == 0U) return x;
            return y;
        }

        constexpr bool operator==(const Vector2&) const noexcept = default;
    };

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector2<Scalar> operator+(const Vector2<Scalar> first, const Vector2<Scalar> second) noexcept {
        return {first.x + second.x, first.y + second.y};
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector2<Scalar> operator-(const Vector2<Scalar> first, const Vector2<Scalar> second) noexcept {
        return {first.x - second.x, first.y - second.y};
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector2<Scalar> operator-(const Vector2<Scalar> value) noexcept {
        return {-value.x, -value.y};
    }

    template <class Scalar, class Factor>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector2<decltype(Scalar{} * Factor{})> operator*(const Vector2<Scalar> value, const Factor factor) noexcept {
        return {value.x * factor, value.y * factor};
    }

    template <class Scalar, class Factor>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector2<decltype(Scalar{} * Factor{})> operator*(const Factor factor, const Vector2<Scalar> value) noexcept {
        return value * factor;
    }

    template <class Scalar, class Divisor>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector2<decltype(Scalar{} / Divisor{})> operator/(const Vector2<Scalar> value, const Divisor divisor) noexcept {
        return {value.x / divisor, value.y / divisor};
    }

    template <class First, class Second>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr auto dot(const Vector2<First> first, const Vector2<Second> second) noexcept {
        return first.x * second.x + first.y * second.y;
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Scalar squared_length(const Vector2<Scalar> value) noexcept {
        return dot(value, value);
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE inline Scalar length(const Vector2<Scalar> value) {
        return static_cast<Scalar>(::cuda::std::sqrt(static_cast<double>(squared_length(value))));
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE inline Vector2<Scalar> normalized(const Vector2<Scalar> value) {
        return value / length(value);
    }

    template <class Scalar>
    struct Vector3 final {
        Scalar x{};
        Scalar y{};
        Scalar z{};

        [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Scalar& operator[](const std::size_t index) noexcept {
            if (index == 0U) return x;
            if (index == 1U) return y;
            return z;
        }

        [[nodiscard]] PHYSICA_HOST_DEVICE constexpr const Scalar& operator[](const std::size_t index) const noexcept {
            if (index == 0U) return x;
            if (index == 1U) return y;
            return z;
        }

        constexpr bool operator==(const Vector3&) const noexcept = default;
    };

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector3<Scalar> operator+(const Vector3<Scalar> first, const Vector3<Scalar> second) noexcept {
        return {first.x + second.x, first.y + second.y, first.z + second.z};
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector3<Scalar> operator-(const Vector3<Scalar> first, const Vector3<Scalar> second) noexcept {
        return {first.x - second.x, first.y - second.y, first.z - second.z};
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector3<Scalar> operator-(const Vector3<Scalar> value) noexcept {
        return {-value.x, -value.y, -value.z};
    }

    template <class Scalar, class Factor>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector3<decltype(Scalar{} * Factor{})> operator*(const Vector3<Scalar> value, const Factor factor) noexcept {
        return {value.x * factor, value.y * factor, value.z * factor};
    }

    template <class Scalar, class Factor>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector3<decltype(Scalar{} * Factor{})> operator*(const Factor factor, const Vector3<Scalar> value) noexcept {
        return value * factor;
    }

    template <class Scalar, class Divisor>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Vector3<decltype(Scalar{} / Divisor{})> operator/(const Vector3<Scalar> value, const Divisor divisor) noexcept {
        return {value.x / divisor, value.y / divisor, value.z / divisor};
    }

    template <class First, class Second>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr auto dot(const Vector3<First> first, const Vector3<Second> second) noexcept {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    template <class First, class Second>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr auto cross(const Vector3<First> first, const Vector3<Second> second) noexcept {
        return Vector3<decltype(first.x * second.x)>{
            first.y * second.z - first.z * second.y,
            first.z * second.x - first.x * second.z,
            first.x * second.y - first.y * second.x,
        };
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr Scalar squared_length(const Vector3<Scalar> value) noexcept {
        return dot(value, value);
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE inline Scalar length(const Vector3<Scalar> value) {
        return static_cast<Scalar>(::cuda::std::sqrt(static_cast<double>(squared_length(value))));
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE inline Vector3<Scalar> normalized(const Vector3<Scalar> value) {
        return value / length(value);
    }

    template <class Scalar>
    struct Matrix4 final {
        std::array<Scalar, 16> values{};

        [[nodiscard]] constexpr Scalar& operator()(const std::size_t row, const std::size_t column) noexcept {
            return values[row * 4U + column];
        }

        [[nodiscard]] constexpr const Scalar& operator()(const std::size_t row, const std::size_t column) const noexcept {
            return values[row * 4U + column];
        }

        constexpr bool operator==(const Matrix4&) const = default;
    };

    template <class Scalar>
    struct AxisAlignedBox3 final {
        Vector3<Scalar> minimum{};
        Vector3<Scalar> maximum{};

        constexpr bool operator==(const AxisAlignedBox3&) const noexcept = default;
    };

    template <class Scalar>
    struct Ray3 final {
        Vector3<Scalar> origin{};
        Vector3<Scalar> direction{};

        constexpr bool operator==(const Ray3&) const noexcept = default;
    };

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE constexpr bool contains(const AxisAlignedBox3<Scalar> box, const Vector3<Scalar> point) noexcept {
        return point.x >= box.minimum.x && point.x <= box.maximum.x && point.y >= box.minimum.y && point.y <= box.maximum.y && point.z >= box.minimum.z && point.z <= box.maximum.z;
    }

    template <class Scalar>
    [[nodiscard]] PHYSICA_HOST_DEVICE inline bool intersect(const AxisAlignedBox3<Scalar> box, const Ray3<Scalar> ray, Scalar& minimum_distance) {
        const Vector3<Scalar> inverse_direction{static_cast<Scalar>(1) / ray.direction.x, static_cast<Scalar>(1) / ray.direction.y, static_cast<Scalar>(1) / ray.direction.z};
        const Vector3<Scalar> first{
            (box.minimum.x - ray.origin.x) * inverse_direction.x,
            (box.minimum.y - ray.origin.y) * inverse_direction.y,
            (box.minimum.z - ray.origin.z) * inverse_direction.z,
        };
        const Vector3<Scalar> second{
            (box.maximum.x - ray.origin.x) * inverse_direction.x,
            (box.maximum.y - ray.origin.y) * inverse_direction.y,
            (box.maximum.z - ray.origin.z) * inverse_direction.z,
        };
        const Scalar x_min = ::cuda::std::min(first.x, second.x);
        const Scalar x_max = ::cuda::std::max(first.x, second.x);
        const Scalar y_min = ::cuda::std::min(first.y, second.y);
        const Scalar y_max = ::cuda::std::max(first.y, second.y);
        const Scalar z_min = ::cuda::std::min(first.z, second.z);
        const Scalar z_max = ::cuda::std::max(first.z, second.z);
        const Scalar entry = ::cuda::std::max(::cuda::std::max(x_min, y_min), z_min);
        const Scalar exit  = ::cuda::std::min(::cuda::std::min(x_max, y_max), z_max);
        minimum_distance   = ::cuda::std::max(entry, static_cast<Scalar>(0));
        return exit >= minimum_distance;
    }

    static_assert(sizeof(Vector2<float>) == 2U * sizeof(float));
    static_assert(sizeof(Vector3<float>) == 3U * sizeof(float));
    static_assert(sizeof(Matrix4<float>) == 16U * sizeof(float));
    static_assert(std::is_trivially_copyable_v<Vector2<float>>);
    static_assert(std::is_trivially_copyable_v<Vector3<float>>);
    static_assert(std::is_trivially_copyable_v<Matrix4<float>>);
} // namespace physica

#undef PHYSICA_HOST_DEVICE

#endif
