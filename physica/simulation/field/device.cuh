#ifndef PHYSICA_SIMULATION_FIELD_DEVICE_CUH
#define PHYSICA_SIMULATION_FIELD_DEVICE_CUH

#include <cuda_runtime.h>
#include <math/math.h>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace physica::simulation {
    template <class Value>
    struct ScalarView final {
        Value* values{};

        constexpr ScalarView() = default;

        template <class Other>
            requires std::is_convertible_v<Other*, Value*>
        constexpr ScalarView(const ScalarView<Other> other) : values(other.values) {}

        constexpr explicit ScalarView(Value* next_values) : values(next_values) {}
    };

    template <class Value>
    struct VectorView final {
        Value* x{};
        Value* y{};
        Value* z{};

        constexpr VectorView() = default;

        template <class Other>
            requires std::is_convertible_v<Other*, Value*>
        constexpr VectorView(const VectorView<Other> other) : x(other.x), y(other.y), z(other.z) {}

        constexpr VectorView(Value* next_x, Value* next_y, Value* next_z) : x(next_x), y(next_y), z(next_z) {}
    };

    template <class Value>
    struct Matrix3View final {
        Value* c00{};
        Value* c01{};
        Value* c02{};
        Value* c10{};
        Value* c11{};
        Value* c12{};
        Value* c20{};
        Value* c21{};
        Value* c22{};

        constexpr Matrix3View() = default;

        template <class Other>
            requires std::is_convertible_v<Other*, Value*>
        constexpr Matrix3View(const Matrix3View<Other> other)
            : c00(other.c00), c01(other.c01), c02(other.c02), c10(other.c10), c11(other.c11), c12(other.c12), c20(other.c20), c21(other.c21), c22(other.c22) {}

        constexpr Matrix3View(Value* next_c00, Value* next_c01, Value* next_c02, Value* next_c10, Value* next_c11, Value* next_c12, Value* next_c20, Value* next_c21, Value* next_c22)
            : c00(next_c00), c01(next_c01), c02(next_c02), c10(next_c10), c11(next_c11), c12(next_c12), c20(next_c20), c21(next_c21), c22(next_c22) {}
    };

    template <class Value>
    __device__ inline Vector3<std::remove_const_t<Value>> load(const VectorView<Value> field, const std::uint32_t index) {
        return {field.x[index], field.y[index], field.z[index]};
    }

    template <class Value>
        requires(!std::is_const_v<Value>)
    __device__ inline void store(const VectorView<Value> field, const std::uint32_t index, const Vector3<Value> value) {
        field.x[index] = value.x;
        field.y[index] = value.y;
        field.z[index] = value.z;
    }

    template <class Value>
        requires(!std::is_const_v<Value>)
    __device__ inline void accumulate(const VectorView<Value> field, const std::uint32_t index, const Vector3<Value> value) {
        field.x[index] += value.x;
        field.y[index] += value.y;
        field.z[index] += value.z;
    }

    template <class Field>
    ScalarView<const std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<const Field&>().values.data())>>> scalar_view(const Field& field) {
        return ScalarView<const std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<const Field&>().values.data())>>>{field.values.data()};
    }

    template <class Field>
    ScalarView<std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<Field&>().values.data())>>> scalar_view(Field& field) {
        return ScalarView<std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<Field&>().values.data())>>>{field.values.data()};
    }

    template <class Field>
    VectorView<const std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<const Field&>().x.data())>>> view(const Field& field) {
        return {field.x.data(), field.y.data(), field.z.data()};
    }

    template <class Field>
    VectorView<std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<Field&>().x.data())>>> view(Field& field) {
        return {field.x.data(), field.y.data(), field.z.data()};
    }

    template <class Field>
    Matrix3View<const std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<const Field&>().c00.data())>>> matrix3_view(const Field& field) {
        return {field.c00.data(), field.c01.data(), field.c02.data(), field.c10.data(), field.c11.data(), field.c12.data(), field.c20.data(), field.c21.data(), field.c22.data()};
    }

    template <class Field>
    Matrix3View<std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<Field&>().c00.data())>>> matrix3_view(Field& field) {
        return {field.c00.data(), field.c01.data(), field.c02.data(), field.c10.data(), field.c11.data(), field.c12.data(), field.c20.data(), field.c21.data(), field.c22.data()};
    }

}

#endif
