#ifndef PHYSICA_DEFORMABLES_CLOTH_DETAIL_CUDA_DEVICE_CUH
#define PHYSICA_DEFORMABLES_CLOTH_DETAIL_CUDA_DEVICE_CUH

#include "types.h"

namespace physica::deformables::cloth::cuda_detail {
    template <class Scalar>
    __device__ Vector<Scalar> operator+(const Vector<Scalar> left, const Vector<Scalar> right) {
        return {left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <class Scalar>
    __device__ Vector<Scalar> operator-(const Vector<Scalar> left, const Vector<Scalar> right) {
        return {left.x - right.x, left.y - right.y, left.z - right.z};
    }

    template <class Scalar>
    __device__ Vector<Scalar> operator*(const Scalar scalar, const Vector<Scalar> vector) {
        return {scalar * vector.x, scalar * vector.y, scalar * vector.z};
    }

    template <class Scalar>
    __device__ Vector<Scalar> operator/(const Vector<Scalar> vector, const Scalar scalar) {
        return {vector.x / scalar, vector.y / scalar, vector.z / scalar};
    }

    template <class Scalar>
    __device__ Scalar dot(const Vector<Scalar> left, const Vector<Scalar> right) {
        return left.x * right.x + left.y * right.y + left.z * right.z;
    }

    template <class Scalar>
    __device__ Vector<Scalar> load(const ConstFieldView<Scalar> field, const unsigned int index) {
        return {field.x[index], field.y[index], field.z[index]};
    }

    template <class Scalar>
    __device__ void store(const FieldView<Scalar> field, const unsigned int index, const Vector<Scalar> value) {
        field.x[index] = value.x;
        field.y[index] = value.y;
        field.z[index] = value.z;
    }

    template <class Scalar>
    __device__ void add(const FieldView<Scalar> field, const unsigned int index, const Vector<Scalar> value) {
        field.x[index] += value.x;
        field.y[index] += value.y;
        field.z[index] += value.z;
    }
} // namespace physica::deformables::cloth::cuda_detail

#endif
