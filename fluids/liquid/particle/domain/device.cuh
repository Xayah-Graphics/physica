#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_DOMAIN_DEVICE_CUH
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_DOMAIN_DEVICE_CUH

#include "device.h"
#include <cuda_runtime.h>

namespace physica::fluids::liquid::particle::cuda_detail {
    __device__ inline Float3 load(const ConstVectorView<float> field, const std::uint32_t index) {
        return {field.x[index], field.y[index], field.z[index]};
    }

    __device__ inline Double3 load(const ConstVectorView<double> field, const std::uint32_t index) {
        return {field.x[index], field.y[index], field.z[index]};
    }

    __device__ inline void store(const VectorView<float> field, const std::uint32_t index, const Float3 value) {
        field.x[index] = value.x;
        field.y[index] = value.y;
        field.z[index] = value.z;
    }

    __device__ inline void accumulate(const VectorView<double> field, const std::uint32_t index, const Double3 value) {
        field.x[index] += value.x;
        field.y[index] += value.y;
        field.z[index] += value.z;
    }

    __device__ inline Float3 add(const Float3 first, const Float3 second) {
        return {first.x + second.x, first.y + second.y, first.z + second.z};
    }

    __device__ inline Double3 add(const Double3 first, const Double3 second) {
        return {first.x + second.x, first.y + second.y, first.z + second.z};
    }

    __device__ inline Float3 subtract(const Float3 first, const Float3 second) {
        return {first.x - second.x, first.y - second.y, first.z - second.z};
    }

    __device__ inline Double3 subtract(const Double3 first, const Double3 second) {
        return {first.x - second.x, first.y - second.y, first.z - second.z};
    }

    __device__ inline Float3 scale(const Float3 value, const float factor) {
        return {factor * value.x, factor * value.y, factor * value.z};
    }

    __device__ inline Double3 scale(const Float3 value, const double factor) {
        return {factor * value.x, factor * value.y, factor * value.z};
    }

    __device__ inline Double3 scale(const Double3 value, const double factor) {
        return {factor * value.x, factor * value.y, factor * value.z};
    }

    __device__ inline float dot(const Float3 first, const Float3 second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    __device__ inline double dot(const Double3 first, const Float3 second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    __device__ inline double dot(const Double3 first, const Double3 second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    __device__ inline Float3 cross(const Float3 first, const Float3 second) {
        return {first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x};
    }

    __device__ inline Double3 cross(const Float3 first, const Double3 second) {
        return {first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x};
    }

    __device__ inline Double3 cross(const Double3 first, const Float3 second) {
        return {first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x};
    }

    __device__ inline float length(const Float3 value) {
        return sqrtf(dot(value, value));
    }
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
