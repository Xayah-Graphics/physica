#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_DENSITY_DEVICE_CUH
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_DENSITY_DEVICE_CUH

#include "../domain/device.cuh"
#include <cuda_runtime.h>

namespace physica::fluids::liquid::particle::cuda_detail {
    constexpr float pi = 3.14159265358979323846F;

    __device__ inline float poly6(const Float3 displacement, const float support_radius) {
        const float squared_distance = dot(displacement, displacement);
        const float squared_radius = support_radius * support_radius;
        if (squared_distance >= squared_radius) return 0.0F;
        const float difference = squared_radius - squared_distance;
        return 315.0F * difference * difference * difference / (64.0F * pi * squared_radius * squared_radius * squared_radius * support_radius * support_radius * support_radius);
    }

    __device__ inline Float3 poly6_gradient(const Float3 displacement, const float support_radius) {
        const float squared_distance = dot(displacement, displacement);
        const float squared_radius = support_radius * support_radius;
        if (squared_distance >= squared_radius) return {};
        const float difference = squared_radius - squared_distance;
        return scale(displacement, -945.0F * difference * difference / (32.0F * pi * squared_radius * squared_radius * squared_radius * support_radius * support_radius * support_radius));
    }

    __device__ inline float cubic(const Float3 displacement, const float support_radius) {
        const float distance = length(displacement);
        if (distance >= support_radius) return 0.0F;
        const float q = distance / support_radius;
        const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
        if (q <= 0.5F) return coefficient * (6.0F * q * q * q - 6.0F * q * q + 1.0F);
        const float difference = 1.0F - q;
        return coefficient * 2.0F * difference * difference * difference;
    }

    __device__ inline Float3 cubic_gradient(const Float3 displacement, const float support_radius) {
        const float distance = length(displacement);
        if (distance == 0.0F || distance >= support_radius) return {};
        const float q = distance / support_radius;
        const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
        const float derivative = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
        return scale(displacement, coefficient * derivative / (support_radius * distance));
    }

    __device__ inline Float3 cubic_gradient_tangent(const Float3 displacement, const Float3 displacement_tangent, const float support_radius) {
        const float distance = length(displacement);
        const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
        if (distance == 0.0F) return scale(displacement_tangent, -12.0F * coefficient / (support_radius * support_radius));
        if (distance >= support_radius) return {};
        const float q = distance / support_radius;
        const float derivative = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
        const float second_derivative = q <= 0.5F ? 36.0F * q - 12.0F : 12.0F * (1.0F - q);
        const float radial_tangent = dot(displacement, displacement_tangent) / distance;
        const float radial_scale = coefficient * derivative / (support_radius * distance);
        const float radial_scale_derivative = coefficient * (second_derivative / (support_radius * support_radius * distance) - derivative / (support_radius * distance * distance));
        return add(scale(displacement_tangent, radial_scale), scale(displacement, radial_scale_derivative * radial_tangent));
    }

    __device__ inline Double3 cubic_hessian_product(const Float3 displacement, const Double3 vector, const float support_radius) {
        const double distance = sqrt(static_cast<double>(dot(displacement, displacement)));
        const double coefficient = 8.0 / (static_cast<double>(pi) * support_radius * support_radius * support_radius);
        if (distance == 0.0) return scale(vector, -12.0 * coefficient / (support_radius * support_radius));
        if (distance >= support_radius) return {};
        const double q = distance / support_radius;
        const double derivative = q <= 0.5 ? 18.0 * q * q - 12.0 * q : -6.0 * (1.0 - q) * (1.0 - q);
        const double second_derivative = q <= 0.5 ? 36.0 * q - 12.0 : 12.0 * (1.0 - q);
        const double radial_scale = coefficient * derivative / (support_radius * distance);
        const double radial_scale_derivative = coefficient * (second_derivative / (support_radius * support_radius * distance) - derivative / (support_radius * distance * distance));
        return add(scale(vector, radial_scale), scale(displacement, radial_scale_derivative * dot(vector, displacement) / distance));
    }

    __device__ inline float viscosity_laplacian(const Float3 displacement, const float support_radius) {
        const float distance = length(displacement);
        if (distance >= support_radius) return 0.0F;
        return 45.0F * (support_radius - distance) / (pi * support_radius * support_radius * support_radius * support_radius * support_radius * support_radius);
    }

    __device__ inline float viscosity_laplacian_tangent(const Float3 displacement, const Float3 displacement_tangent, const float support_radius) {
        const float distance = length(displacement);
        if (distance == 0.0F || distance >= support_radius) return 0.0F;
        return -45.0F * dot(displacement, displacement_tangent) / (pi * support_radius * support_radius * support_radius * support_radius * support_radius * support_radius * distance);
    }

    __device__ inline Double3 viscosity_laplacian_gradient(const Float3 displacement, const float support_radius, const double scalar) {
        const double distance = sqrt(static_cast<double>(dot(displacement, displacement)));
        if (distance == 0.0 || distance >= support_radius) return {};
        return scale(displacement, -45.0 * scalar / (static_cast<double>(pi) * pow(static_cast<double>(support_radius), 6.0) * distance));
    }
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
