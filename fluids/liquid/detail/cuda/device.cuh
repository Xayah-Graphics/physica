#ifndef PHYSICA_FLUIDS_LIQUID_DETAIL_CUDA_DEVICE_CUH
#define PHYSICA_FLUIDS_LIQUID_DETAIL_CUDA_DEVICE_CUH

#include "types.h"
#include <cuda_runtime.h>

namespace physica::fluids::liquid::cuda_detail {
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

    struct CellRange final {
        std::uint32_t first;
        std::uint32_t last;
        std::uint32_t boundary_first;
        std::uint32_t boundary_last;
        bool valid;
    };

    __device__ inline Float3 boundary_position(const BoundaryView boundary, const std::uint32_t index) {
        return {
            boundary.position_x[index] + boundary.time * boundary.velocity_x[index],
            boundary.position_y[index] + boundary.time * boundary.velocity_y[index],
            boundary.position_z[index] + boundary.time * boundary.velocity_z[index],
        };
    }

    __device__ inline Float3 boundary_velocity(const BoundaryView boundary, const std::uint32_t index) {
        return {boundary.velocity_x[index], boundary.velocity_y[index], boundary.velocity_z[index]};
    }

    __device__ inline std::uint32_t cell_index(const NeighborhoodView neighborhood, const int x, const int y, const int z) {
        return (static_cast<std::uint32_t>(z) * neighborhood.cells_y + static_cast<std::uint32_t>(y)) * neighborhood.cells_x + static_cast<std::uint32_t>(x);
    }

    __device__ inline CellRange cell_range(const NeighborhoodView neighborhood, const int x, const int y, const int z) {
        if (x < 0 || x >= static_cast<int>(neighborhood.cells_x) || y < 0 || y >= static_cast<int>(neighborhood.cells_y) || z < 0 || z >= static_cast<int>(neighborhood.cells_z)) return {};
        const std::uint32_t cell = cell_index(neighborhood, x, y, z);
        return {
            .first          = neighborhood.cell_offsets[cell],
            .last           = neighborhood.cell_offsets[cell + 1u],
            .boundary_first = neighborhood.boundary_cell_offsets[cell],
            .boundary_last  = neighborhood.boundary_cell_offsets[cell + 1u],
            .valid          = true,
        };
    }

    __device__ inline void particle_cell(const NeighborhoodView neighborhood, const Float3 position, int& x, int& y, int& z) {
        x = min(static_cast<int>(neighborhood.cells_x) - 1, max(0, __float2int_rd((position.x - neighborhood.origin_x) / neighborhood.cell_size)));
        y = min(static_cast<int>(neighborhood.cells_y) - 1, max(0, __float2int_rd((position.y - neighborhood.origin_y) / neighborhood.cell_size)));
        z = min(static_cast<int>(neighborhood.cells_z) - 1, max(0, __float2int_rd((position.z - neighborhood.origin_z) / neighborhood.cell_size)));
    }

    constexpr float pi = 3.14159265358979323846F;

    __device__ inline float poly6(const Float3 displacement, const float support_radius) {
        const float squared_distance = dot(displacement, displacement);
        const float squared_radius   = support_radius * support_radius;
        if (squared_distance >= squared_radius) return 0.0F;
        const float difference = squared_radius - squared_distance;
        return 315.0F * difference * difference * difference / (64.0F * pi * squared_radius * squared_radius * squared_radius * support_radius * support_radius * support_radius);
    }

    __device__ inline Float3 poly6_gradient(const Float3 displacement, const float support_radius) {
        const float squared_distance = dot(displacement, displacement);
        const float squared_radius   = support_radius * support_radius;
        if (squared_distance >= squared_radius) return {};
        const float difference = squared_radius - squared_distance;
        return scale(displacement, -945.0F * difference * difference / (32.0F * pi * squared_radius * squared_radius * squared_radius * support_radius * support_radius * support_radius));
    }

    __device__ inline float cubic(const Float3 displacement, const float support_radius) {
        const float distance = length(displacement);
        if (distance >= support_radius) return 0.0F;
        const float q           = distance / support_radius;
        const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
        if (q <= 0.5F) return coefficient * (6.0F * q * q * q - 6.0F * q * q + 1.0F);
        const float difference = 1.0F - q;
        return coefficient * 2.0F * difference * difference * difference;
    }

    __device__ inline Float3 cubic_gradient(const Float3 displacement, const float support_radius) {
        const float distance = length(displacement);
        if (distance == 0.0F || distance >= support_radius) return {};
        const float q           = distance / support_radius;
        const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
        const float derivative  = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
        return scale(displacement, coefficient * derivative / (support_radius * distance));
    }

    __device__ inline Float3 cubic_gradient_tangent(const Float3 displacement, const Float3 displacement_tangent, const float support_radius) {
        const float distance    = length(displacement);
        const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
        if (distance == 0.0F) return scale(displacement_tangent, -12.0F * coefficient / (support_radius * support_radius));
        if (distance >= support_radius) return {};
        const float q                       = distance / support_radius;
        const float derivative              = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
        const float second_derivative       = q <= 0.5F ? 36.0F * q - 12.0F : 12.0F * (1.0F - q);
        const float radial_tangent          = dot(displacement, displacement_tangent) / distance;
        const float radial_scale            = coefficient * derivative / (support_radius * distance);
        const float radial_scale_derivative = coefficient * (second_derivative / (support_radius * support_radius * distance) - derivative / (support_radius * distance * distance));
        return add(scale(displacement_tangent, radial_scale), scale(displacement, radial_scale_derivative * radial_tangent));
    }

    __device__ inline Double3 cubic_hessian_product(const Float3 displacement, const Double3 vector, const float support_radius) {
        const double distance    = sqrt(static_cast<double>(dot(displacement, displacement)));
        const double coefficient = 8.0 / (static_cast<double>(pi) * support_radius * support_radius * support_radius);
        if (distance == 0.0) return scale(vector, -12.0 * coefficient / (support_radius * support_radius));
        if (distance >= support_radius) return {};
        const double q                       = distance / support_radius;
        const double derivative              = q <= 0.5 ? 18.0 * q * q - 12.0 * q : -6.0 * (1.0 - q) * (1.0 - q);
        const double second_derivative       = q <= 0.5 ? 36.0 * q - 12.0 : 12.0 * (1.0 - q);
        const double radial_scale            = coefficient * derivative / (support_radius * distance);
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
} // namespace physica::fluids::liquid::cuda_detail

#endif
