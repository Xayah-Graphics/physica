#ifndef PHYSICA_FLUIDS_LIQUID_DEVICE_CUH
#define PHYSICA_FLUIDS_LIQUID_DEVICE_CUH

#include <simulation/field/device.cuh>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/numbers>
#include <cuda_runtime.h>
#include <cstdint>

namespace physica::fluids::liquid::device {
    struct CollisionBox final {
        AxisAlignedBox3<float> bounds;
        Vector3<float> velocity;
        std::uint32_t no_slip;
    };

    struct BoundaryView final {
        simulation::VectorView<const float> positions;
        simulation::VectorView<const float> velocities;
        const float* volumes;
        std::uint32_t count;
        float time;
    };

    __device__ inline Vector3<float> boundary_position(const BoundaryView boundary, const std::uint32_t index) {
        return (simulation::load(boundary.positions, index) + (simulation::load(boundary.velocities, index) * boundary.time));
    }

    __device__ inline Vector3<float> boundary_velocity(const BoundaryView boundary, const std::uint32_t index) {
        return simulation::load(boundary.velocities, index);
    }

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

    struct CellRange final {
        std::uint32_t first;
        std::uint32_t last;
        std::uint32_t boundary_first;
        std::uint32_t boundary_last;
        bool valid;
    };

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

    __device__ inline void particle_cell(const NeighborhoodView neighborhood, const Vector3<float> position, int& x, int& y, int& z) {
        const int cell_x = static_cast<int>(::cuda::std::floor((position.x - neighborhood.origin_x) / neighborhood.cell_size));
        const int cell_y = static_cast<int>(::cuda::std::floor((position.y - neighborhood.origin_y) / neighborhood.cell_size));
        const int cell_z = static_cast<int>(::cuda::std::floor((position.z - neighborhood.origin_z) / neighborhood.cell_size));
        x                = ::cuda::std::clamp(cell_x, 0, static_cast<int>(neighborhood.cells_x) - 1);
        y                = ::cuda::std::clamp(cell_y, 0, static_cast<int>(neighborhood.cells_y) - 1);
        z                = ::cuda::std::clamp(cell_z, 0, static_cast<int>(neighborhood.cells_z) - 1);
    }

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

    __device__ inline float poly6(const Vector3<float> displacement, const float support_radius) {
        const float squared_distance = dot(displacement, displacement);
        const float squared_radius   = support_radius * support_radius;
        if (squared_distance >= squared_radius) return 0.0F;
        const float difference = squared_radius - squared_distance;
        return 315.0F * difference * difference * difference / (64.0F * ::cuda::std::numbers::pi_v<float> * squared_radius * squared_radius * squared_radius * support_radius * support_radius * support_radius);
    }

    __device__ inline Vector3<float> poly6_gradient(const Vector3<float> displacement, const float support_radius) {
        const float squared_distance = dot(displacement, displacement);
        const float squared_radius   = support_radius * support_radius;
        if (squared_distance >= squared_radius) return {};
        const float difference = squared_radius - squared_distance;
        return (displacement * -945.0F * difference * difference / (32.0F * ::cuda::std::numbers::pi_v<float> * squared_radius * squared_radius * squared_radius * support_radius * support_radius * support_radius));
    }

    __device__ inline float cubic(const Vector3<float> displacement, const float support_radius) {
        const float distance = length(displacement);
        if (distance >= support_radius) return 0.0F;
        const float q           = distance / support_radius;
        const float coefficient = 8.0F / (::cuda::std::numbers::pi_v<float> * support_radius * support_radius * support_radius);
        if (q <= 0.5F) return coefficient * (6.0F * q * q * q - 6.0F * q * q + 1.0F);
        const float difference = 1.0F - q;
        return coefficient * 2.0F * difference * difference * difference;
    }

    __device__ inline Vector3<float> cubic_gradient(const Vector3<float> displacement, const float support_radius) {
        const float distance = length(displacement);
        if (distance == 0.0F || distance >= support_radius) return {};
        const float q           = distance / support_radius;
        const float coefficient = 8.0F / (::cuda::std::numbers::pi_v<float> * support_radius * support_radius * support_radius);
        const float derivative  = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
        return (displacement * coefficient * derivative / (support_radius * distance));
    }

    __device__ inline Vector3<float> cubic_gradient_tangent(const Vector3<float> displacement, const Vector3<float> displacement_tangent, const float support_radius) {
        const float distance    = length(displacement);
        const float coefficient = 8.0F / (::cuda::std::numbers::pi_v<float> * support_radius * support_radius * support_radius);
        if (distance == 0.0F) return (displacement_tangent * -12.0F * coefficient / (support_radius * support_radius));
        if (distance >= support_radius) return {};
        const float q                       = distance / support_radius;
        const float derivative              = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
        const float second_derivative       = q <= 0.5F ? 36.0F * q - 12.0F : 12.0F * (1.0F - q);
        const float radial_tangent          = dot(displacement, displacement_tangent) / distance;
        const float radial_scale            = coefficient * derivative / (support_radius * distance);
        const float radial_scale_derivative = coefficient * (second_derivative / (support_radius * support_radius * distance) - derivative / (support_radius * distance * distance));
        return ((displacement_tangent * radial_scale) + (displacement * radial_scale_derivative * radial_tangent));
    }

    __device__ inline Vector3<double> cubic_hessian_product(const Vector3<float> displacement, const Vector3<double> vector, const float support_radius) {
        const double distance    = ::cuda::std::sqrt(static_cast<double>(dot(displacement, displacement)));
        const double coefficient = 8.0 / (::cuda::std::numbers::pi_v<double> * support_radius * support_radius * support_radius);
        if (distance == 0.0) return (vector * -12.0 * coefficient / (support_radius * support_radius));
        if (distance >= support_radius) return {};
        const double q                       = distance / support_radius;
        const double derivative              = q <= 0.5 ? 18.0 * q * q - 12.0 * q : -6.0 * (1.0 - q) * (1.0 - q);
        const double second_derivative       = q <= 0.5 ? 36.0 * q - 12.0 : 12.0 * (1.0 - q);
        const double radial_scale            = coefficient * derivative / (support_radius * distance);
        const double radial_scale_derivative = coefficient * (second_derivative / (support_radius * support_radius * distance) - derivative / (support_radius * distance * distance));
        return ((vector * radial_scale) + (displacement * radial_scale_derivative * dot(vector, displacement) / distance));
    }

    __device__ inline float viscosity_laplacian(const Vector3<float> displacement, const float support_radius) {
        const float distance = length(displacement);
        if (distance >= support_radius) return 0.0F;
        return 45.0F * (support_radius - distance) / (::cuda::std::numbers::pi_v<float> * support_radius * support_radius * support_radius * support_radius * support_radius * support_radius);
    }

    __device__ inline float viscosity_laplacian_tangent(const Vector3<float> displacement, const Vector3<float> displacement_tangent, const float support_radius) {
        const float distance = length(displacement);
        if (distance == 0.0F || distance >= support_radius) return 0.0F;
        return -45.0F * dot(displacement, displacement_tangent) / (::cuda::std::numbers::pi_v<float> * support_radius * support_radius * support_radius * support_radius * support_radius * support_radius * distance);
    }

    __device__ inline Vector3<double> viscosity_laplacian_gradient(const Vector3<float> displacement, const float support_radius, const double scalar) {
        const double distance = ::cuda::std::sqrt(static_cast<double>(dot(displacement, displacement)));
        if (distance == 0.0 || distance >= support_radius) return {};
        return (displacement * -45.0 * scalar / (::cuda::std::numbers::pi_v<double> * ::cuda::std::pow(static_cast<double>(support_radius), 6.0) * distance));
    }
}

#endif
