module;

#include <physica/cuda.h>

export module physica.reconstruction.pinfs.physics;

import std;
import physica.reconstruction.pinfs.field;
import physica.reconstruction.pinfs.network;
import physica.reconstruction.pinfs.scene;

export namespace physica::reconstruction::pinfs {
    inline constexpr std::uint32_t physics_sample_count = 32u * 32u * 32u;

    struct PhysicsSamples final {
        ConstDeviceTensor points;
        const Vector3<float>* positions{};
    };

    struct PhysicsLoss final {
        ConstDeviceTensor velocity_adjoints;
        float loss{};
        float neumann{};
    };

    struct Physics final {
        Physics(::cuda::stream_ref stream, std::span<const Vector3<float>> voxel_positions);

        PhysicsSamples sample(float time, std::uint32_t seed, std::uint32_t step);
        PhysicsLoss loss(ConstDeviceTensor density, ConstDeviceTensor velocity, const DeviceTensor* static_sdf, const float* inverse_deviation, float physics_weight, float velocity_fading, float neumann_weight);

    private:
        ::cuda::stream_ref stream;
        ::cuda::device_buffer<Vector3<float>> voxel_positions;
        ::cuda::device_buffer<SpacetimePoint> points;
        ::cuda::device_buffer<float> point_derivatives;
        ::cuda::device_buffer<Vector3<float>> position_values;
        ::cuda::device_buffer<float> velocity_adjoints;
        ::cuda::device_buffer<float> velocity_derivative_adjoints;
        ::cuda::device_buffer<double> losses;
    };
} // namespace physica::reconstruction::pinfs
