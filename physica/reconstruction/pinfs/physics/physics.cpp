module;

#include "kernels.h"
#include <physica/cuda.h>

module physica.reconstruction.pinfs.physics;

import std;
import physica.reconstruction.pinfs.field;
import physica.reconstruction.pinfs.scene;

namespace physica::reconstruction::pinfs {
    Physics::Physics(const ::cuda::stream_ref source_stream, const std::span<const Vector3<float>> source_voxel_positions)
        : stream{source_stream}, voxel_positions{stream, ::cuda::device_default_memory_pool(stream.device()), source_voxel_positions.size(), ::cuda::no_init}, points{stream, ::cuda::device_default_memory_pool(stream.device()), physics_sample_count, ::cuda::no_init}, point_derivatives{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(physics_sample_count) * 4uz * 4uz, ::cuda::no_init}, position_values{stream, ::cuda::device_default_memory_pool(stream.device()), physics_sample_count, ::cuda::no_init}, velocity_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(physics_sample_count) * 3uz, ::cuda::no_init}, velocity_derivative_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(physics_sample_count) * 3uz * 4uz, ::cuda::no_init}, losses{stream, ::cuda::device_default_memory_pool(stream.device()), 2uz, ::cuda::no_init} {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const Vector3<float>>{source_voxel_positions.data(), source_voxel_positions.size()}, voxel_positions);
    }

    PhysicsSamples Physics::sample(const float time, const std::uint32_t seed, const std::uint32_t step) {
        kernels::sample_physics(stream, voxel_positions.data(), static_cast<std::uint32_t>(voxel_positions.size()), reinterpret_cast<float*>(points.data()), point_derivatives.data(), position_values.data(), physics_sample_count, time, seed, step);
        return {
            .points    = {.values = reinterpret_cast<const float*>(points.data()), .derivatives = point_derivatives.data(), .width = 4u, .sample_count = physics_sample_count, .derivative_count = 4u},
            .positions = position_values.data(),
        };
    }

    PhysicsLoss Physics::loss(const ConstDeviceTensor density, const ConstDeviceTensor velocity, const DeviceTensor* static_sdf, const float* inverse_deviation, const float physics_weight, const float velocity_fading, const float neumann_weight) {
        ::cuda::fill_bytes(stream, velocity_adjoints, 0u);
        ::cuda::fill_bytes(stream, velocity_derivative_adjoints, 0u);
        ::cuda::fill_bytes(stream, losses, 0u);
        kernels::physics_loss(stream, density.values, density.derivatives, velocity.values, velocity.derivatives, static_sdf == nullptr ? nullptr : static_sdf->values, static_sdf == nullptr ? nullptr : static_sdf->derivatives, inverse_deviation, velocity_adjoints.data(), velocity_derivative_adjoints.data(), losses.data(), physics_sample_count, physics_weight, velocity_fading, neumann_weight);
        std::array<double, 2> host{};
        ::cuda::copy_bytes(stream, losses, ::cuda::std::span<double>{host.data(), host.size()});
        stream.sync();
        return {
            .velocity_adjoints = {.values = velocity_adjoints.data(), .derivatives = velocity_derivative_adjoints.data(), .width = 3u, .sample_count = physics_sample_count, .derivative_count = 4u},
            .loss              = static_cast<float>(host[0] - host[1]),
            .neumann           = static_cast<float>(host[1]),
        };
    }
} // namespace physica::reconstruction::pinfs
