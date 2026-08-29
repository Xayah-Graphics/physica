#include "kernels.h"
#include <cuda/launch>
#include <cuda/std/random>
#include <cuda_runtime.h>

namespace physica::reconstruction::pinfs::kernels {
    namespace {
        inline constexpr std::uint32_t physics_random_domain = 3u;

        enum class PhysicsRandomSequence : std::uint32_t {
            voxel,
        };

        __device__ std::uint32_t random_voxel(const std::uint32_t seed, const std::uint32_t step, const std::uint32_t sample) {
            ::cuda::std::philox4x32 random{seed};
            random.set_counter({physics_random_domain, static_cast<std::uint32_t>(PhysicsRandomSequence::voxel), step, sample});
            return random();
        }

        __global__ void sample_physics_kernel(const Vector3<float>* voxel_positions, const std::uint32_t voxel_count, float* points, float* derivatives, Vector3<float>* positions, const std::uint32_t sample_count, const float time, const std::uint32_t seed, const std::uint32_t step) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            const std::uint32_t voxel = random_voxel(seed, step, sample) % voxel_count;
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const float value                                         = voxel_positions[voxel][component];
                points[static_cast<std::size_t>(sample) * 4u + component] = value;
                positions[sample][component]                              = value;
            }
            points[static_cast<std::size_t>(sample) * 4u + 3u] = time;
            for (std::uint32_t derivative = 0u; derivative < 4u; ++derivative)
                for (std::uint32_t component = 0u; component < 4u; ++component) derivatives[static_cast<std::size_t>(derivative) * sample_count * 4u + static_cast<std::size_t>(sample) * 4u + component] = derivative == component ? 1.0F : 0.0F;
        }

        __global__ void physics_loss_kernel(const float* density, const float* density_derivatives, const float* velocity, const float* velocity_derivatives, const float* static_sdf, const float* static_sdf_derivatives, const float* inverse_deviation, float* velocity_adjoints, float* velocity_derivative_adjoints, double* losses, const std::uint32_t sample_count, const float physics_weight, const float velocity_fading, const float neumann_weight) {
            const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
            if (sample >= sample_count) return;
            constexpr float equation_weights[6]{2.0F, 0.001F, 0.001F, 0.001F, 0.005F, 0.005F};
            float velocity_value[3];
            float density_gradient[4];
            float velocity_gradient[4][3];
            for (std::uint32_t component = 0u; component < 3u; ++component) velocity_value[component] = velocity[static_cast<std::size_t>(sample) * 3u + component];
            for (std::uint32_t derivative = 0u; derivative < 4u; ++derivative) {
                density_gradient[derivative] = density_derivatives[static_cast<std::size_t>(derivative) * sample_count * 4u + static_cast<std::size_t>(sample) * 4u + 3u];
                for (std::uint32_t component = 0u; component < 3u; ++component) velocity_gradient[derivative][component] = velocity_derivatives[static_cast<std::size_t>(derivative) * sample_count * 3u + static_cast<std::size_t>(sample) * 3u + component];
            }
            float equations[6];
            equations[0] = density_gradient[3];
            for (std::uint32_t spatial = 0u; spatial < 3u; ++spatial) equations[0] = fmaf(velocity_value[spatial], density_gradient[spatial], equations[0]);
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                equations[component + 1u] = velocity_gradient[3][component];
                for (std::uint32_t spatial = 0u; spatial < 3u; ++spatial) equations[component + 1u] = fmaf(velocity_value[spatial], velocity_gradient[spatial][component], equations[component + 1u]);
            }
            equations[4]      = velocity_gradient[0][0] + velocity_gradient[1][1] + velocity_gradient[2][2];
            equations[5]      = 0.1F * (velocity_value[0] * velocity_value[0] + velocity_value[1] * velocity_value[1] + velocity_value[2] * velocity_value[2]);
            const float scale = physics_weight * velocity_fading / static_cast<float>(sample_count);
            double physics{};
            for (std::uint32_t equation = 0u; equation < 6u; ++equation) physics += static_cast<double>(equations[equation]) * equations[equation] * equation_weights[equation] * scale;
            const float density_material_adjoint = 2.0F * equations[0] * equation_weights[0] * scale;
            for (std::uint32_t spatial = 0u; spatial < 3u; ++spatial) velocity_adjoints[static_cast<std::size_t>(sample) * 3u + spatial] += density_material_adjoint * density_gradient[spatial];
            for (std::uint32_t component = 0u; component < 3u; ++component) {
                const float material_adjoint = 2.0F * equations[component + 1u] * equation_weights[component + 1u] * scale;
                velocity_derivative_adjoints[static_cast<std::size_t>(3u) * sample_count * 3u + static_cast<std::size_t>(sample) * 3u + component] += material_adjoint;
                for (std::uint32_t spatial = 0u; spatial < 3u; ++spatial) {
                    velocity_adjoints[static_cast<std::size_t>(sample) * 3u + spatial] += material_adjoint * velocity_gradient[spatial][component];
                    velocity_derivative_adjoints[static_cast<std::size_t>(spatial) * sample_count * 3u + static_cast<std::size_t>(sample) * 3u + component] += material_adjoint * velocity_value[spatial];
                }
            }
            const float divergence_adjoint = 2.0F * equations[4] * equation_weights[4] * scale;
            for (std::uint32_t spatial = 0u; spatial < 3u; ++spatial) velocity_derivative_adjoints[static_cast<std::size_t>(spatial) * sample_count * 3u + static_cast<std::size_t>(sample) * 3u + spatial] += divergence_adjoint;
            const float speed_adjoint = 2.0F * equations[5] * equation_weights[5] * scale;
            for (std::uint32_t component = 0u; component < 3u; ++component) velocity_adjoints[static_cast<std::size_t>(sample) * 3u + component] += speed_adjoint * 0.2F * velocity_value[component];
            atomicAdd(losses, physics);

            if (static_sdf != nullptr && neumann_weight > 0.0F) {
                const float sdf            = static_sdf[static_cast<std::size_t>(sample) * 257u];
                const float opaque_density = fminf(inverse_deviation[0] / (expf(inverse_deviation[0] * sdf) + 1.0F), 5.0F);
                float dot{};
                for (std::uint32_t component = 0u; component < 3u; ++component) dot = fmaf(velocity_value[component], static_sdf_derivatives[static_cast<std::size_t>(component) * sample_count * 257u + static_cast<std::size_t>(sample) * 257u], dot);
                if (dot < 0.0F) {
                    const float neumann_scale = opaque_density * neumann_weight / static_cast<float>(sample_count);
                    for (std::uint32_t component = 0u; component < 3u; ++component) velocity_adjoints[static_cast<std::size_t>(sample) * 3u + component] -= neumann_scale * static_sdf_derivatives[static_cast<std::size_t>(component) * sample_count * 257u + static_cast<std::size_t>(sample) * 257u];
                    const double neumann = static_cast<double>(-dot) * neumann_scale;
                    atomicAdd(losses, neumann);
                    atomicAdd(losses + 1u, neumann);
                }
            }
        }
    } // namespace

    void sample_physics(const ::cuda::stream_ref stream, const Vector3<float>* voxel_positions, const std::uint32_t voxel_count, float* points, float* derivatives, Vector3<float>* positions, const std::uint32_t sample_count, const float time, const std::uint32_t seed, const std::uint32_t step) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), sample_physics_kernel, voxel_positions, voxel_count, points, derivatives, positions, sample_count, time, seed, step);
    }

    void physics_loss(const ::cuda::stream_ref stream, const float* density, const float* density_derivatives, const float* velocity, const float* velocity_derivatives, const float* static_sdf, const float* static_sdf_derivatives, const float* inverse_deviation, float* velocity_adjoints, float* velocity_derivative_adjoints, double* losses, const std::uint32_t sample_count, const float physics_weight, const float velocity_fading, const float neumann_weight) {
        ::cuda::launch(stream, ::cuda::distribute<128u>(sample_count), physics_loss_kernel, density, density_derivatives, velocity, velocity_derivatives, static_sdf, static_sdf_derivatives, inverse_deviation, velocity_adjoints, velocity_derivative_adjoints, losses, sample_count, physics_weight, velocity_fading, neumann_weight);
    }
} // namespace physica::reconstruction::pinfs::kernels
