#include "transfer-kernels.h"
#include <cuda/launch>
#include <fluids/grid/device.cuh>

namespace physica::fluids::liquid::solvers::pic::kernels::transfer {
    namespace {
        __global__ void flip_particle_to_grid_kernel(const grid::device::Grid grid, const std::uint32_t particle_count, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<float> momentum, const simulation::VectorView<float> mass) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = simulation::load(positions, particle);
            const Vector3<float> velocity = simulation::load(velocities, particle);
            for (int axis = 0; axis < 3; ++axis) {
                int base_x, base_y, base_z;
                float weights_x[3], weights_y[3], weights_z[3];
                grid::device::face_stencil(grid, position, axis, base_x, base_y, base_z, weights_x, weights_y, weights_z);
                for (int oz = 0; oz < 3; ++oz)
                    for (int oy = 0; oy < 3; ++oy)
                        for (int ox = 0; ox < 3; ++ox) {
                            const int x = base_x + ox;
                            const int y = base_y + oy;
                            const int z = base_z + oz;
                            if (!grid::device::valid_face(grid, axis, x, y, z)) continue;
                            const float weight     = weights_x[ox] * weights_y[oy] * weights_z[oz];
                            const std::size_t face = grid::device::face_index(grid, axis, x, y, z);
                            atomicAdd(grid::device::component(mass, axis) + face, weight);
                            atomicAdd(grid::device::component(momentum, axis) + face, weight * velocity[axis]);
                        }
            }
        }

        __global__ void flip_grid_to_particle_kernel(const grid::device::Grid grid, const std::uint32_t particle_count, const float flip_ratio, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> input_velocities, const simulation::VectorView<const float> old_grid_velocity, const simulation::VectorView<const float> new_grid_velocity, const simulation::VectorView<float> output_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = simulation::load(positions, particle);
            const Vector3<float> input    = simulation::load(input_velocities, particle);
            const Vector3<float> old_grid = grid::device::sample_velocity(grid, position, old_grid_velocity);
            const Vector3<float> new_grid = grid::device::sample_velocity(grid, position, new_grid_velocity);
            const Vector3<float> flip     = (input + (new_grid - old_grid));
            simulation::store(output_velocities, particle, flip * flip_ratio + new_grid * (1.0F - flip_ratio));
        }
    } // namespace

    void flip_particle_to_grid(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t particle_count, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::VectorView<float> momentum, const simulation::VectorView<float> mass) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), flip_particle_to_grid_kernel, grid, particle_count, positions, velocities, momentum, mass);
    }

    void flip_grid_to_particle(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t particle_count, const float flip_ratio, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> input_velocities, const simulation::VectorView<const float> old_grid_velocity, const simulation::VectorView<const float> new_grid_velocity, const simulation::VectorView<float> output_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), flip_grid_to_particle_kernel, grid, particle_count, flip_ratio, positions, input_velocities, old_grid_velocity, new_grid_velocity, output_velocities);
    }
} // namespace physica::fluids::liquid::solvers::pic::kernels::transfer

namespace physica::fluids::liquid::solvers::pic::kernels::transfer {
    namespace {
        struct SymmetricMatrix final {
            float m00;
            float m01;
            float m02;
            float m11;
            float m12;
            float m22;
        };

        __device__ float affine_component(const simulation::Matrix3View<const float> affine, const std::uint32_t index, const int axis, const Vector3<float> displacement) {
            if (axis == 0) return affine.c00[index] * displacement.x + affine.c01[index] * displacement.y + affine.c02[index] * displacement.z;
            if (axis == 1) return affine.c10[index] * displacement.x + affine.c11[index] * displacement.y + affine.c12[index] * displacement.z;
            return affine.c20[index] * displacement.x + affine.c21[index] * displacement.y + affine.c22[index] * displacement.z;
        }

        __device__ void affine_row(const grid::device::Grid grid, const Vector3<float> position, const int axis, const float* values, const float ratio, float& c0, float& c1, float& c2) {
            int base_x, base_y, base_z;
            float weights_x[3], weights_y[3], weights_z[3];
            grid::device::face_stencil(grid, position, axis, base_x, base_y, base_z, weights_x, weights_y, weights_z);
            float weight_sum{};
            for (int oz = 0; oz < 3; ++oz)
                for (int oy = 0; oy < 3; ++oy)
                    for (int ox = 0; ox < 3; ++ox)
                        if (grid::device::valid_face(grid, axis, base_x + ox, base_y + oy, base_z + oz)) weight_sum += weights_x[ox] * weights_y[oy] * weights_z[oz];
            SymmetricMatrix covariance{};
            Vector3<float> moment{};
            for (int oz = 0; oz < 3; ++oz)
                for (int oy = 0; oy < 3; ++oy)
                    for (int ox = 0; ox < 3; ++ox) {
                        const int x = base_x + ox;
                        const int y = base_y + oy;
                        const int z = base_z + oz;
                        if (!grid::device::valid_face(grid, axis, x, y, z)) continue;
                        const float weight                = weights_x[ox] * weights_y[oy] * weights_z[oz] / weight_sum;
                        const Vector3<float> displacement = (grid::device::face_position(grid, axis, x, y, z) - position);
                        covariance.m00 += weight * displacement.x * displacement.x;
                        covariance.m01 += weight * displacement.x * displacement.y;
                        covariance.m02 += weight * displacement.x * displacement.z;
                        covariance.m11 += weight * displacement.y * displacement.y;
                        covariance.m12 += weight * displacement.y * displacement.z;
                        covariance.m22 += weight * displacement.z * displacement.z;
                        const float velocity = values[grid::device::face_index(grid, axis, x, y, z)];
                        moment.x += weight * velocity * displacement.x;
                        moment.y += weight * velocity * displacement.y;
                        moment.z += weight * velocity * displacement.z;
                    }
            const float determinant = covariance.m00 * (covariance.m11 * covariance.m22 - covariance.m12 * covariance.m12) - covariance.m01 * (covariance.m01 * covariance.m22 - covariance.m12 * covariance.m02) + covariance.m02 * (covariance.m01 * covariance.m12 - covariance.m11 * covariance.m02);
            const float inverse00   = (covariance.m11 * covariance.m22 - covariance.m12 * covariance.m12) / determinant;
            const float inverse01   = (covariance.m02 * covariance.m12 - covariance.m01 * covariance.m22) / determinant;
            const float inverse02   = (covariance.m01 * covariance.m12 - covariance.m02 * covariance.m11) / determinant;
            const float inverse11   = (covariance.m00 * covariance.m22 - covariance.m02 * covariance.m02) / determinant;
            const float inverse12   = (covariance.m01 * covariance.m02 - covariance.m00 * covariance.m12) / determinant;
            const float inverse22   = (covariance.m00 * covariance.m11 - covariance.m01 * covariance.m01) / determinant;
            c0                      = ratio * (moment.x * inverse00 + moment.y * inverse01 + moment.z * inverse02);
            c1                      = ratio * (moment.x * inverse01 + moment.y * inverse11 + moment.z * inverse12);
            c2                      = ratio * (moment.x * inverse02 + moment.y * inverse12 + moment.z * inverse22);
        }

        __device__ void reconstruct_affine(const grid::device::Grid grid, const Vector3<float> position, const simulation::VectorView<const float> velocity, const float ratio, const std::uint32_t particle, const simulation::Matrix3View<float> affine) {
            affine_row(grid, position, 0, velocity.x, ratio, affine.c00[particle], affine.c01[particle], affine.c02[particle]);
            affine_row(grid, position, 1, velocity.y, ratio, affine.c10[particle], affine.c11[particle], affine.c12[particle]);
            affine_row(grid, position, 2, velocity.z, ratio, affine.c20[particle], affine.c21[particle], affine.c22[particle]);
        }

        __global__ void apic_particle_to_grid_kernel(const grid::device::Grid grid, const std::uint32_t particle_count, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::Matrix3View<const float> affine, const simulation::VectorView<float> momentum, const simulation::VectorView<float> mass) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = simulation::load(positions, particle);
            const Vector3<float> velocity = simulation::load(velocities, particle);
            for (int axis = 0; axis < 3; ++axis) {
                int base_x, base_y, base_z;
                float weights_x[3], weights_y[3], weights_z[3];
                grid::device::face_stencil(grid, position, axis, base_x, base_y, base_z, weights_x, weights_y, weights_z);
                for (int oz = 0; oz < 3; ++oz)
                    for (int oy = 0; oy < 3; ++oy)
                        for (int ox = 0; ox < 3; ++ox) {
                            const int x = base_x + ox;
                            const int y = base_y + oy;
                            const int z = base_z + oz;
                            if (!grid::device::valid_face(grid, axis, x, y, z)) continue;
                            const float weight                = weights_x[ox] * weights_y[oy] * weights_z[oz];
                            const Vector3<float> displacement = (grid::device::face_position(grid, axis, x, y, z) - position);
                            const float face_velocity         = velocity[axis] + affine_component(affine, particle, axis, displacement);
                            const std::size_t face            = grid::device::face_index(grid, axis, x, y, z);
                            atomicAdd(grid::device::component(mass, axis) + face, weight);
                            atomicAdd(grid::device::component(momentum, axis) + face, weight * face_velocity);
                        }
            }
        }

        __global__ void apic_grid_to_particle_kernel(const grid::device::Grid grid, const std::uint32_t particle_count, const float affine_ratio, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> grid_velocity, const simulation::VectorView<float> output_velocities, const simulation::Matrix3View<float> output_affine) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = simulation::load(positions, particle);
            simulation::store(output_velocities, particle, grid::device::sample_velocity(grid, position, grid_velocity));
            reconstruct_affine(grid, position, grid_velocity, affine_ratio, particle, output_affine);
        }

        __global__ void compact_affine_kernel(const std::uint32_t particle_count, const simulation::Matrix3View<const float> source, const std::uint32_t* keep_flags, const std::uint32_t* destinations, const simulation::Matrix3View<float> output) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count || keep_flags[particle] == 0u) return;
            const std::uint32_t destination = destinations[particle];
            output.c00[destination]         = source.c00[particle];
            output.c01[destination]         = source.c01[particle];
            output.c02[destination]         = source.c02[particle];
            output.c10[destination]         = source.c10[particle];
            output.c11[destination]         = source.c11[particle];
            output.c12[destination]         = source.c12[particle];
            output.c20[destination]         = source.c20[particle];
            output.c21[destination]         = source.c21[particle];
            output.c22[destination]         = source.c22[particle];
        }

        __global__ void seed_affine_kernel(const grid::device::Grid grid, const std::uint32_t survivor_count, const std::uint32_t seed_count, const float affine_ratio, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> grid_velocity, const simulation::Matrix3View<float> output) {
            const std::uint32_t seed = blockIdx.x * blockDim.x + threadIdx.x;
            if (seed >= seed_count) return;
            const std::uint32_t particle = survivor_count + seed;
            reconstruct_affine(grid, simulation::load(positions, particle), grid_velocity, affine_ratio, particle, output);
        }
    } // namespace

    void apic_particle_to_grid(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t particle_count, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> velocities, const simulation::Matrix3View<const float> affine, const simulation::VectorView<float> momentum, const simulation::VectorView<float> mass) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), apic_particle_to_grid_kernel, grid, particle_count, positions, velocities, affine, momentum, mass);
    }

    void apic_grid_to_particle(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t particle_count, const float affine_ratio, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> grid_velocity, const simulation::VectorView<float> output_velocities, const simulation::Matrix3View<float> output_affine) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), apic_grid_to_particle_kernel, grid, particle_count, affine_ratio, positions, grid_velocity, output_velocities, output_affine);
    }

    void apic_compact_and_seed(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t source_particle_count, const std::uint32_t survivor_count, const std::uint32_t seed_count, const float affine_ratio, const simulation::VectorView<const float> compacted_positions, const simulation::VectorView<const float> grid_velocity, const simulation::Matrix3View<const float> source_affine, const std::uint32_t* keep_flags, const std::uint32_t* destinations, const simulation::Matrix3View<float> output_affine) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(source_particle_count), compact_affine_kernel, source_particle_count, source_affine, keep_flags, destinations, output_affine);
        if (seed_count > 0u) ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(seed_count), seed_affine_kernel, grid, survivor_count, seed_count, affine_ratio, compacted_positions, grid_velocity, output_affine);
    }
} // namespace physica::fluids::liquid::solvers::pic::kernels::transfer
