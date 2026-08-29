#include <physica/fluids/grid/device.cuh>
#include "particle-step-kernels.h"
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda_runtime_api.h>

namespace physica::fluids::liquid::pic::kernels::particle_step {
    namespace {
        constexpr std::uint32_t fluid = 1u;
        __global__ void particle_speeds_kernel(const std::uint32_t particle_count, const field::VectorView<const float> velocities, float* speeds) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            speeds[particle] = length(field::load(velocities, particle));
        }

        __global__ void particle_diagnostics_kernel(const grid::device::Grid grid, const std::uint32_t particle_count, const std::size_t stride, const float particle_mass, const field::VectorView<const float> positions, const field::VectorView<const float> velocities, float* values) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = field::load(positions, particle);
            const Vector3<float> velocity = field::load(velocities, particle);
            const Vector3<float> relative_position{position.x - grid.origin_x, position.y - grid.origin_y, position.z - grid.origin_z};
            values[particle]              = 0.5F * particle_mass * dot(velocity, velocity);
            values[stride + particle]     = particle_mass * velocity.x;
            values[2u * stride + particle] = particle_mass * velocity.y;
            values[3u * stride + particle] = particle_mass * velocity.z;
            values[4u * stride + particle] = particle_mass * (relative_position.y * velocity.z - relative_position.z * velocity.y);
            values[5u * stride + particle] = particle_mass * (relative_position.z * velocity.x - relative_position.x * velocity.z);
            values[6u * stride + particle] = particle_mass * (relative_position.x * velocity.y - relative_position.y * velocity.x);
        }

        __device__ void collide_axis(const float minimum, const float maximum, const float boundary_velocity_value, float& position, float& velocity, bool& collided) {
            if (position < minimum) {
                position = minimum;
                velocity = boundary_velocity_value;
                collided = true;
            }
            if (position > maximum) {
                position = maximum;
                velocity = boundary_velocity_value;
                collided = true;
            }
        }

        __global__ void advect_kernel(const grid::device::Grid grid, const float particle_radius, const bool no_slip, const float time_step, const std::uint32_t particle_count, const field::VectorView<const float> grid_velocity, const field::VectorView<float> positions, const field::VectorView<float> velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = field::load(positions, particle);
            const Vector3<float> first_velocity = grid::device::sample_velocity(grid, position, grid_velocity);
            const Vector3<float> midpoint       = (position + (first_velocity * 0.5F * time_step));
            const Vector3<float> second_velocity = grid::device::sample_velocity(grid, midpoint, grid_velocity);
            Vector3<float> next_position        = (position + (second_velocity * time_step));
            Vector3<float> next_velocity        = field::load(velocities, particle);
            bool collided{};
            collide_axis(grid.origin_x + grid.cell_size + particle_radius, grid.origin_x + (static_cast<float>(grid.nx) - 1.0F) * grid.cell_size - particle_radius, grid.velocity_x, next_position.x, next_velocity.x, collided);
            collide_axis(grid.origin_y + grid.cell_size + particle_radius, grid.origin_y + (static_cast<float>(grid.ny) - 1.0F) * grid.cell_size - particle_radius, grid.velocity_y, next_position.y, next_velocity.y, collided);
            collide_axis(grid.origin_z + grid.cell_size + particle_radius, grid.origin_z + (static_cast<float>(grid.nz) - 1.0F) * grid.cell_size - particle_radius, grid.velocity_z, next_position.z, next_velocity.z, collided);
            if (collided && no_slip) next_velocity = {grid.velocity_x, grid.velocity_y, grid.velocity_z};
            field::store(positions, particle, next_position);
            field::store(velocities, particle, next_velocity);
        }

        __global__ void rank_particles_kernel(const grid::device::Grid grid, const std::uint32_t particle_count, const std::uint32_t maximum_particles_per_cell, const field::VectorView<const float> positions, std::uint32_t* raw_counts, std::uint32_t* survivor_counts, std::uint32_t* keep_flags) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            int x, y, z;
            grid::device::interior_cell(grid, field::load(positions, particle), x, y, z);
            const std::size_t cell = grid::device::cell_index(grid, x, y, z);
            const std::uint32_t rank = atomicAdd(raw_counts + cell, 1u);
            keep_flags[particle] = rank < maximum_particles_per_cell ? 1u : 0u;
            if (rank < maximum_particles_per_cell) atomicAdd(survivor_counts + cell, 1u);
        }

        __global__ void survivor_total_kernel(const std::uint32_t particle_count, const std::uint32_t* keep_flags, const std::uint32_t* destinations, std::uint32_t* totals) {
            if (threadIdx.x != 0u || blockIdx.x != 0u) return;
            totals[0] = particle_count == 0u ? 0u : destinations[particle_count - 1u] + keep_flags[particle_count - 1u];
        }

        __global__ void seed_counts_kernel(const grid::device::Grid grid, const std::uint32_t minimum_particles_per_cell, const std::uint32_t target_particles_per_cell, const std::uint32_t* cell_types, const float* level_set, const std::uint32_t* survivor_counts, std::uint32_t* seed_counts) {
            const std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cell >= grid::device::cell_count(grid)) return;
            const bool interior_liquid = cell_types[cell] == fluid && level_set[cell] < -0.25F * grid.cell_size;
            seed_counts[cell] = interior_liquid && survivor_counts[cell] < minimum_particles_per_cell ? target_particles_per_cell - survivor_counts[cell] : 0u;
        }

        __global__ void seed_total_kernel(const std::size_t cell_count, const std::uint32_t particle_count, const std::uint32_t* seed_counts, const std::uint32_t* seed_offsets, std::uint32_t* totals) {
            if (threadIdx.x != 0u || blockIdx.x != 0u) return;
            const std::uint32_t requested = cell_count == 0u ? 0u : seed_offsets[cell_count - 1u] + seed_counts[cell_count - 1u];
            totals[1] = ::cuda::std::min(requested, particle_count - totals[0]);
        }

        __global__ void compact_kernel(const std::uint32_t particle_count, const field::VectorView<const float> source_positions, const field::VectorView<const float> source_velocities, const std::uint32_t* keep_flags, const std::uint32_t* destinations, const field::VectorView<float> output_positions, const field::VectorView<float> output_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count || keep_flags[particle] == 0u) return;
            const std::uint32_t destination = destinations[particle];
            field::store(output_positions, destination, field::load(source_positions, particle));
            field::store(output_velocities, destination, field::load(source_velocities, particle));
        }

        __device__ std::uint32_t hash(std::uint32_t value) {
            value ^= value >> 16u;
            value *= 0x7feb352du;
            value ^= value >> 15u;
            value *= 0x846ca68bu;
            return value ^ (value >> 16u);
        }

        __device__ float random_fraction(const std::uint32_t value) {
            return static_cast<float>(hash(value) & 0x00ffffffu) / 16777216.0F;
        }

        __global__ void seed_kernel(const grid::device::Grid grid, const std::uint64_t seed, const std::uint32_t survivor_count, const std::uint32_t seed_count, const std::uint32_t* seed_counts, const std::uint32_t* seed_offsets, const field::VectorView<const float> grid_velocity, const field::VectorView<float> output_positions, const field::VectorView<float> output_velocities) {
            const std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cell >= grid::device::cell_count(grid) || seed_counts[cell] == 0u) return;
            int x, y, z;
            grid::device::decode(cell, static_cast<int>(grid.nx), static_cast<int>(grid.ny), x, y, z);
            for (std::uint32_t local = 0u; local < seed_counts[cell]; ++local) {
                const std::uint32_t seed_index = seed_offsets[cell] + local;
                if (seed_index >= seed_count) return;
                const std::uint32_t particle = survivor_count + seed_index;
                const std::uint32_t key      = static_cast<std::uint32_t>(seed) ^ static_cast<std::uint32_t>(cell) * 0x9e3779b9u ^ local * 0x85ebca6bu;
                const Vector3<float> position{
                    grid.origin_x + (static_cast<float>(x) + 0.2F + 0.6F * random_fraction(key)) * grid.cell_size,
                    grid.origin_y + (static_cast<float>(y) + 0.2F + 0.6F * random_fraction(key + 1u)) * grid.cell_size,
                    grid.origin_z + (static_cast<float>(z) + 0.2F + 0.6F * random_fraction(key + 2u)) * grid.cell_size,
                };
                field::store(output_positions, particle, position);
                field::store(output_velocities, particle, grid::device::sample_velocity(grid, position, grid_velocity));
            }
        }
    } // namespace

    std::size_t scan_storage_size(const std::size_t count) {
        std::size_t bytes{};
        cub::DeviceScan::ExclusiveSum(nullptr, bytes, static_cast<const std::uint32_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), count);
        return bytes;
    }

    std::size_t reduction_storage_size(const std::size_t count) {
        std::size_t maximum_bytes{};
        std::size_t sum_bytes{};
        cub::DeviceReduce::Max(nullptr, maximum_bytes, static_cast<const float*>(nullptr), static_cast<float*>(nullptr), count);
        cub::DeviceReduce::Sum(nullptr, sum_bytes, static_cast<const float*>(nullptr), static_cast<float*>(nullptr), count);
        return maximum_bytes > sum_bytes ? maximum_bytes : sum_bytes;
    }

    void particle_speeds(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const field::VectorView<const float> velocities, float* speeds) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), particle_speeds_kernel, particle_count, velocities, speeds);
    }

    void particle_diagnostics(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t particle_count, const std::size_t stride, const float particle_mass, const field::VectorView<const float> positions, const field::VectorView<const float> velocities, float* values) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), particle_diagnostics_kernel, grid, particle_count, stride, particle_mass, positions, velocities, values);
    }

    void reduce_maximum(const ::cuda::stream_ref stream, const float* values, const std::size_t count, float* output, void* scratch, std::size_t scratch_bytes) {
        cub::DeviceReduce::Max(scratch, scratch_bytes, values, output, count, stream.get());
    }

    void reduce_sum(const ::cuda::stream_ref stream, const float* values, const std::size_t count, float* output, void* scratch, std::size_t scratch_bytes) {
        cub::DeviceReduce::Sum(scratch, scratch_bytes, values, output, count, stream.get());
    }

    void advect(const ::cuda::stream_ref stream, const grid::device::Grid grid, const float particle_radius, const bool no_slip, const float time_step, const std::uint32_t particle_count, const field::VectorView<const float> grid_velocity, const field::VectorView<float> positions, const field::VectorView<float> velocities) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), advect_kernel, grid, particle_radius, no_slip, time_step, particle_count, grid_velocity, positions, velocities);
    }

    void plan_maintenance(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t particle_count, const std::uint32_t minimum_particles_per_cell, const std::uint32_t target_particles_per_cell, const std::uint32_t maximum_particles_per_cell, const field::VectorView<const float> positions, const std::uint32_t* cell_types, const float* level_set, std::uint32_t* raw_counts, std::uint32_t* survivor_counts, std::uint32_t* keep_flags, std::uint32_t* destinations, std::uint32_t* seed_counts, std::uint32_t* seed_offsets, std::uint32_t* totals, void* scan_scratch, std::size_t scan_scratch_bytes) {
        cudaMemsetAsync(raw_counts, 0, grid::device::cell_count(grid) * sizeof(std::uint32_t), stream.get());
        cudaMemsetAsync(survivor_counts, 0, grid::device::cell_count(grid) * sizeof(std::uint32_t), stream.get());
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), rank_particles_kernel, grid, particle_count, maximum_particles_per_cell, positions, raw_counts, survivor_counts, keep_flags);
        cub::DeviceScan::ExclusiveSum(scan_scratch, scan_scratch_bytes, keep_flags, destinations, particle_count, stream.get());
        ::cuda::launch(stream, ::cuda::distribute<1u>(1u), survivor_total_kernel, particle_count, keep_flags, destinations, totals);
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::cell_count(grid)), seed_counts_kernel, grid, minimum_particles_per_cell, target_particles_per_cell, cell_types, level_set, survivor_counts, seed_counts);
        cub::DeviceScan::ExclusiveSum(scan_scratch, scan_scratch_bytes, seed_counts, seed_offsets, grid::device::cell_count(grid), stream.get());
        ::cuda::launch(stream, ::cuda::distribute<1u>(1u), seed_total_kernel, grid::device::cell_count(grid), particle_count, seed_counts, seed_offsets, totals);
    }

    void compact_and_seed(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint64_t seed, const std::uint32_t particle_count, const std::uint32_t survivor_count, const std::uint32_t seed_count, const field::VectorView<const float> source_positions, const field::VectorView<const float> source_velocities, const std::uint32_t* keep_flags, const std::uint32_t* destinations, const std::uint32_t* seed_counts, const std::uint32_t* seed_offsets, const field::VectorView<const float> grid_velocity, const field::VectorView<float> output_positions, const field::VectorView<float> output_velocities) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), compact_kernel, particle_count, source_positions, source_velocities, keep_flags, destinations, output_positions, output_velocities);
        if (seed_count > 0u) ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::cell_count(grid)), seed_kernel, grid, seed, survivor_count, seed_count, seed_counts, seed_offsets, grid_velocity, output_positions, output_velocities);
    }
} // namespace physica::fluids::liquid::pic::kernels::particle_step
