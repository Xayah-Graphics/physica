#include "neighborhood-kernels.h"
#include <cub/device/device_radix_sort.cuh>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>

namespace physica::fluids::liquid::operators::kernels::neighborhood {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __device__ std::uint32_t lower_bound(const std::uint64_t* keys, const std::uint32_t count, const std::uint64_t key) {
            std::uint32_t first = 0u;
            std::uint32_t last  = count;
            while (first < last) {
                const std::uint32_t middle = first + (last - first) / 2u;
                if (keys[middle] < key) first = middle + 1u;
                else last = middle;
            }
            return first;
        }

        __global__ void key_kernel(const std::uint32_t count, const float support_radius, const device::CollisionBox collision_box, const float time, const bool moving_positions, const field::VectorView<const float> positions, const field::VectorView<const float> velocities, std::uint64_t* keys, std::uint32_t* indices) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            float x = positions.x[index];
            float y = positions.y[index];
            float z = positions.z[index];
            if (moving_positions) {
                x += time * velocities.x[index];
                y += time * velocities.y[index];
                z += time * velocities.z[index];
            }
            const std::uint32_t cells_x = static_cast<std::uint32_t>(::cuda::std::ceil((collision_box.bounds.maximum.x - collision_box.bounds.minimum.x) / support_radius));
            const std::uint32_t cells_y = static_cast<std::uint32_t>(::cuda::std::ceil((collision_box.bounds.maximum.y - collision_box.bounds.minimum.y) / support_radius));
            const std::uint32_t cells_z = static_cast<std::uint32_t>(::cuda::std::ceil((collision_box.bounds.maximum.z - collision_box.bounds.minimum.z) / support_radius));
            const int cell_x            = ::cuda::std::clamp(__float2int_rd((x - collision_box.bounds.minimum.x - time * collision_box.velocity.x) / support_radius), 0, static_cast<int>(cells_x) - 1);
            const int cell_y            = ::cuda::std::clamp(__float2int_rd((y - collision_box.bounds.minimum.y - time * collision_box.velocity.y) / support_radius), 0, static_cast<int>(cells_y) - 1);
            const int cell_z            = ::cuda::std::clamp(__float2int_rd((z - collision_box.bounds.minimum.z - time * collision_box.velocity.z) / support_radius), 0, static_cast<int>(cells_z) - 1);
            keys[index]                 = (static_cast<std::uint64_t>(cell_z) * cells_y + static_cast<std::uint64_t>(cell_y)) * cells_x + static_cast<std::uint64_t>(cell_x);
            indices[index]              = index;
        }

        __global__ void cell_offsets_kernel(const std::uint32_t item_count, const std::uint64_t* sorted_keys, const std::uint32_t cell_count, std::uint32_t* cell_offsets) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            if (cell > cell_count) return;
            cell_offsets[cell] = lower_bound(sorted_keys, item_count, cell);
        }
    } // namespace

    std::size_t radix_sort_storage_size(const std::uint32_t count) {
        std::size_t bytes{};
        cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<std::uint64_t*>(nullptr), static_cast<std::uint64_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), count);
        return bytes;
    }

    void build_neighborhood(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const std::uint32_t boundary_count, const float support_radius, const float time_step, const std::uint64_t step_index, const device::CollisionBox collision_box, const field::VectorView<const float> positions, const field::VectorView<const float> boundary_positions, const field::VectorView<const float> boundary_velocities, std::uint64_t* unsorted_keys, std::uint32_t* unsorted_particle_indices, std::uint64_t* unsorted_boundary_keys, std::uint32_t* unsorted_boundary_indices, void* sort_scratch, std::size_t sort_scratch_bytes, void* boundary_sort_scratch, std::size_t boundary_sort_scratch_bytes, std::uint64_t* sorted_keys, std::uint32_t* sorted_particle_indices, std::uint32_t* cell_offsets, std::uint64_t* sorted_boundary_keys, std::uint32_t* sorted_boundary_indices, std::uint32_t* boundary_cell_offsets) {
        const float time = static_cast<float>(step_index) * time_step;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), key_kernel, particle_count, support_radius, collision_box, time, false, positions, field::VectorView<const float>{}, unsorted_keys, unsorted_particle_indices);
        cub::DeviceRadixSort::SortPairs(sort_scratch, sort_scratch_bytes, unsorted_keys, sorted_keys, unsorted_particle_indices, sorted_particle_indices, particle_count, 0, 64, stream.get());
        const std::uint32_t cells_x    = static_cast<std::uint32_t>(::cuda::std::ceil((collision_box.bounds.maximum.x - collision_box.bounds.minimum.x) / support_radius));
        const std::uint32_t cells_y    = static_cast<std::uint32_t>(::cuda::std::ceil((collision_box.bounds.maximum.y - collision_box.bounds.minimum.y) / support_radius));
        const std::uint32_t cells_z    = static_cast<std::uint32_t>(::cuda::std::ceil((collision_box.bounds.maximum.z - collision_box.bounds.minimum.z) / support_radius));
        const std::uint32_t cell_count = cells_x * cells_y * cells_z;
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count + 1u), cell_offsets_kernel, particle_count, sorted_keys, cell_count, cell_offsets);
        if (boundary_count == 0u) {
            ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count + 1u), cell_offsets_kernel, 0u, static_cast<const std::uint64_t*>(nullptr), cell_count, boundary_cell_offsets);
            return;
        }
        ::cuda::launch(stream, ::cuda::distribute<block_size>(boundary_count), key_kernel, boundary_count, support_radius, collision_box, time, true, boundary_positions, boundary_velocities, unsorted_boundary_keys, unsorted_boundary_indices);
        cub::DeviceRadixSort::SortPairs(boundary_sort_scratch, boundary_sort_scratch_bytes, unsorted_boundary_keys, sorted_boundary_keys, unsorted_boundary_indices, sorted_boundary_indices, boundary_count, 0, 64, stream.get());
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count + 1u), cell_offsets_kernel, boundary_count, sorted_boundary_keys, cell_count, boundary_cell_offsets);
    }
} // namespace physica::fluids::liquid::operators::kernels::neighborhood
