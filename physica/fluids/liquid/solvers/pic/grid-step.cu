#include "grid-step-kernels.h"
#include <cuda/launch>
#include <cuda/std/bit>
#include <cuda/std/cmath>

namespace physica::fluids::liquid::solvers::pic::kernels::grid_step {
    namespace {
        constexpr std::uint32_t air   = 0u;
        constexpr std::uint32_t fluid = 1u;
        constexpr std::uint32_t solid = 2u;

        __device__ float atomic_minimum(float* address, const float value) {
            int previous = *reinterpret_cast<int*>(address);
            while (value < ::cuda::std::bit_cast<float>(previous)) {
                const int assumed = previous;
                previous          = atomicCAS(reinterpret_cast<int*>(address), assumed, ::cuda::std::bit_cast<int>(value));
                if (previous == assumed) break;
            }
            return ::cuda::std::bit_cast<float>(previous);
        }

        __device__ std::uint32_t cell_type(const grid::device::Grid grid, const std::uint32_t* cell_types, const int x, const int y, const int z) {
            if (!grid::device::valid_cell(grid, x, y, z)) return solid;
            return cell_types[grid::device::cell_index(grid, x, y, z)];
        }

        __global__ void initialize_cells_kernel(const grid::device::Grid grid, std::uint32_t* particle_counts, std::uint32_t* cell_types, float* level_set) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::cell_count(grid)) return;
            int x, y, z;
            grid::device::decode(index, static_cast<int>(grid.nx), static_cast<int>(grid.ny), x, y, z);
            particle_counts[index] = 0u;
            cell_types[index]      = x == 0 || y == 0 || z == 0 || x == static_cast<int>(grid.nx) - 1 || y == static_cast<int>(grid.ny) - 1 || z == static_cast<int>(grid.nz) - 1 ? solid : air;
            level_set[index]       = 3.0F * grid.cell_size;
        }

        __global__ void particle_level_set_kernel(const grid::device::Grid grid, const std::uint32_t particle_count, const float level_set_radius, const simulation::VectorView<const float> positions, std::uint32_t* particle_counts, float* level_set) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = simulation::load(positions, particle);
            int cell_x, cell_y, cell_z;
            grid::device::interior_cell(grid, position, cell_x, cell_y, cell_z);
            atomicAdd(particle_counts + grid::device::cell_index(grid, cell_x, cell_y, cell_z), 1u);
            const int radius = static_cast<int>(::cuda::std::ceil((level_set_radius + grid.cell_size) / grid.cell_size));
            for (int z = cell_z - radius; z <= cell_z + radius; ++z)
                for (int y = cell_y - radius; y <= cell_y + radius; ++y)
                    for (int x = cell_x - radius; x <= cell_x + radius; ++x) {
                        if (!grid::device::valid_cell(grid, x, y, z)) continue;
                        const Vector3<float> displacement = (grid::device::cell_position(grid, x, y, z) - position);
                        atomic_minimum(level_set + grid::device::cell_index(grid, x, y, z), length(displacement) - level_set_radius);
                    }
        }

        __global__ void finalize_cells_kernel(const grid::device::Grid grid, const std::uint32_t* particle_counts, std::uint32_t* cell_types, float* level_set) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::cell_count(grid) || cell_types[index] == solid) return;
            if (particle_counts[index] > 0u || level_set[index] < 0.0F) cell_types[index] = fluid;
        }

        __global__ void normalize_kernel(const grid::device::Grid grid, const int axis, const float* mass, float* momentum, std::uint32_t* valid) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::face_count(grid, axis)) return;
            valid[index]    = mass[index] > 0.0F ? 1u : 0u;
            momentum[index] = mass[index] > 0.0F ? momentum[index] / mass[index] : 0.0F;
        }

        __global__ void extrapolate_kernel(const grid::device::Grid grid, const int axis, const float* input, const std::uint32_t* input_valid, float* output, std::uint32_t* output_valid) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::face_count(grid, axis)) return;
            if (input_valid[index] != 0u) {
                output[index]       = input[index];
                output_valid[index] = 1u;
                return;
            }
            int x, y, z;
            grid::device::decode(index, grid::device::extent(grid, axis, 0), grid::device::extent(grid, axis, 1), x, y, z);
            constexpr int offsets[6][3]{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
            float sum{};
            std::uint32_t count{};
            for (int neighbor = 0; neighbor < 6; ++neighbor) {
                const int nx = x + offsets[neighbor][0];
                const int ny = y + offsets[neighbor][1];
                const int nz = z + offsets[neighbor][2];
                if (!grid::device::valid_face(grid, axis, nx, ny, nz)) continue;
                const std::size_t neighbor_index = grid::device::face_index(grid, axis, nx, ny, nz);
                if (input_valid[neighbor_index] == 0u) continue;
                sum += input[neighbor_index];
                ++count;
            }
            output[index]       = count == 0u ? input[index] : sum / static_cast<float>(count);
            output_valid[index] = count == 0u ? 0u : 1u;
        }

        __device__ float boundary_velocity(const grid::device::Grid grid, const int axis) {
            if (axis == 0) return grid.velocity_x;
            if (axis == 1) return grid.velocity_y;
            return grid.velocity_z;
        }

        __device__ bool constrained_face(const grid::device::Grid grid, const bool no_slip, const int axis, const int x, const int y, const int z, const std::uint32_t* cell_types) {
            const int negative_x = x - (axis == 0 ? 1 : 0);
            const int negative_y = y - (axis == 1 ? 1 : 0);
            const int negative_z = z - (axis == 2 ? 1 : 0);
            if (cell_type(grid, cell_types, negative_x, negative_y, negative_z) == solid || cell_type(grid, cell_types, x, y, z) == solid) return true;
            if (!no_slip) return false;
            if (axis != 0 && (x == 0 || x == static_cast<int>(grid.nx) - 1)) return true;
            if (axis != 1 && (y == 0 || y == static_cast<int>(grid.ny) - 1)) return true;
            return axis != 2 && (z == 0 || z == static_cast<int>(grid.nz) - 1);
        }

        __device__ bool fluid_face(const grid::device::Grid grid, const int axis, const int x, const int y, const int z, const std::uint32_t* cell_types) {
            const int negative_x = x - (axis == 0 ? 1 : 0);
            const int negative_y = y - (axis == 1 ? 1 : 0);
            const int negative_z = z - (axis == 2 ? 1 : 0);
            return cell_type(grid, cell_types, negative_x, negative_y, negative_z) == fluid || cell_type(grid, cell_types, x, y, z) == fluid;
        }

        __global__ void force_and_constrain_kernel(const grid::device::Grid grid, const bool no_slip, const float time_step, const Vector3<float> acceleration, const int axis, const std::uint32_t* cell_types, float* velocity) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::face_count(grid, axis)) return;
            int x, y, z;
            grid::device::decode(index, grid::device::extent(grid, axis, 0), grid::device::extent(grid, axis, 1), x, y, z);
            if (constrained_face(grid, no_slip, axis, x, y, z, cell_types)) {
                velocity[index] = boundary_velocity(grid, axis);
                return;
            }
            if (fluid_face(grid, axis, x, y, z, cell_types)) velocity[index] += time_step * acceleration[axis];
        }

        __global__ void mark_fluid_faces_kernel(const grid::device::Grid grid, const bool no_slip, const int axis, const std::uint32_t* cell_types, std::uint32_t* valid) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::face_count(grid, axis)) return;
            int x, y, z;
            grid::device::decode(index, grid::device::extent(grid, axis, 0), grid::device::extent(grid, axis, 1), x, y, z);
            valid[index] = constrained_face(grid, no_slip, axis, x, y, z, cell_types) || fluid_face(grid, axis, x, y, z, cell_types) ? 1u : 0u;
        }

        __global__ void divergence_kernel(const grid::device::Grid grid, const std::uint32_t* cell_types, const simulation::VectorView<const float> velocity, float* divergence, float* metrics) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= grid::device::cell_count(grid)) return;
            if (cell_types[index] != fluid) {
                divergence[index] = 0.0F;
                return;
            }
            int x, y, z;
            grid::device::decode(index, static_cast<int>(grid.nx), static_cast<int>(grid.ny), x, y, z);
            const float value = (velocity.x[grid::device::face_index(grid, 0, x + 1, y, z)] - velocity.x[grid::device::face_index(grid, 0, x, y, z)] + velocity.y[grid::device::face_index(grid, 1, x, y + 1, z)] - velocity.y[grid::device::face_index(grid, 1, x, y, z)] + velocity.z[grid::device::face_index(grid, 2, x, y, z + 1)] - velocity.z[grid::device::face_index(grid, 2, x, y, z)]) / grid.cell_size;
            divergence[index] = value;
            atomicAdd(metrics, value * value);
            atomicMax(reinterpret_cast<unsigned int*>(metrics + 1u), ::cuda::std::bit_cast<unsigned int>(::cuda::std::abs(value)));
            atomicAdd(metrics + 2u, 1.0F);
        }

    } // namespace

    void classify(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t particle_count, const float level_set_radius, const simulation::VectorView<const float> positions, std::uint32_t* particle_counts, std::uint32_t* cell_types, float* level_set) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::cell_count(grid)), initialize_cells_kernel, grid, particle_counts, cell_types, level_set);
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(particle_count), particle_level_set_kernel, grid, particle_count, level_set_radius, positions, particle_counts, level_set);
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::cell_count(grid)), finalize_cells_kernel, grid, particle_counts, cell_types, level_set);
    }

    void normalize(const ::cuda::stream_ref stream, const grid::device::Grid grid, const simulation::VectorView<const float> mass, const simulation::VectorView<float> momentum, const simulation::VectorView<std::uint32_t> valid) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::face_count(grid, axis)), normalize_kernel, grid, axis, grid::device::component(mass, axis), grid::device::component(momentum, axis), grid::device::component(valid, axis));
    }

    void extrapolate_layer(const ::cuda::stream_ref stream, const grid::device::Grid grid, const simulation::VectorView<const float> input, const simulation::VectorView<const std::uint32_t> input_valid, const simulation::VectorView<float> output, const simulation::VectorView<std::uint32_t> output_valid) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::face_count(grid, axis)), extrapolate_kernel, grid, axis, grid::device::component(input, axis), grid::device::component(input_valid, axis), grid::device::component(output, axis), grid::device::component(output_valid, axis));
    }

    void add_force_and_constrain(const ::cuda::stream_ref stream, const grid::device::Grid grid, const bool no_slip, const float time_step, const Vector3<float> acceleration, const std::uint32_t* cell_types, const simulation::VectorView<float> velocity) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::face_count(grid, axis)), force_and_constrain_kernel, grid, no_slip, time_step, acceleration, axis, cell_types, grid::device::component(velocity, axis));
    }

    void mark_fluid_faces(const ::cuda::stream_ref stream, const grid::device::Grid grid, const bool no_slip, const std::uint32_t* cell_types, const simulation::VectorView<std::uint32_t> valid) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::face_count(grid, axis)), mark_fluid_faces_kernel, grid, no_slip, axis, cell_types, grid::device::component(valid, axis));
    }

    void compute_divergence(const ::cuda::stream_ref stream, const grid::device::Grid grid, const std::uint32_t* cell_types, const simulation::VectorView<const float> velocity, float* divergence, float* metrics) {
        ::cuda::launch(stream, ::cuda::distribute<grid::device::block_size>(grid::device::cell_count(grid)), divergence_kernel, grid, cell_types, velocity, divergence, metrics);
    }

} // namespace physica::fluids::liquid::solvers::pic::kernels::grid_step
