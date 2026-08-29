#include <physica/fluids/gas/device.cuh>
#include "pointwise-kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::operators::kernels {
    namespace {
        __global__ void source_forward_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* state, const float* source, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            output[index] = collider_ids[index] == 0u ? state[index] + grid.time_step * source[index] : 0.0F;
        }

        __global__ void source_vjp_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const double* output_adjoint, double* state_adjoint, double* source_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            state_adjoint[index] += output_adjoint[index];
            source_adjoint[index] += grid.time_step * output_adjoint[index];
        }

        __device__ float averaged_center_force(const float* force, const int axis, const int x, const int y, const int z, const device::Discretization grid, const std::uint32_t* collider_ids) {
            int first_x = x, first_y = y, first_z = z;
            int second_x = x, second_y = y, second_z = z;
            if (axis == 0) --first_x;
            if (axis == 1) --first_y;
            if (axis == 2) --first_z;
            float sum   = 0.0F;
            float count = 0.0F;
            if (first_x >= 0 && first_x < grid.grid.nx && first_y >= 0 && first_y < grid.grid.ny && first_z >= 0 && first_z < grid.grid.nz) {
                const std::uint64_t cell = fluids::grid::device::index3(first_x, first_y, first_z, grid.grid.nx, grid.grid.ny);
                if (collider_ids[cell] == 0u) {
                    sum += force[cell];
                    count += 1.0F;
                }
            }
            if (second_x >= 0 && second_x < grid.grid.nx && second_y >= 0 && second_y < grid.grid.ny && second_z >= 0 && second_z < grid.grid.nz) {
                const std::uint64_t cell = fluids::grid::device::index3(second_x, second_y, second_z, grid.grid.nx, grid.grid.ny);
                if (collider_ids[cell] == 0u) {
                    sum += force[cell];
                    count += 1.0F;
                }
            }
            return count == 0.0F ? 0.0F : sum / count;
        }

        __global__ void integrate_kernel(const device::Discretization grid, const int axis, const std::uint32_t* collider_ids, const float* velocity, const float* force, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            output[index] = velocity[index] + grid.time_step * averaged_center_force(force, axis, x, y, z, grid, collider_ids);
        }

        __global__ void integrate_vjp_kernel(const device::Discretization grid, const int axis, const std::uint32_t* collider_ids, const double* output_adjoint, double* velocity_adjoint, double* force_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            velocity_adjoint[index] += output_adjoint[index];
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            int cells[2][3]{{x - (axis == 0), y - (axis == 1), z - (axis == 2)}, {x, y, z}};
            int count = 0;
            for (int side = 0; side < 2; ++side)
                if (cells[side][0] >= 0 && cells[side][0] < grid.grid.nx && cells[side][1] >= 0 && cells[side][1] < grid.grid.ny && cells[side][2] >= 0 && cells[side][2] < grid.grid.nz && collider_ids[fluids::grid::device::index3(cells[side][0], cells[side][1], cells[side][2], grid.grid.nx, grid.grid.ny)] == 0u) ++count;
            if (count == 0) return;
            for (int side = 0; side < 2; ++side)
                if (cells[side][0] >= 0 && cells[side][0] < grid.grid.nx && cells[side][1] >= 0 && cells[side][1] < grid.grid.ny && cells[side][2] >= 0 && cells[side][2] < grid.grid.nz && collider_ids[fluids::grid::device::index3(cells[side][0], cells[side][1], cells[side][2], grid.grid.nx, grid.grid.ny)] == 0u) atomicAdd(force_adjoint + fluids::grid::device::index3(cells[side][0], cells[side][1], cells[side][2], grid.grid.nx, grid.grid.ny), grid.time_step * output_adjoint[index] / count);
        }

        __device__ bool adjacent_solid(const int axis, const int x, const int y, const int z, const device::Discretization grid, const std::uint32_t* collider_ids) {
            int first[3]{x - (axis == 0), y - (axis == 1), z - (axis == 2)};
            int second[3]{x, y, z};
            for (int side = 0; side < 2; ++side) {
                const int* cell = side == 0 ? first : second;
                if (cell[0] >= 0 && cell[0] < grid.grid.nx && cell[1] >= 0 && cell[1] < grid.grid.ny && cell[2] >= 0 && cell[2] < grid.grid.nz && collider_ids[fluids::grid::device::index3(cell[0], cell[1], cell[2], grid.grid.nx, grid.grid.ny)] != 0u) return true;
            }
            return false;
        }

        __device__ bool constrained_boundary(const int axis, const int x, const int y, const int z, const device::Discretization grid, const device::VelocityBoundary boundary) {
            const int coordinate = axis == 0 ? x : axis == 1 ? y : z;
            const int maximum    = axis == 0 ? grid.grid.nx : axis == 1 ? grid.grid.ny : grid.grid.nz;
            if (coordinate != 0 && coordinate != maximum) return false;
            const std::uint32_t mode = boundary.faces[axis * 2 + (coordinate == maximum)].mode;
            return mode == 0u || mode == 2u;
        }

        __device__ float boundary_velocity_value(const int axis, const int x, const int y, const int z, const device::Discretization grid, const device::VelocityBoundary boundary) {
            const int coordinate = axis == 0 ? x : axis == 1 ? y : z;
            const int maximum    = axis == 0 ? grid.grid.nx : axis == 1 ? grid.grid.ny : grid.grid.nz;
            const int face       = axis * 2 + (coordinate == maximum);
            return boundary.faces[face].value[axis];
        }

        __global__ void constrain_velocity_forward_kernel(const device::Discretization grid, const int axis, const std::uint32_t* collider_ids, const float* collider_velocity, const float* velocity, const device::VelocityBoundary boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            if (constrained_boundary(axis, x, y, z, grid, boundary)) {
                output[index] = boundary_velocity_value(axis, x, y, z, grid, boundary);
                return;
            }
            if (adjacent_solid(axis, x, y, z, grid, collider_ids)) {
                output[index] = collider_velocity == nullptr ? 0.0F : collider_velocity[index];
                return;
            }
            if (device::periodic(boundary, axis) && (axis == 0 ? x == grid.grid.nx : axis == 1 ? y == grid.grid.ny : z == grid.grid.nz)) {
                if (axis == 0) x = 0;
                if (axis == 1) y = 0;
                if (axis == 2) z = 0;
                output[index] = velocity[fluids::grid::device::index3(x, y, z, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1))];
                return;
            }
            output[index] = velocity[index];
        }

        __global__ void constrain_velocity_vjp_kernel(const device::Discretization grid, const int axis, const std::uint32_t* collider_ids, const double* output_adjoint, const device::VelocityBoundary boundary, double* velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            if (constrained_boundary(axis, x, y, z, grid, boundary) || adjacent_solid(axis, x, y, z, grid, collider_ids)) return;
            if (device::periodic(boundary, axis) && (axis == 0 ? x == grid.grid.nx : axis == 1 ? y == grid.grid.ny : z == grid.grid.nz)) {
                if (axis == 0) x = 0;
                if (axis == 1) y = 0;
                if (axis == 2) z = 0;
                atomicAdd(velocity_adjoint + fluids::grid::device::index3(x, y, z, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1)), output_adjoint[index]);
                return;
            }
            atomicAdd(velocity_adjoint + index, output_adjoint[index]);
        }
    } // namespace

    void source_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::ScalarView<const float> state, const field::ScalarView<const float> source, const field::ScalarView<float> output) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), source_forward_kernel, grid, collider_ids, state.values, source.values, output.values);
    }

    void source_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::ScalarView<const float> state_tangent, const field::ScalarView<const float> source_tangent, const field::ScalarView<float> output_tangent) {
        source_forward(stream, grid, collider_ids, state_tangent, source_tangent, output_tangent);
    }

    void source_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::ScalarView<const double> output_adjoint, const field::ScalarView<double> state_adjoint, const field::ScalarView<double> source_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), source_vjp_kernel, grid, collider_ids, output_adjoint.values, state_adjoint.values, source_adjoint.values);
    }

    void integrate_velocity_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::VectorView<const float> velocity, const field::VectorView<const float> force, const field::VectorView<float> output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::face_count(grid.grid, axis)), integrate_kernel, grid, axis, collider_ids, fluids::grid::device::component(velocity, axis), fluids::grid::device::component(force, axis), fluids::grid::device::component(output, axis));
    }

    void integrate_velocity_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::VectorView<const float> velocity_tangent, const field::VectorView<const float> force_tangent, const field::VectorView<float> output_tangent) {
        integrate_velocity_forward(stream, grid, collider_ids, velocity_tangent, force_tangent, output_tangent);
    }

    void integrate_velocity_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::VectorView<const double> output_adjoint, const field::VectorView<double> velocity_adjoint, const field::VectorView<double> force_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::face_count(grid.grid, axis)), integrate_vjp_kernel, grid, axis, collider_ids, fluids::grid::device::component(output_adjoint, axis), fluids::grid::device::component(velocity_adjoint, axis), fluids::grid::device::component(force_adjoint, axis));
    }

    void constrain_velocity_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::VectorView<const float> collider_velocity, const field::VectorView<const float> velocity, const device::VelocityBoundary boundary, const field::VectorView<float> output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::face_count(grid.grid, axis)), constrain_velocity_forward_kernel, grid, axis, collider_ids, fluids::grid::device::component(collider_velocity, axis), fluids::grid::device::component(velocity, axis), boundary, fluids::grid::device::component(output, axis));
    }

    void constrain_velocity_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::VectorView<const float> velocity_tangent, const device::VelocityBoundary boundary, const field::VectorView<float> output_tangent) {
        field::VectorView<const float> zero{};
        device::VelocityBoundary tangent_boundary = boundary;
        for (device::VelocityBoundaryFace& face : tangent_boundary.faces) face.value = {};
        constrain_velocity_forward(stream, grid, collider_ids, zero, velocity_tangent, tangent_boundary, output_tangent);
    }

    void constrain_velocity_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const field::VectorView<const double> output_adjoint, const device::VelocityBoundary boundary, const field::VectorView<double> velocity_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::face_count(grid.grid, axis)), constrain_velocity_vjp_kernel, grid, axis, collider_ids, fluids::grid::device::component(output_adjoint, axis), boundary, fluids::grid::device::component(velocity_adjoint, axis));
    }
} // namespace physica::fluids::gas::operators::kernels
