#include "sampling.cuh"
#include "kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    namespace {
        __device__ float averaged_center_force(const float* force, const int axis, const int x, const int y, const int z, const Grid grid) {
            const int cells[2][3]{{x - (axis == 0), y - (axis == 1), z - (axis == 2)}, {x, y, z}};
            float sum = 0.0F;
            float count = 0.0F;
            for (int side = 0; side < 2; ++side) {
                if (cells[side][0] < 0 || cells[side][0] >= grid.nx || cells[side][1] < 0 || cells[side][1] >= grid.ny || cells[side][2] < 0 || cells[side][2] >= grid.nz) continue;
                sum += force[index3(cells[side][0], cells[side][1], cells[side][2], grid.nx, grid.ny)];
                count += 1.0F;
            }
            return count == 0.0F ? 0.0F : sum / count;
        }

        __global__ void integrate_kernel(const Grid grid, const int axis, const float* velocity, const float* force, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            output[index] = velocity[index] + grid.time_step * averaged_center_force(force, axis, x, y, z, grid);
        }

        __global__ void integrate_vjp_kernel(const Grid grid, const int axis, const double* output_adjoint, double* velocity_adjoint, double* force_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            velocity_adjoint[index] += output_adjoint[index];
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            const int cells[2][3]{{x - (axis == 0), y - (axis == 1), z - (axis == 2)}, {x, y, z}};
            int count = 0;
            for (int side = 0; side < 2; ++side) if (cells[side][0] >= 0 && cells[side][0] < grid.nx && cells[side][1] >= 0 && cells[side][1] < grid.ny && cells[side][2] >= 0 && cells[side][2] < grid.nz) ++count;
            if (count == 0) return;
            for (int side = 0; side < 2; ++side) if (cells[side][0] >= 0 && cells[side][0] < grid.nx && cells[side][1] >= 0 && cells[side][1] < grid.ny && cells[side][2] >= 0 && cells[side][2] < grid.nz) atomicAdd(force_adjoint + index3(cells[side][0], cells[side][1], cells[side][2], grid.nx, grid.ny), grid.time_step * output_adjoint[index] / count);
        }

        __global__ void advect_velocity_forward_kernel(const Grid grid, const int axis, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            const Trace trace = trace_rk2(face_position(axis, x, y, z, grid), velocity, grid, boundary);
            output[index] = sample_face(component(velocity, axis), axis, trace.position, grid, boundary).value;
        }

        __global__ void advect_velocity_jvp_kernel(const Grid grid, const int axis, const ConstStaggeredVectorView velocity, const ConstStaggeredVectorView velocity_tangent, const VelocityBoundaryData boundary, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            const Vector start = face_position(axis, x, y, z, grid);
            VelocityBoundaryData tangent_boundary = boundary;
            for (float& value : tangent_boundary.values) value = 0.0F;
            const Vector value0 = sample_velocity_value(velocity, start, grid, boundary);
            const Vector tangent0 = sample_velocity_value(velocity_tangent, start, grid, tangent_boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Vector midpoint_tangent{-0.5F * grid.time_step * tangent0.x, -0.5F * grid.time_step * tangent0.y, -0.5F * grid.time_step * tangent0.z};
            const Sample velocity_samples[3]{sample_face(velocity.x, 0, midpoint, grid, boundary), sample_face(velocity.y, 1, midpoint, grid, boundary), sample_face(velocity.z, 2, midpoint, grid, boundary)};
            const Vector sampled_tangent = sample_velocity_value(velocity_tangent, midpoint, grid, tangent_boundary);
            const Vector value1_tangent{
                sampled_tangent.x + velocity_samples[0].gradient.x * midpoint_tangent.x + velocity_samples[0].gradient.y * midpoint_tangent.y + velocity_samples[0].gradient.z * midpoint_tangent.z,
                sampled_tangent.y + velocity_samples[1].gradient.x * midpoint_tangent.x + velocity_samples[1].gradient.y * midpoint_tangent.y + velocity_samples[1].gradient.z * midpoint_tangent.z,
                sampled_tangent.z + velocity_samples[2].gradient.x * midpoint_tangent.x + velocity_samples[2].gradient.y * midpoint_tangent.y + velocity_samples[2].gradient.z * midpoint_tangent.z,
            };
            const Trace trace = trace_rk2(start, velocity, grid, boundary);
            const Vector position_tangent{-trace.derivative.x * grid.time_step * value1_tangent.x, -trace.derivative.y * grid.time_step * value1_tangent.y, -trace.derivative.z * grid.time_step * value1_tangent.z};
            const Sample source_sample = sample_face(component(velocity, axis), axis, trace.position, grid, boundary);
            output_tangent[index] = sample_face(component(velocity_tangent, axis), axis, trace.position, grid, tangent_boundary).value + source_sample.gradient.x * position_tangent.x + source_sample.gradient.y * position_tangent.y + source_sample.gradient.z * position_tangent.z;
        }

        __global__ void advect_velocity_vjp_kernel(const Grid grid, const int axis, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, const double* output_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            const Vector start = face_position(axis, x, y, z, grid);
            const Vector value0 = sample_velocity_value(velocity, start, grid, boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Trace trace = trace_rk2(start, velocity, grid, boundary);
            const Sample source_sample = sample_face(component(velocity, axis), axis, trace.position, grid, boundary);
            scatter_face(component(velocity_adjoint, axis), axis, trace.position, output_adjoint[index], grid, boundary);
            const double value1_adjoint_x = -static_cast<double>(trace.derivative.x * grid.time_step * source_sample.gradient.x) * output_adjoint[index];
            const double value1_adjoint_y = -static_cast<double>(trace.derivative.y * grid.time_step * source_sample.gradient.y) * output_adjoint[index];
            const double value1_adjoint_z = -static_cast<double>(trace.derivative.z * grid.time_step * source_sample.gradient.z) * output_adjoint[index];
            const Sample midpoint_samples[3]{sample_face(velocity.x, 0, midpoint, grid, boundary), sample_face(velocity.y, 1, midpoint, grid, boundary), sample_face(velocity.z, 2, midpoint, grid, boundary)};
            scatter_face(velocity_adjoint.x, 0, midpoint, value1_adjoint_x, grid, boundary);
            scatter_face(velocity_adjoint.y, 1, midpoint, value1_adjoint_y, grid, boundary);
            scatter_face(velocity_adjoint.z, 2, midpoint, value1_adjoint_z, grid, boundary);
            const double midpoint_adjoint_x = midpoint_samples[0].gradient.x * value1_adjoint_x + midpoint_samples[1].gradient.x * value1_adjoint_y + midpoint_samples[2].gradient.x * value1_adjoint_z;
            const double midpoint_adjoint_y = midpoint_samples[0].gradient.y * value1_adjoint_x + midpoint_samples[1].gradient.y * value1_adjoint_y + midpoint_samples[2].gradient.y * value1_adjoint_z;
            const double midpoint_adjoint_z = midpoint_samples[0].gradient.z * value1_adjoint_x + midpoint_samples[1].gradient.z * value1_adjoint_y + midpoint_samples[2].gradient.z * value1_adjoint_z;
            scatter_face(velocity_adjoint.x, 0, start, -0.5 * grid.time_step * midpoint_adjoint_x, grid, boundary);
            scatter_face(velocity_adjoint.y, 1, start, -0.5 * grid.time_step * midpoint_adjoint_y, grid, boundary);
            scatter_face(velocity_adjoint.z, 2, start, -0.5 * grid.time_step * midpoint_adjoint_z, grid, boundary);
        }

        __device__ bool constrained_boundary(const int axis, const int x, const int y, const int z, const Grid grid, const VelocityBoundaryData boundary) {
            const int coordinate = axis == 0 ? x : axis == 1 ? y : z;
            const int maximum = axis == 0 ? grid.nx : axis == 1 ? grid.ny : grid.nz;
            if (coordinate != 0 && coordinate != maximum) return false;
            const std::uint32_t mode = boundary.modes[axis * 2 + (coordinate == maximum)];
            return mode == 0u || mode == 2u;
        }

        __device__ float boundary_velocity_value(const int axis, const int x, const int y, const int z, const Grid grid, const VelocityBoundaryData boundary) {
            const int coordinate = axis == 0 ? x : axis == 1 ? y : z;
            const int maximum = axis == 0 ? grid.nx : axis == 1 ? grid.ny : grid.nz;
            const int face = axis * 2 + (coordinate == maximum);
            return boundary.values[face * 3 + axis];
        }

        __global__ void constrain_velocity_forward_kernel(const Grid grid, const int axis, const float* velocity, const VelocityBoundaryData boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            if (constrained_boundary(axis, x, y, z, grid, boundary)) {
                output[index] = boundary_velocity_value(axis, x, y, z, grid, boundary);
                return;
            }
            if (periodic(boundary, axis) && (axis == 0 ? x == grid.nx : axis == 1 ? y == grid.ny : z == grid.nz)) {
                if (axis == 0) x = 0;
                if (axis == 1) y = 0;
                if (axis == 2) z = 0;
                output[index] = velocity[index3(x, y, z, extent(grid, axis, 0), extent(grid, axis, 1))];
                return;
            }
            output[index] = velocity[index];
        }

        __global__ void constrain_velocity_vjp_kernel(const Grid grid, const int axis, const double* output_adjoint, const VelocityBoundaryData boundary, double* velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            if (constrained_boundary(axis, x, y, z, grid, boundary)) return;
            if (periodic(boundary, axis) && (axis == 0 ? x == grid.nx : axis == 1 ? y == grid.ny : z == grid.nz)) {
                if (axis == 0) x = 0;
                if (axis == 1) y = 0;
                if (axis == 2) z = 0;
                atomicAdd(velocity_adjoint + index3(x, y, z, extent(grid, axis, 0), extent(grid, axis, 1)), output_adjoint[index]);
                return;
            }
            atomicAdd(velocity_adjoint + index, output_adjoint[index]);
        }

        __global__ void advect_scalar_forward_kernel(const Grid grid, const float* source, const ConstStaggeredVectorView velocity, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const Trace trace = trace_rk2(cell_position(x, y, z, grid), velocity, grid, velocity_boundary);
            output[index] = sample_scalar(source, trace.position, grid, scalar_boundary).value;
        }

        __global__ void advect_scalar_jvp_kernel(const Grid grid, const float* source, const float* source_tangent, const ConstStaggeredVectorView velocity, const ConstStaggeredVectorView velocity_tangent, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const Vector start = cell_position(x, y, z, grid);
            ScalarBoundaryData scalar_tangent_boundary = scalar_boundary;
            for (float& value : scalar_tangent_boundary.values) value = 0.0F;
            VelocityBoundaryData velocity_tangent_boundary = velocity_boundary;
            for (float& value : velocity_tangent_boundary.values) value = 0.0F;
            const Vector value0 = sample_velocity_value(velocity, start, grid, velocity_boundary);
            const Vector tangent0 = sample_velocity_value(velocity_tangent, start, grid, velocity_tangent_boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Vector midpoint_tangent{-0.5F * grid.time_step * tangent0.x, -0.5F * grid.time_step * tangent0.y, -0.5F * grid.time_step * tangent0.z};
            const Sample velocity_samples[3]{sample_face(velocity.x, 0, midpoint, grid, velocity_boundary), sample_face(velocity.y, 1, midpoint, grid, velocity_boundary), sample_face(velocity.z, 2, midpoint, grid, velocity_boundary)};
            const Vector sampled_tangent = sample_velocity_value(velocity_tangent, midpoint, grid, velocity_tangent_boundary);
            const Vector value1_tangent{
                sampled_tangent.x + velocity_samples[0].gradient.x * midpoint_tangent.x + velocity_samples[0].gradient.y * midpoint_tangent.y + velocity_samples[0].gradient.z * midpoint_tangent.z,
                sampled_tangent.y + velocity_samples[1].gradient.x * midpoint_tangent.x + velocity_samples[1].gradient.y * midpoint_tangent.y + velocity_samples[1].gradient.z * midpoint_tangent.z,
                sampled_tangent.z + velocity_samples[2].gradient.x * midpoint_tangent.x + velocity_samples[2].gradient.y * midpoint_tangent.y + velocity_samples[2].gradient.z * midpoint_tangent.z,
            };
            const Trace trace = trace_rk2(start, velocity, grid, velocity_boundary);
            const Vector position_tangent{-trace.derivative.x * grid.time_step * value1_tangent.x, -trace.derivative.y * grid.time_step * value1_tangent.y, -trace.derivative.z * grid.time_step * value1_tangent.z};
            const Sample source_sample = sample_scalar(source, trace.position, grid, scalar_boundary);
            output_tangent[index] = sample_scalar(source_tangent, trace.position, grid, scalar_tangent_boundary).value + source_sample.gradient.x * position_tangent.x + source_sample.gradient.y * position_tangent.y + source_sample.gradient.z * position_tangent.z;
        }

        __global__ void advect_scalar_vjp_kernel(const Grid grid, const float* source, const ConstStaggeredVectorView velocity, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, const double* output_adjoint, double* source_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const Vector start = cell_position(x, y, z, grid);
            const Vector value0 = sample_velocity_value(velocity, start, grid, velocity_boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Trace trace = trace_rk2(start, velocity, grid, velocity_boundary);
            const Sample source_sample = sample_scalar(source, trace.position, grid, scalar_boundary);
            scatter_scalar(source_adjoint, trace.position, output_adjoint[index], grid, scalar_boundary);
            const double value1_adjoint_x = -static_cast<double>(trace.derivative.x * grid.time_step * source_sample.gradient.x) * output_adjoint[index];
            const double value1_adjoint_y = -static_cast<double>(trace.derivative.y * grid.time_step * source_sample.gradient.y) * output_adjoint[index];
            const double value1_adjoint_z = -static_cast<double>(trace.derivative.z * grid.time_step * source_sample.gradient.z) * output_adjoint[index];
            const Sample midpoint_samples[3]{sample_face(velocity.x, 0, midpoint, grid, velocity_boundary), sample_face(velocity.y, 1, midpoint, grid, velocity_boundary), sample_face(velocity.z, 2, midpoint, grid, velocity_boundary)};
            scatter_face(velocity_adjoint.x, 0, midpoint, value1_adjoint_x, grid, velocity_boundary);
            scatter_face(velocity_adjoint.y, 1, midpoint, value1_adjoint_y, grid, velocity_boundary);
            scatter_face(velocity_adjoint.z, 2, midpoint, value1_adjoint_z, grid, velocity_boundary);
            const double midpoint_adjoint_x = midpoint_samples[0].gradient.x * value1_adjoint_x + midpoint_samples[1].gradient.x * value1_adjoint_y + midpoint_samples[2].gradient.x * value1_adjoint_z;
            const double midpoint_adjoint_y = midpoint_samples[0].gradient.y * value1_adjoint_x + midpoint_samples[1].gradient.y * value1_adjoint_y + midpoint_samples[2].gradient.y * value1_adjoint_z;
            const double midpoint_adjoint_z = midpoint_samples[0].gradient.z * value1_adjoint_x + midpoint_samples[1].gradient.z * value1_adjoint_y + midpoint_samples[2].gradient.z * value1_adjoint_z;
            scatter_face(velocity_adjoint.x, 0, start, -0.5 * grid.time_step * midpoint_adjoint_x, grid, velocity_boundary);
            scatter_face(velocity_adjoint.y, 1, start, -0.5 * grid.time_step * midpoint_adjoint_y, grid, velocity_boundary);
            scatter_face(velocity_adjoint.z, 2, start, -0.5 * grid.time_step * midpoint_adjoint_z, grid, velocity_boundary);
        }
    } // namespace

    void integrate_velocity_forward(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorView velocity, const ConstCenteredVectorView force, const StaggeredVectorView output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), integrate_kernel, grid, axis, component(velocity, axis), component(force, axis), component(output, axis));
    }

    void integrate_velocity_vjp(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorAdjointView output_adjoint, const StaggeredVectorAdjointView velocity_adjoint, const CenteredVectorAdjointView force_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), integrate_vjp_kernel, grid, axis, component(output_adjoint, axis), component(velocity_adjoint, axis), component(force_adjoint, axis));
    }

    void advect_velocity_forward(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, const StaggeredVectorView output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), advect_velocity_forward_kernel, grid, axis, velocity, boundary, component(output, axis));
    }

    void advect_velocity_jvp(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorView velocity, const ConstStaggeredVectorView velocity_tangent, const VelocityBoundaryData boundary, const StaggeredVectorView output_tangent) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), advect_velocity_jvp_kernel, grid, axis, velocity, velocity_tangent, boundary, component(output_tangent, axis));
    }

    void advect_velocity_vjp(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, const ConstStaggeredVectorAdjointView output_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), advect_velocity_vjp_kernel, grid, axis, velocity, boundary, component(output_adjoint, axis), velocity_adjoint);
    }

    void constrain_velocity_forward(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, const StaggeredVectorView output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), constrain_velocity_forward_kernel, grid, axis, component(velocity, axis), boundary, component(output, axis));
    }

    void constrain_velocity_vjp(const ::cuda::stream_ref stream, const Grid grid, const ConstStaggeredVectorAdjointView output_adjoint, const VelocityBoundaryData boundary, const StaggeredVectorAdjointView velocity_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<block_size>(face_count(grid, axis)), constrain_velocity_vjp_kernel, grid, axis, component(output_adjoint, axis), boundary, component(velocity_adjoint, axis));
    }

    void advect_scalar_forward(const ::cuda::stream_ref stream, const Grid grid, const ConstScalarView source, const ConstStaggeredVectorView velocity, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, const ScalarView output) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), advect_scalar_forward_kernel, grid, source.values, velocity, scalar_boundary, velocity_boundary, output.values);
    }

    void advect_scalar_jvp(const ::cuda::stream_ref stream, const Grid grid, const ConstScalarView source, const ConstScalarView source_tangent, const ConstStaggeredVectorView velocity, const ConstStaggeredVectorView velocity_tangent, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, const ScalarView output_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), advect_scalar_jvp_kernel, grid, source.values, source_tangent.values, velocity, velocity_tangent, scalar_boundary, velocity_boundary, output_tangent.values);
    }

    void advect_scalar_vjp(const ::cuda::stream_ref stream, const Grid grid, const ConstScalarView source, const ConstStaggeredVectorView velocity, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, const ConstScalarAdjointView output_adjoint, const ScalarAdjointView source_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), advect_scalar_vjp_kernel, grid, source.values, velocity, scalar_boundary, velocity_boundary, output_adjoint.values, source_adjoint.values, velocity_adjoint);
    }
} // namespace physica::fluids::gas::keyframe_smoke::cuda_detail
