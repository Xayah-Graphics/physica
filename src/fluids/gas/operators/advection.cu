#include "advection-kernels.h"
#include "advection-sampling.cuh"
#include <cuda/launch>

namespace physica::fluids::gas::operators::cuda_backend {
    namespace {
        __global__ void advect_velocity_forward_kernel(const detail::cuda::Grid grid, const int axis, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::VelocityBoundaryData boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::face_count(grid, axis)) return;
            int x, y, z;
            detail::cuda::decode(index, detail::cuda::extent(grid, axis, 0), detail::cuda::extent(grid, axis, 1), x, y, z);
            const Trace trace = trace_rk2(detail::cuda::face_position(axis, x, y, z, grid), velocity, collider_ids, grid, boundary);
            output[index]     = sample_face(detail::cuda::component(velocity, axis), axis, trace.position, grid, boundary).value;
        }

        __global__ void advect_velocity_jvp_kernel(const detail::cuda::Grid grid, const int axis, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::StaggeredVectorView<const float> velocity_tangent, const detail::cuda::VelocityBoundaryData boundary, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::face_count(grid, axis)) return;
            int x, y, z;
            detail::cuda::decode(index, detail::cuda::extent(grid, axis, 0), detail::cuda::extent(grid, axis, 1), x, y, z);
            const detail::cuda::Vector start                    = detail::cuda::face_position(axis, x, y, z, grid);
            detail::cuda::VelocityBoundaryData tangent_boundary = boundary;
            for (float& value : tangent_boundary.values) value = 0.0F;
            const detail::cuda::Vector value0   = sample_velocity_value(velocity, start, grid, boundary);
            const detail::cuda::Vector tangent0 = sample_velocity_value(velocity_tangent, start, grid, tangent_boundary);
            const detail::cuda::Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const detail::cuda::Vector midpoint_tangent{-0.5F * grid.time_step * tangent0.x, -0.5F * grid.time_step * tangent0.y, -0.5F * grid.time_step * tangent0.z};
            Sample velocity_samples[3]{sample_face(velocity.x, 0, midpoint, grid, boundary), sample_face(velocity.y, 1, midpoint, grid, boundary), sample_face(velocity.z, 2, midpoint, grid, boundary)};
            const detail::cuda::Vector sampled_tangent = sample_velocity_value(velocity_tangent, midpoint, grid, tangent_boundary);
            detail::cuda::Vector value1_tangent{
                sampled_tangent.x + velocity_samples[0].gradient.x * midpoint_tangent.x + velocity_samples[0].gradient.y * midpoint_tangent.y + velocity_samples[0].gradient.z * midpoint_tangent.z,
                sampled_tangent.y + velocity_samples[1].gradient.x * midpoint_tangent.x + velocity_samples[1].gradient.y * midpoint_tangent.y + velocity_samples[1].gradient.z * midpoint_tangent.z,
                sampled_tangent.z + velocity_samples[2].gradient.x * midpoint_tangent.x + velocity_samples[2].gradient.y * midpoint_tangent.y + velocity_samples[2].gradient.z * midpoint_tangent.z,
            };
            const Trace trace = trace_rk2(start, velocity, collider_ids, grid, boundary);
            const detail::cuda::Vector position_tangent{-trace.derivative.x * grid.time_step * value1_tangent.x, -trace.derivative.y * grid.time_step * value1_tangent.y, -trace.derivative.z * grid.time_step * value1_tangent.z};
            const Sample source_sample = sample_face(detail::cuda::component(velocity, axis), axis, trace.position, grid, boundary);
            output_tangent[index]      = sample_face(detail::cuda::component(velocity_tangent, axis), axis, trace.position, grid, tangent_boundary).value + source_sample.gradient.x * position_tangent.x + source_sample.gradient.y * position_tangent.y + source_sample.gradient.z * position_tangent.z;
        }

        __global__ void advect_velocity_vjp_kernel(const detail::cuda::Grid grid, const int axis, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::VelocityBoundaryData boundary, const double* output_adjoint, const detail::cuda::StaggeredVectorView<double> velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::face_count(grid, axis)) return;
            int x, y, z;
            detail::cuda::decode(index, detail::cuda::extent(grid, axis, 0), detail::cuda::extent(grid, axis, 1), x, y, z);
            const detail::cuda::Vector start  = detail::cuda::face_position(axis, x, y, z, grid);
            const detail::cuda::Vector value0 = sample_velocity_value(velocity, start, grid, boundary);
            const detail::cuda::Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Trace trace          = trace_rk2(start, velocity, collider_ids, grid, boundary);
            const Sample source_sample = sample_face(detail::cuda::component(velocity, axis), axis, trace.position, grid, boundary);
            scatter_face(detail::cuda::component(velocity_adjoint, axis), axis, trace.position, output_adjoint[index], grid, boundary);
            const double value1_adjoint_x = -static_cast<double>(trace.derivative.x * grid.time_step * source_sample.gradient.x) * output_adjoint[index];
            const double value1_adjoint_y = -static_cast<double>(trace.derivative.y * grid.time_step * source_sample.gradient.y) * output_adjoint[index];
            const double value1_adjoint_z = -static_cast<double>(trace.derivative.z * grid.time_step * source_sample.gradient.z) * output_adjoint[index];
            Sample midpoint_samples[3]{sample_face(velocity.x, 0, midpoint, grid, boundary), sample_face(velocity.y, 1, midpoint, grid, boundary), sample_face(velocity.z, 2, midpoint, grid, boundary)};
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

        __global__ void advect_scalar_forward_kernel(const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const float* collider_value, const float* source, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::ScalarBoundaryData scalar_boundary, const detail::cuda::VelocityBoundaryData velocity_boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::cell_count(grid)) return;
            if (collider_ids[index] != 0u) {
                output[index] = collider_value[index];
                return;
            }
            int x, y, z;
            detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
            const Trace trace = trace_rk2(detail::cuda::cell_position(x, y, z, grid), velocity, collider_ids, grid, velocity_boundary);
            output[index]     = sample_scalar(source, trace.position, grid, scalar_boundary).value;
        }

        __global__ void advect_scalar_jvp_kernel(const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const float* source, const float* source_tangent, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::StaggeredVectorView<const float> velocity_tangent, const detail::cuda::ScalarBoundaryData scalar_boundary, const detail::cuda::VelocityBoundaryData velocity_boundary, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::cell_count(grid)) return;
            if (collider_ids[index] != 0u) {
                output_tangent[index] = 0.0F;
                return;
            }
            int x, y, z;
            detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
            const detail::cuda::Vector start                         = detail::cuda::cell_position(x, y, z, grid);
            detail::cuda::ScalarBoundaryData scalar_tangent_boundary = scalar_boundary;
            for (float& value : scalar_tangent_boundary.values) value = 0.0F;
            detail::cuda::VelocityBoundaryData velocity_tangent_boundary = velocity_boundary;
            for (float& value : velocity_tangent_boundary.values) value = 0.0F;
            const detail::cuda::Vector value0   = sample_velocity_value(velocity, start, grid, velocity_boundary);
            const detail::cuda::Vector tangent0 = sample_velocity_value(velocity_tangent, start, grid, velocity_tangent_boundary);
            const detail::cuda::Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const detail::cuda::Vector midpoint_tangent{-0.5F * grid.time_step * tangent0.x, -0.5F * grid.time_step * tangent0.y, -0.5F * grid.time_step * tangent0.z};
            Sample velocity_samples[3]{sample_face(velocity.x, 0, midpoint, grid, velocity_boundary), sample_face(velocity.y, 1, midpoint, grid, velocity_boundary), sample_face(velocity.z, 2, midpoint, grid, velocity_boundary)};
            const detail::cuda::Vector sampled_tangent = sample_velocity_value(velocity_tangent, midpoint, grid, velocity_tangent_boundary);
            const detail::cuda::Vector value1_tangent{
                sampled_tangent.x + velocity_samples[0].gradient.x * midpoint_tangent.x + velocity_samples[0].gradient.y * midpoint_tangent.y + velocity_samples[0].gradient.z * midpoint_tangent.z,
                sampled_tangent.y + velocity_samples[1].gradient.x * midpoint_tangent.x + velocity_samples[1].gradient.y * midpoint_tangent.y + velocity_samples[1].gradient.z * midpoint_tangent.z,
                sampled_tangent.z + velocity_samples[2].gradient.x * midpoint_tangent.x + velocity_samples[2].gradient.y * midpoint_tangent.y + velocity_samples[2].gradient.z * midpoint_tangent.z,
            };
            const Trace trace = trace_rk2(start, velocity, collider_ids, grid, velocity_boundary);
            const detail::cuda::Vector position_tangent{-trace.derivative.x * grid.time_step * value1_tangent.x, -trace.derivative.y * grid.time_step * value1_tangent.y, -trace.derivative.z * grid.time_step * value1_tangent.z};
            const Sample source_sample = sample_scalar(source, trace.position, grid, scalar_boundary);
            output_tangent[index]      = sample_scalar(source_tangent, trace.position, grid, scalar_tangent_boundary).value + source_sample.gradient.x * position_tangent.x + source_sample.gradient.y * position_tangent.y + source_sample.gradient.z * position_tangent.z;
        }

        __global__ void advect_scalar_vjp_kernel(const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const float* source, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::ScalarBoundaryData scalar_boundary, const detail::cuda::VelocityBoundaryData velocity_boundary, const double* output_adjoint, double* source_adjoint, const detail::cuda::StaggeredVectorView<double> velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::cell_count(grid) || collider_ids[index] != 0u) return;
            int x, y, z;
            detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
            const detail::cuda::Vector start  = detail::cuda::cell_position(x, y, z, grid);
            const detail::cuda::Vector value0 = sample_velocity_value(velocity, start, grid, velocity_boundary);
            const detail::cuda::Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Trace trace          = trace_rk2(start, velocity, collider_ids, grid, velocity_boundary);
            const Sample source_sample = sample_scalar(source, trace.position, grid, scalar_boundary);
            scatter_scalar(source_adjoint, trace.position, output_adjoint[index], grid, scalar_boundary);
            const double value1_adjoint_x = -static_cast<double>(trace.derivative.x * grid.time_step * source_sample.gradient.x) * output_adjoint[index];
            const double value1_adjoint_y = -static_cast<double>(trace.derivative.y * grid.time_step * source_sample.gradient.y) * output_adjoint[index];
            const double value1_adjoint_z = -static_cast<double>(trace.derivative.z * grid.time_step * source_sample.gradient.z) * output_adjoint[index];
            Sample midpoint_samples[3]{sample_face(velocity.x, 0, midpoint, grid, velocity_boundary), sample_face(velocity.y, 1, midpoint, grid, velocity_boundary), sample_face(velocity.z, 2, midpoint, grid, velocity_boundary)};
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

    void advect_velocity_forward(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::VelocityBoundaryData boundary, const detail::cuda::StaggeredVectorView<float> output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::face_count(grid, axis)), advect_velocity_forward_kernel, grid, axis, collider_ids, velocity, boundary, detail::cuda::component(output, axis));
    }

    void advect_velocity_jvp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::StaggeredVectorView<const float> velocity_tangent, const detail::cuda::VelocityBoundaryData boundary, const detail::cuda::StaggeredVectorView<float> output_tangent) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::face_count(grid, axis)), advect_velocity_jvp_kernel, grid, axis, collider_ids, velocity, velocity_tangent, boundary, detail::cuda::component(output_tangent, axis));
    }

    void advect_velocity_vjp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::VelocityBoundaryData boundary, const detail::cuda::StaggeredVectorView<const double> output_adjoint, const detail::cuda::StaggeredVectorView<double> velocity_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::face_count(grid, axis)), advect_velocity_vjp_kernel, grid, axis, collider_ids, velocity, boundary, detail::cuda::component(output_adjoint, axis), velocity_adjoint);
    }

    void advect_scalar_forward(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const detail::cuda::ScalarView<const float> collider_value, const detail::cuda::ScalarView<const float> source, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::ScalarBoundaryData scalar_boundary, const detail::cuda::VelocityBoundaryData velocity_boundary, const detail::cuda::ScalarView<float> output) {
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::cell_count(grid)), advect_scalar_forward_kernel, grid, collider_ids, collider_value.values, source.values, velocity, scalar_boundary, velocity_boundary, output.values);
    }

    void advect_scalar_jvp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const detail::cuda::ScalarView<const float> source, const detail::cuda::ScalarView<const float> source_tangent, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::StaggeredVectorView<const float> velocity_tangent, const detail::cuda::ScalarBoundaryData scalar_boundary, const detail::cuda::VelocityBoundaryData velocity_boundary, const detail::cuda::ScalarView<float> output_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::cell_count(grid)), advect_scalar_jvp_kernel, grid, collider_ids, source.values, source_tangent.values, velocity, velocity_tangent, scalar_boundary, velocity_boundary, output_tangent.values);
    }

    void advect_scalar_vjp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t* collider_ids, const detail::cuda::ScalarView<const float> source, const detail::cuda::StaggeredVectorView<const float> velocity, const detail::cuda::ScalarBoundaryData scalar_boundary, const detail::cuda::VelocityBoundaryData velocity_boundary, const detail::cuda::ScalarView<const double> output_adjoint, const detail::cuda::ScalarView<double> source_adjoint, const detail::cuda::StaggeredVectorView<double> velocity_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::cell_count(grid)), advect_scalar_vjp_kernel, grid, collider_ids, source.values, velocity, scalar_boundary, velocity_boundary, output_adjoint.values, source_adjoint.values, velocity_adjoint);
    }
} // namespace physica::fluids::gas::operators::cuda_backend
