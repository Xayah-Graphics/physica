#include "advection-kernels.h"
#include "advection-sampling.cuh"
#include <cuda/launch>

namespace physica::fluids::gas::operators::kernels {
    namespace {
        __global__ void advect_velocity_forward_kernel(const device::Discretization grid, const int axis, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const device::VelocityBoundary boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            const Trace trace = trace_rk2(fluids::grid::device::face_position(grid.grid, axis, x, y, z), velocity, collider_ids, grid, boundary);
            output[index]     = sample_face(fluids::grid::device::component(velocity, axis), axis, trace.position, grid, boundary).value;
        }

        __global__ void advect_velocity_jvp_kernel(const device::Discretization grid, const int axis, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const simulation::VectorView<const float> velocity_tangent, const device::VelocityBoundary boundary, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            const Vector3<float> start                    = fluids::grid::device::face_position(grid.grid, axis, x, y, z);
            device::VelocityBoundary tangent_boundary = boundary;
            for (device::VelocityBoundaryFace& face : tangent_boundary.faces) face.value = {};
            const Vector3<float> value0   = sample_velocity_value(velocity, start, grid, boundary);
            const Vector3<float> tangent0 = sample_velocity_value(velocity_tangent, start, grid, tangent_boundary);
            const Vector3<float> midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Vector3<float> midpoint_tangent{-0.5F * grid.time_step * tangent0.x, -0.5F * grid.time_step * tangent0.y, -0.5F * grid.time_step * tangent0.z};
            Sample velocity_samples[3]{sample_face(velocity.x, 0, midpoint, grid, boundary), sample_face(velocity.y, 1, midpoint, grid, boundary), sample_face(velocity.z, 2, midpoint, grid, boundary)};
            const Vector3<float> sampled_tangent = sample_velocity_value(velocity_tangent, midpoint, grid, tangent_boundary);
            Vector3<float> value1_tangent{
                sampled_tangent.x + velocity_samples[0].gradient.x * midpoint_tangent.x + velocity_samples[0].gradient.y * midpoint_tangent.y + velocity_samples[0].gradient.z * midpoint_tangent.z,
                sampled_tangent.y + velocity_samples[1].gradient.x * midpoint_tangent.x + velocity_samples[1].gradient.y * midpoint_tangent.y + velocity_samples[1].gradient.z * midpoint_tangent.z,
                sampled_tangent.z + velocity_samples[2].gradient.x * midpoint_tangent.x + velocity_samples[2].gradient.y * midpoint_tangent.y + velocity_samples[2].gradient.z * midpoint_tangent.z,
            };
            const Trace trace = trace_rk2(start, velocity, collider_ids, grid, boundary);
            const Vector3<float> position_tangent{-trace.derivative.x * grid.time_step * value1_tangent.x, -trace.derivative.y * grid.time_step * value1_tangent.y, -trace.derivative.z * grid.time_step * value1_tangent.z};
            const Sample source_sample = sample_face(fluids::grid::device::component(velocity, axis), axis, trace.position, grid, boundary);
            output_tangent[index]      = sample_face(fluids::grid::device::component(velocity_tangent, axis), axis, trace.position, grid, tangent_boundary).value + source_sample.gradient.x * position_tangent.x + source_sample.gradient.y * position_tangent.y + source_sample.gradient.z * position_tangent.z;
        }

        __global__ void advect_velocity_vjp_kernel(const device::Discretization grid, const int axis, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const device::VelocityBoundary boundary, const double* output_adjoint, const simulation::VectorView<double> velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::face_count(grid.grid, axis)) return;
            int x, y, z;
            fluids::grid::device::decode(index, fluids::grid::device::extent(grid.grid, axis, 0), fluids::grid::device::extent(grid.grid, axis, 1), x, y, z);
            const Vector3<float> start  = fluids::grid::device::face_position(grid.grid, axis, x, y, z);
            const Vector3<float> value0 = sample_velocity_value(velocity, start, grid, boundary);
            const Vector3<float> midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Trace trace          = trace_rk2(start, velocity, collider_ids, grid, boundary);
            const Sample source_sample = sample_face(fluids::grid::device::component(velocity, axis), axis, trace.position, grid, boundary);
            scatter_face(fluids::grid::device::component(velocity_adjoint, axis), axis, trace.position, output_adjoint[index], grid, boundary);
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

        __global__ void advect_scalar_forward_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* collider_value, const float* source, const simulation::VectorView<const float> velocity, const device::ScalarBoundary scalar_boundary, const device::VelocityBoundary velocity_boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            if (collider_ids[index] != 0u) {
                output[index] = collider_value[index];
                return;
            }
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            const Trace trace = trace_rk2(fluids::grid::device::cell_position(grid.grid, x, y, z), velocity, collider_ids, grid, velocity_boundary);
            output[index]     = sample_scalar(source, trace.position, grid, scalar_boundary).value;
        }

        __global__ void advect_scalar_jvp_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* source, const float* source_tangent, const simulation::VectorView<const float> velocity, const simulation::VectorView<const float> velocity_tangent, const device::ScalarBoundary scalar_boundary, const device::VelocityBoundary velocity_boundary, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            if (collider_ids[index] != 0u) {
                output_tangent[index] = 0.0F;
                return;
            }
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            const Vector3<float> start                         = fluids::grid::device::cell_position(grid.grid, x, y, z);
            device::ScalarBoundary scalar_tangent_boundary = scalar_boundary;
            for (device::ScalarBoundaryFace& face : scalar_tangent_boundary.faces) face.value = 0.0F;
            device::VelocityBoundary velocity_tangent_boundary = velocity_boundary;
            for (device::VelocityBoundaryFace& face : velocity_tangent_boundary.faces) face.value = {};
            const Vector3<float> value0   = sample_velocity_value(velocity, start, grid, velocity_boundary);
            const Vector3<float> tangent0 = sample_velocity_value(velocity_tangent, start, grid, velocity_tangent_boundary);
            const Vector3<float> midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Vector3<float> midpoint_tangent{-0.5F * grid.time_step * tangent0.x, -0.5F * grid.time_step * tangent0.y, -0.5F * grid.time_step * tangent0.z};
            Sample velocity_samples[3]{sample_face(velocity.x, 0, midpoint, grid, velocity_boundary), sample_face(velocity.y, 1, midpoint, grid, velocity_boundary), sample_face(velocity.z, 2, midpoint, grid, velocity_boundary)};
            const Vector3<float> sampled_tangent = sample_velocity_value(velocity_tangent, midpoint, grid, velocity_tangent_boundary);
            const Vector3<float> value1_tangent{
                sampled_tangent.x + velocity_samples[0].gradient.x * midpoint_tangent.x + velocity_samples[0].gradient.y * midpoint_tangent.y + velocity_samples[0].gradient.z * midpoint_tangent.z,
                sampled_tangent.y + velocity_samples[1].gradient.x * midpoint_tangent.x + velocity_samples[1].gradient.y * midpoint_tangent.y + velocity_samples[1].gradient.z * midpoint_tangent.z,
                sampled_tangent.z + velocity_samples[2].gradient.x * midpoint_tangent.x + velocity_samples[2].gradient.y * midpoint_tangent.y + velocity_samples[2].gradient.z * midpoint_tangent.z,
            };
            const Trace trace = trace_rk2(start, velocity, collider_ids, grid, velocity_boundary);
            const Vector3<float> position_tangent{-trace.derivative.x * grid.time_step * value1_tangent.x, -trace.derivative.y * grid.time_step * value1_tangent.y, -trace.derivative.z * grid.time_step * value1_tangent.z};
            const Sample source_sample = sample_scalar(source, trace.position, grid, scalar_boundary);
            output_tangent[index]      = sample_scalar(source_tangent, trace.position, grid, scalar_tangent_boundary).value + source_sample.gradient.x * position_tangent.x + source_sample.gradient.y * position_tangent.y + source_sample.gradient.z * position_tangent.z;
        }

        __global__ void advect_scalar_vjp_kernel(const device::Discretization grid, const std::uint32_t* collider_ids, const float* source, const simulation::VectorView<const float> velocity, const device::ScalarBoundary scalar_boundary, const device::VelocityBoundary velocity_boundary, const double* output_adjoint, double* source_adjoint, const simulation::VectorView<double> velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid) || collider_ids[index] != 0u) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            const Vector3<float> start  = fluids::grid::device::cell_position(grid.grid, x, y, z);
            const Vector3<float> value0 = sample_velocity_value(velocity, start, grid, velocity_boundary);
            const Vector3<float> midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
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

    void advect_velocity_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const device::VelocityBoundary boundary, const simulation::VectorView<float> output) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::face_count(grid.grid, axis)), advect_velocity_forward_kernel, grid, axis, collider_ids, velocity, boundary, fluids::grid::device::component(output, axis));
    }

    void advect_velocity_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const simulation::VectorView<const float> velocity_tangent, const device::VelocityBoundary boundary, const simulation::VectorView<float> output_tangent) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::face_count(grid.grid, axis)), advect_velocity_jvp_kernel, grid, axis, collider_ids, velocity, velocity_tangent, boundary, fluids::grid::device::component(output_tangent, axis));
    }

    void advect_velocity_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::VectorView<const float> velocity, const device::VelocityBoundary boundary, const simulation::VectorView<const double> output_adjoint, const simulation::VectorView<double> velocity_adjoint) {
        for (int axis = 0; axis < 3; ++axis) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::face_count(grid.grid, axis)), advect_velocity_vjp_kernel, grid, axis, collider_ids, velocity, boundary, fluids::grid::device::component(output_adjoint, axis), velocity_adjoint);
    }

    void advect_scalar_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::ScalarView<const float> collider_value, const simulation::ScalarView<const float> source, const simulation::VectorView<const float> velocity, const device::ScalarBoundary scalar_boundary, const device::VelocityBoundary velocity_boundary, const simulation::ScalarView<float> output) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), advect_scalar_forward_kernel, grid, collider_ids, collider_value.values, source.values, velocity, scalar_boundary, velocity_boundary, output.values);
    }

    void advect_scalar_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::ScalarView<const float> source, const simulation::ScalarView<const float> source_tangent, const simulation::VectorView<const float> velocity, const simulation::VectorView<const float> velocity_tangent, const device::ScalarBoundary scalar_boundary, const device::VelocityBoundary velocity_boundary, const simulation::ScalarView<float> output_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), advect_scalar_jvp_kernel, grid, collider_ids, source.values, source_tangent.values, velocity, velocity_tangent, scalar_boundary, velocity_boundary, output_tangent.values);
    }

    void advect_scalar_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t* collider_ids, const simulation::ScalarView<const float> source, const simulation::VectorView<const float> velocity, const device::ScalarBoundary scalar_boundary, const device::VelocityBoundary velocity_boundary, const simulation::ScalarView<const double> output_adjoint, const simulation::ScalarView<double> source_adjoint, const simulation::VectorView<double> velocity_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), advect_scalar_vjp_kernel, grid, collider_ids, source.values, velocity, scalar_boundary, velocity_boundary, output_adjoint.values, source_adjoint.values, velocity_adjoint);
    }
} // namespace physica::fluids::gas::operators::kernels
