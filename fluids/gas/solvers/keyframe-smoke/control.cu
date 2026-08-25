#include "../../detail/cuda/device.cuh"
#include "control-kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::keyframe_smoke::cuda_backend {
    namespace {
        __device__ detail::cuda::Vector cross(const detail::cuda::Vector first, const detail::cuda::Vector second) {
            return {first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x};
        }

        __global__ void control_forward_kernel(const detail::cuda::Grid grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const detail::cuda::CenteredVectorView<float> output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::cell_count(grid)) return;
            int x, y, z;
            detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
            const detail::cuda::Vector position = detail::cuda::cell_position(x, y, z, grid);
            detail::cuda::Vector force{};
            for (std::uint32_t primitive = 0u; primitive < wind_count; ++primitive) {
                const WindData wind = winds[primitive];
                if (step < wind.begin_step || step >= wind.end_step) continue;
                const std::uint32_t offset = wind.parameter_offset;
                const detail::cuda::Vector radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const float gaussian = expf(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * wind.width * wind.width));
                force.x += gaussian * static_cast<float>(parameters[offset + 3u]);
                force.y += gaussian * static_cast<float>(parameters[offset + 4u]);
                force.z += gaussian * static_cast<float>(parameters[offset + 5u]);
            }
            for (std::uint32_t primitive = 0u; primitive < vortex_count; ++primitive) {
                const VortexData vortex = vortices[primitive];
                if (step < vortex.begin_step || step >= vortex.end_step) continue;
                const std::uint32_t offset = vortex.parameter_offset;
                const detail::cuda::Vector radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const detail::cuda::Vector tangent = cross(vortex.axis, radial);
                const float gaussian               = expf(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * vortex.width * vortex.width));
                const float scale                  = gaussian * static_cast<float>(parameters[offset + 3u]);
                force.x += scale * tangent.x;
                force.y += scale * tangent.y;
                force.z += scale * tangent.z;
            }
            output.x[index] = force.x;
            output.y[index] = force.y;
            output.z[index] = force.z;
        }

        __global__ void control_jvp_kernel(const detail::cuda::Grid grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const double* direction, const detail::cuda::CenteredVectorView<float> output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= detail::cuda::cell_count(grid)) return;
            int x, y, z;
            detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
            const detail::cuda::Vector position = detail::cuda::cell_position(x, y, z, grid);
            detail::cuda::Vector force_tangent{};
            for (std::uint32_t primitive = 0u; primitive < wind_count; ++primitive) {
                const WindData wind = winds[primitive];
                if (step < wind.begin_step || step >= wind.end_step) continue;
                const std::uint32_t offset = wind.parameter_offset;
                const detail::cuda::Vector radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const detail::cuda::Vector center_tangent{static_cast<float>(direction[offset]), static_cast<float>(direction[offset + 1u]), static_cast<float>(direction[offset + 2u])};
                const detail::cuda::Vector vector{static_cast<float>(parameters[offset + 3u]), static_cast<float>(parameters[offset + 4u]), static_cast<float>(parameters[offset + 5u])};
                const detail::cuda::Vector vector_tangent{static_cast<float>(direction[offset + 3u]), static_cast<float>(direction[offset + 4u]), static_cast<float>(direction[offset + 5u])};
                const float gaussian         = expf(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * wind.width * wind.width));
                const float gaussian_tangent = gaussian * (radial.x * center_tangent.x + radial.y * center_tangent.y + radial.z * center_tangent.z) / (wind.width * wind.width);
                force_tangent.x += gaussian * vector_tangent.x + gaussian_tangent * vector.x;
                force_tangent.y += gaussian * vector_tangent.y + gaussian_tangent * vector.y;
                force_tangent.z += gaussian * vector_tangent.z + gaussian_tangent * vector.z;
            }
            for (std::uint32_t primitive = 0u; primitive < vortex_count; ++primitive) {
                const VortexData vortex = vortices[primitive];
                if (step < vortex.begin_step || step >= vortex.end_step) continue;
                const std::uint32_t offset = vortex.parameter_offset;
                const detail::cuda::Vector radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const detail::cuda::Vector center_tangent{static_cast<float>(direction[offset]), static_cast<float>(direction[offset + 1u]), static_cast<float>(direction[offset + 2u])};
                const detail::cuda::Vector tangent         = cross(vortex.axis, radial);
                const detail::cuda::Vector tangent_tangent = cross(vortex.axis, {-center_tangent.x, -center_tangent.y, -center_tangent.z});
                const float gaussian                       = expf(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * vortex.width * vortex.width));
                const float gaussian_tangent               = gaussian * (radial.x * center_tangent.x + radial.y * center_tangent.y + radial.z * center_tangent.z) / (vortex.width * vortex.width);
                const float strength                       = static_cast<float>(parameters[offset + 3u]);
                const float strength_tangent               = static_cast<float>(direction[offset + 3u]);
                force_tangent.x += strength_tangent * gaussian * tangent.x + strength * gaussian_tangent * tangent.x + strength * gaussian * tangent_tangent.x;
                force_tangent.y += strength_tangent * gaussian * tangent.y + strength * gaussian_tangent * tangent.y + strength * gaussian * tangent_tangent.y;
                force_tangent.z += strength_tangent * gaussian * tangent.z + strength * gaussian_tangent * tangent.z + strength * gaussian * tangent_tangent.z;
            }
            output.x[index] = force_tangent.x;
            output.y[index] = force_tangent.y;
            output.z[index] = force_tangent.z;
        }

        __global__ void wind_vjp_kernel(const detail::cuda::Grid grid, const std::uint32_t step, const WindData* winds, const double* parameters, const detail::cuda::CenteredVectorView<const double> output_adjoint, double* gradient) {
            const WindData wind = winds[blockIdx.x];
            if (step < wind.begin_step || step >= wind.end_step) return;
            __shared__ double reductions[6][detail::cuda::block_size];
            double local[6]{};
            const std::uint32_t offset = wind.parameter_offset;
            const detail::cuda::Vector vector{static_cast<float>(parameters[offset + 3u]), static_cast<float>(parameters[offset + 4u]), static_cast<float>(parameters[offset + 5u])};
            for (std::uint64_t index = threadIdx.x; index < detail::cuda::cell_count(grid); index += blockDim.x) {
                int x, y, z;
                detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
                const detail::cuda::Vector position = detail::cuda::cell_position(x, y, z, grid);
                const detail::cuda::Vector radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const double gaussian     = exp(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * wind.width * wind.width));
                const double dot          = output_adjoint.x[index] * vector.x + output_adjoint.y[index] * vector.y + output_adjoint.z[index] * vector.z;
                const double center_scale = gaussian * dot / (wind.width * wind.width);
                local[0] += center_scale * radial.x;
                local[1] += center_scale * radial.y;
                local[2] += center_scale * radial.z;
                local[3] += gaussian * output_adjoint.x[index];
                local[4] += gaussian * output_adjoint.y[index];
                local[5] += gaussian * output_adjoint.z[index];
            }
            for (int parameter = 0; parameter < 6; ++parameter) reductions[parameter][threadIdx.x] = local[parameter];
            __syncthreads();
            for (unsigned reduction = blockDim.x / 2u; reduction > 0u; reduction /= 2u) {
                if (threadIdx.x < reduction)
                    for (int parameter = 0; parameter < 6; ++parameter) reductions[parameter][threadIdx.x] += reductions[parameter][threadIdx.x + reduction];
                __syncthreads();
            }
            if (threadIdx.x == 0u)
                for (int parameter = 0; parameter < 6; ++parameter) gradient[offset + parameter] += reductions[parameter][0];
        }

        __global__ void vortex_vjp_kernel(const detail::cuda::Grid grid, const std::uint32_t step, const VortexData* vortices, const double* parameters, const detail::cuda::CenteredVectorView<const double> output_adjoint, double* gradient) {
            const VortexData vortex = vortices[blockIdx.x];
            if (step < vortex.begin_step || step >= vortex.end_step) return;
            __shared__ double reductions[4][detail::cuda::block_size];
            double local[4]{};
            const std::uint32_t offset = vortex.parameter_offset;
            const float strength       = static_cast<float>(parameters[offset + 3u]);
            for (std::uint64_t index = threadIdx.x; index < detail::cuda::cell_count(grid); index += blockDim.x) {
                int x, y, z;
                detail::cuda::decode(index, grid.nx, grid.ny, x, y, z);
                const detail::cuda::Vector position = detail::cuda::cell_position(x, y, z, grid);
                const detail::cuda::Vector radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const detail::cuda::Vector tangent = cross(vortex.axis, radial);
                const double gaussian              = exp(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * vortex.width * vortex.width));
                const detail::cuda::Vector tangent_adjoint{static_cast<float>(strength * gaussian * output_adjoint.x[index]), static_cast<float>(strength * gaussian * output_adjoint.y[index]), static_cast<float>(strength * gaussian * output_adjoint.z[index])};
                const detail::cuda::Vector center_from_tangent = cross(vortex.axis, tangent_adjoint);
                const double dot                               = output_adjoint.x[index] * tangent.x + output_adjoint.y[index] * tangent.y + output_adjoint.z[index] * tangent.z;
                const double center_scale                      = strength * gaussian * dot / (vortex.width * vortex.width);
                local[0] += center_scale * radial.x + center_from_tangent.x;
                local[1] += center_scale * radial.y + center_from_tangent.y;
                local[2] += center_scale * radial.z + center_from_tangent.z;
                local[3] += gaussian * dot;
            }
            for (int parameter = 0; parameter < 4; ++parameter) reductions[parameter][threadIdx.x] = local[parameter];
            __syncthreads();
            for (unsigned reduction = blockDim.x / 2u; reduction > 0u; reduction /= 2u) {
                if (threadIdx.x < reduction)
                    for (int parameter = 0; parameter < 4; ++parameter) reductions[parameter][threadIdx.x] += reductions[parameter][threadIdx.x + reduction];
                __syncthreads();
            }
            if (threadIdx.x == 0u)
                for (int parameter = 0; parameter < 4; ++parameter) gradient[offset + parameter] += reductions[parameter][0];
        }
    } // namespace

    void control_forward(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const detail::cuda::CenteredVectorView<float> output) {
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::cell_count(grid)), control_forward_kernel, grid, step, winds, wind_count, vortices, vortex_count, parameters, output);
    }

    void control_jvp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const double* direction, const detail::cuda::CenteredVectorView<float> output_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(detail::cuda::cell_count(grid)), control_jvp_kernel, grid, step, winds, wind_count, vortices, vortex_count, parameters, direction, output_tangent);
    }

    void control_vjp(const ::cuda::stream_ref stream, const detail::cuda::Grid grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const detail::cuda::CenteredVectorView<const double> output_adjoint, double* gradient) {
        if (wind_count != 0u) ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(wind_count * detail::cuda::block_size), wind_vjp_kernel, grid, step, winds, parameters, output_adjoint, gradient);
        if (vortex_count != 0u) ::cuda::launch(stream, ::cuda::distribute<detail::cuda::block_size>(vortex_count * detail::cuda::block_size), vortex_vjp_kernel, grid, step, vortices, parameters, output_adjoint, gradient);
    }
} // namespace physica::fluids::gas::keyframe_smoke::cuda_backend
