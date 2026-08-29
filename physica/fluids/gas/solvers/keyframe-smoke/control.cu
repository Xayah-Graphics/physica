#include "control-kernels.h"
#include <cuda/launch>
#include <cuda/std/cmath>

namespace physica::fluids::gas::keyframe_smoke::kernels {
    namespace {
        __global__ void control_forward_kernel(const device::Discretization grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const field::VectorView<float> output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            const Vector3<float> position = fluids::grid::device::cell_position(grid.grid, x, y, z);
            Vector3<float> force{};
            for (std::uint32_t primitive = 0u; primitive < wind_count; ++primitive) {
                const WindData wind = winds[primitive];
                if (step < wind.begin_step || step >= wind.end_step) continue;
                const std::uint32_t offset = wind.parameter_offset;
                const Vector3<float> radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const float gaussian = ::cuda::std::exp(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * wind.width * wind.width));
                force.x += gaussian * static_cast<float>(parameters[offset + 3u]);
                force.y += gaussian * static_cast<float>(parameters[offset + 4u]);
                force.z += gaussian * static_cast<float>(parameters[offset + 5u]);
            }
            for (std::uint32_t primitive = 0u; primitive < vortex_count; ++primitive) {
                const VortexData vortex = vortices[primitive];
                if (step < vortex.begin_step || step >= vortex.end_step) continue;
                const std::uint32_t offset = vortex.parameter_offset;
                const Vector3<float> radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const Vector3<float> tangent = cross(vortex.axis, radial);
                const float gaussian               = ::cuda::std::exp(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * vortex.width * vortex.width));
                const float scale                  = gaussian * static_cast<float>(parameters[offset + 3u]);
                force.x += scale * tangent.x;
                force.y += scale * tangent.y;
                force.z += scale * tangent.z;
            }
            output.x[index] = force.x;
            output.y[index] = force.y;
            output.z[index] = force.z;
        }

        __global__ void control_jvp_kernel(const device::Discretization grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const double* direction, const field::VectorView<float> output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            const Vector3<float> position = fluids::grid::device::cell_position(grid.grid, x, y, z);
            Vector3<float> force_tangent{};
            for (std::uint32_t primitive = 0u; primitive < wind_count; ++primitive) {
                const WindData wind = winds[primitive];
                if (step < wind.begin_step || step >= wind.end_step) continue;
                const std::uint32_t offset = wind.parameter_offset;
                const Vector3<float> radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const Vector3<float> center_tangent{static_cast<float>(direction[offset]), static_cast<float>(direction[offset + 1u]), static_cast<float>(direction[offset + 2u])};
                const Vector3<float> vector{static_cast<float>(parameters[offset + 3u]), static_cast<float>(parameters[offset + 4u]), static_cast<float>(parameters[offset + 5u])};
                const Vector3<float> vector_tangent{static_cast<float>(direction[offset + 3u]), static_cast<float>(direction[offset + 4u]), static_cast<float>(direction[offset + 5u])};
                const float gaussian         = ::cuda::std::exp(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * wind.width * wind.width));
                const float gaussian_tangent = gaussian * (radial.x * center_tangent.x + radial.y * center_tangent.y + radial.z * center_tangent.z) / (wind.width * wind.width);
                force_tangent.x += gaussian * vector_tangent.x + gaussian_tangent * vector.x;
                force_tangent.y += gaussian * vector_tangent.y + gaussian_tangent * vector.y;
                force_tangent.z += gaussian * vector_tangent.z + gaussian_tangent * vector.z;
            }
            for (std::uint32_t primitive = 0u; primitive < vortex_count; ++primitive) {
                const VortexData vortex = vortices[primitive];
                if (step < vortex.begin_step || step >= vortex.end_step) continue;
                const std::uint32_t offset = vortex.parameter_offset;
                const Vector3<float> radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const Vector3<float> center_tangent{static_cast<float>(direction[offset]), static_cast<float>(direction[offset + 1u]), static_cast<float>(direction[offset + 2u])};
                const Vector3<float> tangent         = cross(vortex.axis, radial);
                const Vector3<float> tangent_tangent = cross(vortex.axis, -center_tangent);
                const float gaussian                       = ::cuda::std::exp(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * vortex.width * vortex.width));
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

        __global__ void wind_vjp_kernel(const device::Discretization grid, const std::uint32_t step, const WindData* winds, const double* parameters, const field::VectorView<const double> output_adjoint, double* gradient) {
            const WindData wind = winds[blockIdx.x];
            if (step < wind.begin_step || step >= wind.end_step) return;
            __shared__ double reductions[6][fluids::grid::device::block_size];
            double local[6]{};
            const std::uint32_t offset = wind.parameter_offset;
            const Vector3<float> vector{static_cast<float>(parameters[offset + 3u]), static_cast<float>(parameters[offset + 4u]), static_cast<float>(parameters[offset + 5u])};
            for (std::uint64_t index = threadIdx.x; index < fluids::grid::device::cell_count(grid.grid); index += blockDim.x) {
                int x, y, z;
                fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
                const Vector3<float> position = fluids::grid::device::cell_position(grid.grid, x, y, z);
                const Vector3<float> radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const double gaussian     = ::cuda::std::exp(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * wind.width * wind.width));
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

        __global__ void vortex_vjp_kernel(const device::Discretization grid, const std::uint32_t step, const VortexData* vortices, const double* parameters, const field::VectorView<const double> output_adjoint, double* gradient) {
            const VortexData vortex = vortices[blockIdx.x];
            if (step < vortex.begin_step || step >= vortex.end_step) return;
            __shared__ double reductions[4][fluids::grid::device::block_size];
            double local[4]{};
            const std::uint32_t offset = vortex.parameter_offset;
            const float strength       = static_cast<float>(parameters[offset + 3u]);
            for (std::uint64_t index = threadIdx.x; index < fluids::grid::device::cell_count(grid.grid); index += blockDim.x) {
                int x, y, z;
                fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
                const Vector3<float> position = fluids::grid::device::cell_position(grid.grid, x, y, z);
                const Vector3<float> radial{position.x - static_cast<float>(parameters[offset]), position.y - static_cast<float>(parameters[offset + 1u]), position.z - static_cast<float>(parameters[offset + 2u])};
                const Vector3<float> tangent = cross(vortex.axis, radial);
                const double gaussian              = ::cuda::std::exp(-(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z) / (2.0F * vortex.width * vortex.width));
                const Vector3<float> tangent_adjoint{static_cast<float>(strength * gaussian * output_adjoint.x[index]), static_cast<float>(strength * gaussian * output_adjoint.y[index]), static_cast<float>(strength * gaussian * output_adjoint.z[index])};
                const Vector3<float> center_from_tangent = cross(vortex.axis, tangent_adjoint);
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

    void control_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const field::VectorView<float> output) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), control_forward_kernel, grid, step, winds, wind_count, vortices, vortex_count, parameters, output);
    }

    void control_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const double* direction, const field::VectorView<float> output_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), control_jvp_kernel, grid, step, winds, wind_count, vortices, vortex_count, parameters, direction, output_tangent);
    }

    void control_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t step, const WindData* winds, const std::uint32_t wind_count, const VortexData* vortices, const std::uint32_t vortex_count, const double* parameters, const field::VectorView<const double> output_adjoint, double* gradient) {
        if (wind_count != 0u) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(wind_count * fluids::grid::device::block_size), wind_vjp_kernel, grid, step, winds, parameters, output_adjoint, gradient);
        if (vortex_count != 0u) ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(vortex_count * fluids::grid::device::block_size), vortex_vjp_kernel, grid, step, vortices, parameters, output_adjoint, gradient);
    }
} // namespace physica::fluids::gas::keyframe_smoke::kernels
