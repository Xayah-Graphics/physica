#include "control-kernels.h"
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>

namespace physica::fluids::gas::solvers::adjoint_control::kernels {
    namespace {
        __device__ float gaussian(const float x, const float y, const float z, const float sigma) {
            return ::cuda::std::exp(-0.5F * (x * x + y * y + z * z) / (sigma * sigma));
        }

        __global__ void apply_control_kernel(const device::Discretization grid, const std::uint32_t step, const ControlLatticeData lattice, const float sigma, const std::uint32_t step_count, const double* parameters, const simulation::VectorView<float> output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            if (step >= step_count) {
                output.x[index] = 0.0F;
                output.y[index] = 0.0F;
                output.z[index] = 0.0F;
                return;
            }
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            const float px              = (x + 0.5F) * grid.grid.cell_size;
            const float py              = (y + 0.5F) * grid.grid.cell_size;
            const float pz              = (z + 0.5F) * grid.grid.cell_size;
            const float support         = 3.0F * sigma;
            const int min_x             = ::cuda::std::max(0, static_cast<int>(::cuda::std::ceil((px - support) * lattice.x / (grid.grid.nx * grid.grid.cell_size) - 0.5F)));
            const int max_x             = ::cuda::std::min(static_cast<int>(lattice.x) - 1, static_cast<int>(::cuda::std::floor((px + support) * lattice.x / (grid.grid.nx * grid.grid.cell_size) - 0.5F)));
            const int min_y             = ::cuda::std::max(0, static_cast<int>(::cuda::std::ceil((py - support) * lattice.y / (grid.grid.ny * grid.grid.cell_size) - 0.5F)));
            const int max_y             = ::cuda::std::min(static_cast<int>(lattice.y) - 1, static_cast<int>(::cuda::std::floor((py + support) * lattice.y / (grid.grid.ny * grid.grid.cell_size) - 0.5F)));
            const int min_z             = ::cuda::std::max(0, static_cast<int>(::cuda::std::ceil((pz - support) * lattice.z / (grid.grid.nz * grid.grid.cell_size) - 0.5F)));
            const int max_z             = ::cuda::std::min(static_cast<int>(lattice.z) - 1, static_cast<int>(::cuda::std::floor((pz + support) * lattice.z / (grid.grid.nz * grid.grid.cell_size) - 0.5F)));
            double force_x              = 0.0;
            double force_y              = 0.0;
            double force_z              = 0.0;
            const std::uint64_t centers = static_cast<std::uint64_t>(lattice.x) * lattice.y * lattice.z;
            for (int cz = min_z; cz <= max_z; ++cz)
                for (int cy = min_y; cy <= max_y; ++cy)
                    for (int cx = min_x; cx <= max_x; ++cx) {
                        const float center_x       = (cx + 0.5F) * grid.grid.nx * grid.grid.cell_size / lattice.x;
                        const float center_y       = (cy + 0.5F) * grid.grid.ny * grid.grid.cell_size / lattice.y;
                        const float center_z       = (cz + 0.5F) * grid.grid.nz * grid.grid.cell_size / lattice.z;
                        const float weight         = gaussian(px - center_x, py - center_y, pz - center_z, sigma);
                        const std::uint64_t center = fluids::grid::device::index3(cx, cy, cz, lattice.x, lattice.y);
                        const std::uint64_t offset = (static_cast<std::uint64_t>(step) * centers + center) * 3u;
                        force_x += weight * parameters[offset];
                        force_y += weight * parameters[offset + 1u];
                        force_z += weight * parameters[offset + 2u];
                    }
            output.x[index] = static_cast<float>(force_x);
            output.y[index] = static_cast<float>(force_y);
            output.z[index] = static_cast<float>(force_z);
        }

        __global__ void control_vjp_kernel(const device::Discretization grid, const std::uint32_t step, const ControlLatticeData lattice, const float sigma, const std::uint32_t step_count, const simulation::VectorView<const double> output_adjoint, double* gradient) {
            const std::uint64_t center  = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::uint64_t centers = static_cast<std::uint64_t>(lattice.x) * lattice.y * lattice.z;
            if (center >= centers || step >= step_count) return;
            int cx, cy, cz;
            fluids::grid::device::decode(center, lattice.x, lattice.y, cx, cy, cz);
            const float center_x = (cx + 0.5F) * grid.grid.nx * grid.grid.cell_size / lattice.x;
            const float center_y = (cy + 0.5F) * grid.grid.ny * grid.grid.cell_size / lattice.y;
            const float center_z = (cz + 0.5F) * grid.grid.nz * grid.grid.cell_size / lattice.z;
            const float support  = 3.0F * sigma;
            const int min_x      = ::cuda::std::max(0, static_cast<int>(::cuda::std::ceil((center_x - support) / grid.grid.cell_size - 0.5F)));
            const int max_x      = ::cuda::std::min(static_cast<int>(grid.grid.nx) - 1, static_cast<int>(::cuda::std::floor((center_x + support) / grid.grid.cell_size - 0.5F)));
            const int min_y      = ::cuda::std::max(0, static_cast<int>(::cuda::std::ceil((center_y - support) / grid.grid.cell_size - 0.5F)));
            const int max_y      = ::cuda::std::min(static_cast<int>(grid.grid.ny) - 1, static_cast<int>(::cuda::std::floor((center_y + support) / grid.grid.cell_size - 0.5F)));
            const int min_z      = ::cuda::std::max(0, static_cast<int>(::cuda::std::ceil((center_z - support) / grid.grid.cell_size - 0.5F)));
            const int max_z      = ::cuda::std::min(static_cast<int>(grid.grid.nz) - 1, static_cast<int>(::cuda::std::floor((center_z + support) / grid.grid.cell_size - 0.5F)));
            double x_adjoint     = 0.0;
            double y_adjoint     = 0.0;
            double z_adjoint     = 0.0;
            for (int z = min_z; z <= max_z; ++z)
                for (int y = min_y; y <= max_y; ++y)
                    for (int x = min_x; x <= max_x; ++x) {
                        const float px           = (x + 0.5F) * grid.grid.cell_size;
                        const float py           = (y + 0.5F) * grid.grid.cell_size;
                        const float pz           = (z + 0.5F) * grid.grid.cell_size;
                        const double weight      = gaussian(px - center_x, py - center_y, pz - center_z, sigma);
                        const std::uint64_t cell = fluids::grid::device::index3(x, y, z, grid.grid.nx, grid.grid.ny);
                        x_adjoint += weight * output_adjoint.x[cell];
                        y_adjoint += weight * output_adjoint.y[cell];
                        z_adjoint += weight * output_adjoint.z[cell];
                    }
            const std::uint64_t offset = (static_cast<std::uint64_t>(step) * centers + center) * 3u;
            gradient[offset] += x_adjoint;
            gradient[offset + 1u] += y_adjoint;
            gradient[offset + 2u] += z_adjoint;
        }
    } // namespace

    void control_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t step, const ControlLatticeData lattice, const float sigma, const std::uint32_t step_count, const double* parameters, const simulation::VectorView<float> output) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), apply_control_kernel, grid, step, lattice, sigma, step_count, parameters, output);
    }

    void control_jvp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t step, const ControlLatticeData lattice, const float sigma, const std::uint32_t step_count, const double* direction, const simulation::VectorView<float> output) {
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), apply_control_kernel, grid, step, lattice, sigma, step_count, direction, output);
    }

    void control_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t step, const ControlLatticeData lattice, const float sigma, const std::uint32_t step_count, const simulation::VectorView<const double> output_adjoint, double* gradient) {
        const std::uint64_t centers = static_cast<std::uint64_t>(lattice.x) * lattice.y * lattice.z;
        ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(centers), control_vjp_kernel, grid, step, lattice, sigma, step_count, output_adjoint, gradient);
    }
} // namespace physica::fluids::gas::solvers::adjoint_control::kernels
