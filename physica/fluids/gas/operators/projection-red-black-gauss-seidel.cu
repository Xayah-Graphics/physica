#include <physica/fluids/gas/device.cuh>
#include "projection-kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::operators::kernels {
    namespace {
        __device__ bool pressure_periodic(const device::ScalarBoundary boundary, const int dimension) {
            return boundary.faces[dimension * 2].mode == 2u && boundary.faces[dimension * 2 + 1].mode == 2u;
        }

        __device__ void pressure_neighbor(const device::Discretization grid, const device::ScalarBoundary boundary, const std::uint32_t* collider_ids, const int x, const int y, const int z, const int dimension, const int direction, int& neighbor_x, int& neighbor_y, int& neighbor_z, bool& connected, bool& fixed, float& fixed_value) {
            neighbor_x           = x + (dimension == 0 ? direction : 0);
            neighbor_y           = y + (dimension == 1 ? direction : 0);
            neighbor_z           = z + (dimension == 2 ? direction : 0);
            const int coordinate = dimension == 0 ? neighbor_x : dimension == 1 ? neighbor_y : neighbor_z;
            const int size       = dimension == 0 ? grid.grid.nx : dimension == 1 ? grid.grid.ny : grid.grid.nz;
            if (coordinate < 0 || coordinate >= size) {
                if (pressure_periodic(boundary, dimension)) {
                    if (dimension == 0) neighbor_x = device::wrap(neighbor_x, grid.grid.nx);
                    if (dimension == 1) neighbor_y = device::wrap(neighbor_y, grid.grid.ny);
                    if (dimension == 2) neighbor_z = device::wrap(neighbor_z, grid.grid.nz);
                } else {
                    const int face = dimension * 2 + (direction > 0);
                    fixed          = boundary.faces[face].mode == 0u;
                    fixed_value    = boundary.faces[face].value;
                    connected      = false;
                    return;
                }
            }
            connected = collider_ids[fluids::grid::device::index3(neighbor_x, neighbor_y, neighbor_z, grid.grid.nx, grid.grid.ny)] == 0u;
            fixed     = false;
        }

        template <std::uint32_t dimensions>
        __global__ void rbgs_kernel(const device::Discretization grid, const int parity, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const device::ScalarBoundary boundary, const float* rhs, float* pressure) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            if (((x + y + z) & 1) != parity || collider_ids[index] != 0u || index == pressure_anchor) return;
            float sum      = rhs[index];
            float diagonal = 0.0F;
            for (int dimension = 0; dimension < dimensions; ++dimension) {
                for (int direction = -1; direction <= 1; direction += 2) {
                    int nx, ny, nz;
                    bool connected, fixed;
                    float fixed_value{};
                    pressure_neighbor(grid, boundary, collider_ids, x, y, z, dimension, direction, nx, ny, nz, connected, fixed, fixed_value);
                    if (connected) {
                        sum += pressure[fluids::grid::device::index3(nx, ny, nz, grid.grid.nx, grid.grid.ny)];
                        diagonal += 1.0F;
                    } else if (fixed) {
                        sum += fixed_value;
                        diagonal += 1.0F;
                    }
                }
            }
            pressure[index] = diagonal == 0.0F ? 0.0F : sum / diagonal;
        }

        template <std::uint32_t dimensions>
        __global__ void rbgs_reverse_kernel(const device::Discretization grid, const int parity, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const device::ScalarBoundary boundary, double* pressure_adjoint, double* rhs_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= fluids::grid::device::cell_count(grid.grid)) return;
            int x, y, z;
            fluids::grid::device::decode(index, grid.grid.nx, grid.grid.ny, x, y, z);
            if (((x + y + z) & 1) != parity || collider_ids[index] != 0u || index == pressure_anchor) return;
            float diagonal = 0.0F;
            int neighbors[6];
            int neighbor_count = 0;
            for (int dimension = 0; dimension < dimensions; ++dimension) {
                for (int direction = -1; direction <= 1; direction += 2) {
                    int nx, ny, nz;
                    bool connected, fixed;
                    float fixed_value{};
                    pressure_neighbor(grid, boundary, collider_ids, x, y, z, dimension, direction, nx, ny, nz, connected, fixed, fixed_value);
                    if (connected) {
                        neighbors[neighbor_count++] = static_cast<int>(fluids::grid::device::index3(nx, ny, nz, grid.grid.nx, grid.grid.ny));
                        diagonal += 1.0F;
                    } else if (fixed) diagonal += 1.0F;
                }
            }
            const double adjoint    = diagonal == 0.0F ? 0.0 : pressure_adjoint[index] / diagonal;
            pressure_adjoint[index] = 0.0;
            rhs_adjoint[index] += adjoint;
            for (int neighbor = 0; neighbor < neighbor_count; ++neighbor) atomicAdd(pressure_adjoint + neighbors[neighbor], adjoint);
        }

        template <std::uint32_t dimensions>
        void launch_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const device::ScalarBoundary boundary, const field::ScalarView<const float> rhs, const field::ScalarView<float> pressure) {
            for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
                ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), rbgs_kernel<dimensions>, grid, 0, pressure_anchor, collider_ids, boundary, rhs.values, pressure.values);
                ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), rbgs_kernel<dimensions>, grid, 1, pressure_anchor, collider_ids, boundary, rhs.values, pressure.values);
            }
        }

        template <std::uint32_t dimensions>
        void launch_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const device::ScalarBoundary boundary, const field::ScalarView<double> pressure_adjoint, const field::ScalarView<double> rhs_adjoint) {
            for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
                ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), rbgs_reverse_kernel<dimensions>, grid, 1, pressure_anchor, collider_ids, boundary, pressure_adjoint.values, rhs_adjoint.values);
                ::cuda::launch(stream, ::cuda::distribute<fluids::grid::device::block_size>(fluids::grid::device::cell_count(grid.grid)), rbgs_reverse_kernel<dimensions>, grid, 0, pressure_anchor, collider_ids, boundary, pressure_adjoint.values, rhs_adjoint.values);
            }
        }
    } // namespace

    void red_black_gauss_seidel_forward(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const device::ScalarBoundary boundary, const field::ScalarView<const float> rhs, const field::ScalarView<float> pressure) {
        switch (grid.dimensions) {
        case 2u: launch_forward<2u>(stream, grid, iterations, pressure_anchor, collider_ids, boundary, rhs, pressure); break;
        case 3u: launch_forward<3u>(stream, grid, iterations, pressure_anchor, collider_ids, boundary, rhs, pressure); break;
        }
    }

    void red_black_gauss_seidel_vjp(const ::cuda::stream_ref stream, const device::Discretization grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, const device::ScalarBoundary boundary, const field::ScalarView<double> pressure_adjoint, const field::ScalarView<double> rhs_adjoint) {
        switch (grid.dimensions) {
        case 2u: launch_vjp<2u>(stream, grid, iterations, pressure_anchor, collider_ids, boundary, pressure_adjoint, rhs_adjoint); break;
        case 3u: launch_vjp<3u>(stream, grid, iterations, pressure_anchor, collider_ids, boundary, pressure_adjoint, rhs_adjoint); break;
        }
    }
} // namespace physica::fluids::gas::operators::kernels
