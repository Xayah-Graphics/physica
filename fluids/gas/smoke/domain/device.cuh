#ifndef PHYSICA_FLUIDS_GAS_SMOKE_DOMAIN_DEVICE_CUH
#define PHYSICA_FLUIDS_GAS_SMOKE_DOMAIN_DEVICE_CUH

#include "device.h"
#include <cuda_runtime.h>

namespace physica::fluids::gas::smoke::cuda_detail {
    namespace {
        constexpr unsigned block_size = 256u;

        __host__ __device__ std::uint64_t cell_count(const Grid grid) {
            return static_cast<std::uint64_t>(grid.nx) * grid.ny * grid.nz;
        }

        __host__ __device__ int extent(const Grid grid, const int component_axis, const int dimension) {
            const int base = dimension == 0 ? grid.nx : dimension == 1 ? grid.ny : grid.nz;
            return base + (component_axis == dimension ? 1 : 0);
        }

        __host__ __device__ std::uint64_t face_count(const Grid grid, const int axis) {
            return static_cast<std::uint64_t>(extent(grid, axis, 0)) * extent(grid, axis, 1) * extent(grid, axis, 2);
        }

        __device__ std::uint64_t index3(const int x, const int y, const int z, const int nx, const int ny) {
            return static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(nx) * (static_cast<std::uint64_t>(y) + static_cast<std::uint64_t>(ny) * z);
        }

        __device__ void decode(const std::uint64_t index, const int nx, const int ny, int& x, int& y, int& z) {
            x                      = static_cast<int>(index % nx);
            const std::uint64_t yz = index / nx;
            y                      = static_cast<int>(yz % ny);
            z                      = static_cast<int>(yz / ny);
        }

        __host__ __device__ float* component(const StaggeredVectorView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const float* component(const ConstStaggeredVectorView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const float* component(const ConstCenteredVectorView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ double* component(const StaggeredVectorAdjointView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const double* component(const ConstStaggeredVectorAdjointView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ double* component(const CenteredVectorAdjointView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __device__ int wrap(const int value, const int period) {
            const int remainder = value % period;
            return remainder < 0 ? remainder + period : remainder;
        }
    } // namespace
} // namespace physica::fluids::gas::smoke::cuda_detail

#endif
