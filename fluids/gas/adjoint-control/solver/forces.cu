#include "../domain/device.cuh"
#include "kernels.h"
#include <cuda/launch>

namespace physica::fluids::gas::adjoint_control::cuda_detail {
    namespace {
        __global__ void heat_kernel(const Grid grid, const float buoyancy, const float* density, const CenteredVectorView force) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            force.x[index] = 0.0F;
            force.y[index] = buoyancy * density[index];
            force.z[index] = 0.0F;
        }

        __global__ void heat_reverse_kernel(const Grid grid, const float buoyancy, const ConstCenteredVectorAdjointView force_adjoint, double* density_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < cell_count(grid)) density_adjoint[index] += buoyancy * force_adjoint.y[index];
        }
    } // namespace

    void heat_forward(const ::cuda::stream_ref stream, const Grid grid, const float buoyancy, const ConstScalarView density, const CenteredVectorView force) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), heat_kernel, grid, buoyancy, density.values, force);
    }

    void heat_jvp(const ::cuda::stream_ref stream, const Grid grid, const float buoyancy, const ConstScalarView density_tangent, const CenteredVectorView force_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), heat_kernel, grid, buoyancy, density_tangent.values, force_tangent);
    }

    void heat_vjp(const ::cuda::stream_ref stream, const Grid grid, const float buoyancy, const ConstCenteredVectorAdjointView force_adjoint, const ScalarAdjointView density_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(cell_count(grid)), heat_reverse_kernel, grid, buoyancy, force_adjoint, density_adjoint.values);
    }
} // namespace physica::fluids::gas::adjoint_control::cuda_detail
