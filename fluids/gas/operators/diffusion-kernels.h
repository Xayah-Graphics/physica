#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_DIFFUSION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_DIFFUSION_KERNELS_H

#include "../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::cuda_backend {
    void identity_velocity_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, detail::cuda::StaggeredVectorView<const double> output_adjoint, detail::cuda::StaggeredVectorView<double> source_adjoint);
    void diffusion_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t iterations, float viscosity, const std::uint32_t* collider_ids, detail::cuda::VelocityBoundaryData boundary, detail::cuda::StaggeredVectorView<const float> source, detail::cuda::StaggeredVectorView<float> first, detail::cuda::StaggeredVectorView<float> second, detail::cuda::StaggeredVectorView<float> output);
    void diffusion_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t iterations, float viscosity, const std::uint32_t* collider_ids, detail::cuda::VelocityBoundaryData boundary, detail::cuda::StaggeredVectorView<const double> output_adjoint, detail::cuda::StaggeredVectorView<double> first, detail::cuda::StaggeredVectorView<double> second, detail::cuda::StaggeredVectorView<double> source_adjoint);
} // namespace physica::fluids::gas::operators::cuda_backend

#endif
