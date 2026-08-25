#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_PROJECTION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_PROJECTION_KERNELS_H

#include "../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::cuda_backend {
    void pressure_rhs_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::ScalarView<float> rhs);
    void pressure_rhs_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const double> rhs_adjoint, detail::cuda::StaggeredVectorView<double> velocity_adjoint);
    void project_velocity_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::ScalarView<const float> pressure, detail::cuda::StaggeredVectorView<float> output);
    void project_velocity_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const double> output_adjoint, detail::cuda::StaggeredVectorView<double> velocity_adjoint, detail::cuda::ScalarView<double> pressure_adjoint);
    void red_black_gauss_seidel_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, detail::cuda::ScalarBoundaryData boundary, detail::cuda::ScalarView<const float> rhs, detail::cuda::ScalarView<float> pressure);
    void red_black_gauss_seidel_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* collider_ids, detail::cuda::ScalarBoundaryData boundary, detail::cuda::ScalarView<double> pressure_adjoint, detail::cuda::ScalarView<double> rhs_adjoint);
} // namespace physica::fluids::gas::operators::cuda_backend

#endif
