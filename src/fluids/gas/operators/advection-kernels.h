#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_ADVECTION_KERNELS_H

#include "../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::cuda_backend {
    void advect_velocity_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::VelocityBoundaryData boundary, detail::cuda::StaggeredVectorView<float> output);
    void advect_velocity_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::StaggeredVectorView<const float> velocity_tangent, detail::cuda::VelocityBoundaryData boundary, detail::cuda::StaggeredVectorView<float> output_tangent);
    void advect_velocity_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::VelocityBoundaryData boundary, detail::cuda::StaggeredVectorView<const double> output_adjoint, detail::cuda::StaggeredVectorView<double> velocity_adjoint);

    void advect_scalar_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const float> collider_value, detail::cuda::ScalarView<const float> source, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::ScalarBoundaryData scalar_boundary, detail::cuda::VelocityBoundaryData velocity_boundary, detail::cuda::ScalarView<float> output);
    void advect_scalar_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const float> source, detail::cuda::ScalarView<const float> source_tangent, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::StaggeredVectorView<const float> velocity_tangent, detail::cuda::ScalarBoundaryData scalar_boundary, detail::cuda::VelocityBoundaryData velocity_boundary, detail::cuda::ScalarView<float> output_tangent);
    void advect_scalar_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const float> source, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::ScalarBoundaryData scalar_boundary, detail::cuda::VelocityBoundaryData velocity_boundary, detail::cuda::ScalarView<const double> output_adjoint, detail::cuda::ScalarView<double> source_adjoint, detail::cuda::StaggeredVectorView<double> velocity_adjoint);
} // namespace physica::fluids::gas::operators::cuda_backend

#endif
