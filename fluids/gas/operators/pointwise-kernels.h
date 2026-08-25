#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_POINTWISE_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_POINTWISE_KERNELS_H

#include "../detail/cuda/types.h"
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::cuda_backend {
    void source_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const float> state, detail::cuda::ScalarView<const float> source, detail::cuda::ScalarView<float> output);
    void source_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const float> state_tangent, detail::cuda::ScalarView<const float> source_tangent, detail::cuda::ScalarView<float> output_tangent);
    void source_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::ScalarView<const double> output_adjoint, detail::cuda::ScalarView<double> state_adjoint, detail::cuda::ScalarView<double> source_adjoint);

    void integrate_velocity_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::CenteredVectorView<const float> force, detail::cuda::StaggeredVectorView<float> output);
    void integrate_velocity_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity_tangent, detail::cuda::CenteredVectorView<const float> force_tangent, detail::cuda::StaggeredVectorView<float> output_tangent);
    void integrate_velocity_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const double> output_adjoint, detail::cuda::StaggeredVectorView<double> velocity_adjoint, detail::cuda::CenteredVectorView<double> force_adjoint);

    void constrain_velocity_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> collider_velocity, detail::cuda::StaggeredVectorView<const float> velocity, detail::cuda::VelocityBoundaryData boundary, detail::cuda::StaggeredVectorView<float> output);
    void constrain_velocity_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const float> velocity_tangent, detail::cuda::VelocityBoundaryData boundary, detail::cuda::StaggeredVectorView<float> output_tangent);
    void constrain_velocity_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, const std::uint32_t* collider_ids, detail::cuda::StaggeredVectorView<const double> output_adjoint, detail::cuda::VelocityBoundaryData boundary, detail::cuda::StaggeredVectorView<double> velocity_adjoint);
} // namespace physica::fluids::gas::operators::cuda_backend

#endif
