#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_CONSERVATION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_CONSERVATION_KERNELS_H

#include "../detail/cuda/types.h"
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::cuda_backend {
    void mass_forward(::cuda::stream_ref stream, detail::cuda::Grid grid, float retention, detail::cuda::ScalarView<const float> input, detail::cuda::ScalarView<const float> advected, double* input_mass, double* advected_mass, detail::cuda::ScalarView<float> output);
    void mass_jvp(::cuda::stream_ref stream, detail::cuda::Grid grid, float retention, detail::cuda::ScalarView<const float> input, detail::cuda::ScalarView<const float> advected, detail::cuda::ScalarView<const float> input_tangent, detail::cuda::ScalarView<const float> advected_tangent, const double* input_mass, const double* advected_mass, double* input_mass_tangent, double* advected_mass_tangent, detail::cuda::ScalarView<float> output_tangent);
    void mass_vjp(::cuda::stream_ref stream, detail::cuda::Grid grid, float retention, detail::cuda::ScalarView<const float> advected, const double* input_mass, const double* advected_mass, detail::cuda::ScalarView<const double> output_adjoint, double* density_dot, detail::cuda::ScalarView<double> input_adjoint, detail::cuda::ScalarView<double> advected_adjoint);
} // namespace physica::fluids::gas::operators::cuda_backend

#endif
