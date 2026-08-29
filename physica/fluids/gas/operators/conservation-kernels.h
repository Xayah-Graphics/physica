#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_CONSERVATION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_CONSERVATION_KERNELS_H

#include <physica/fluids/gas/device.cuh>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void mass_forward(::cuda::stream_ref stream, device::Discretization grid, float retention, field::ScalarView<const float> input, field::ScalarView<const float> advected, double* input_mass, double* advected_mass, field::ScalarView<float> output);
    void mass_jvp(::cuda::stream_ref stream, device::Discretization grid, float retention, field::ScalarView<const float> input, field::ScalarView<const float> advected, field::ScalarView<const float> input_tangent, field::ScalarView<const float> advected_tangent, const double* input_mass, const double* advected_mass, double* input_mass_tangent, double* advected_mass_tangent, field::ScalarView<float> output_tangent);
    void mass_vjp(::cuda::stream_ref stream, device::Discretization grid, float retention, field::ScalarView<const float> advected, const double* input_mass, const double* advected_mass, field::ScalarView<const double> output_adjoint, double* density_dot, field::ScalarView<double> input_adjoint, field::ScalarView<double> advected_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
