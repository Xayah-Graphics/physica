#ifndef PHYSICA_FLUIDS_GAS_OPERATORS_CONSERVATION_KERNELS_H
#define PHYSICA_FLUIDS_GAS_OPERATORS_CONSERVATION_KERNELS_H

#include <fluids/gas/device.cuh>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::operators::kernels {
    void mass_forward(::cuda::stream_ref stream, device::Discretization grid, float retention, simulation::ScalarView<const float> input, simulation::ScalarView<const float> advected, double* input_mass, double* advected_mass, simulation::ScalarView<float> output);
    void mass_jvp(::cuda::stream_ref stream, device::Discretization grid, float retention, simulation::ScalarView<const float> input, simulation::ScalarView<const float> advected, simulation::ScalarView<const float> input_tangent, simulation::ScalarView<const float> advected_tangent, const double* input_mass, const double* advected_mass, double* input_mass_tangent, double* advected_mass_tangent, simulation::ScalarView<float> output_tangent);
    void mass_vjp(::cuda::stream_ref stream, device::Discretization grid, float retention, simulation::ScalarView<const float> advected, const double* input_mass, const double* advected_mass, simulation::ScalarView<const double> output_adjoint, double* density_dot, simulation::ScalarView<double> input_adjoint, simulation::ScalarView<double> advected_adjoint);
} // namespace physica::fluids::gas::operators::kernels

#endif
