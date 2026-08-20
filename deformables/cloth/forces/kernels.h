#ifndef PHYSICA_DEFORMABLES_CLOTH_FORCES_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_FORCES_KERNELS_H

#include "../domain/device.h"
#include <cstdint>
#include <cuda/stream>

namespace physica::deformables::cloth::cuda_detail {
    struct SpringTopologyView final {
        const std::uint32_t* first;
        const std::uint32_t* second;
        const std::uint32_t* offsets;
        const std::uint32_t* indices;
    };

    struct SpringParametersView final {
        const float* stiffnesses;
        const float* dampings;
        const float* rest_lengths;
    };

    struct SpringParameterAdjointView final {
        double* stiffnesses;
        double* dampings;
        double* rest_lengths;
    };

    void force_forward(::cuda::stream_ref stream, std::uint32_t particle_count, Vector<float> gravity, ConstFieldView<float> positions, ConstFieldView<float> velocities, ConstFieldView<float> controls, const float* masses, SpringTopologyView stretch_topology, SpringParametersView stretch_parameters, SpringTopologyView bending_topology, SpringParametersView bending_parameters, FieldView<float> forces);
    void force_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, Vector<float> gravity, ConstFieldView<float> positions, ConstFieldView<float> velocities, ConstFieldView<float> control_tangent, ConstFieldView<float> position_tangent, ConstFieldView<float> velocity_tangent, const float* mass_tangent, SpringTopologyView stretch_topology, SpringParametersView stretch_parameters, SpringParametersView stretch_tangent, SpringTopologyView bending_topology, SpringParametersView bending_parameters, SpringParametersView bending_tangent, FieldView<float> force_tangent);
    void force_state_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, Vector<float> gravity, ConstFieldView<float> positions, ConstFieldView<float> velocities, ConstFieldView<double> force_adjoint, SpringTopologyView stretch_topology, SpringParametersView stretch_parameters, SpringTopologyView bending_topology, SpringParametersView bending_parameters, FieldView<double> position_adjoint, FieldView<double> velocity_adjoint, FieldView<double> control_adjoint, double* mass_adjoint);
    void force_parameter_vjp(::cuda::stream_ref stream, std::uint32_t spring_count, ConstFieldView<float> positions, ConstFieldView<float> velocities, ConstFieldView<double> force_adjoint, SpringTopologyView topology, SpringParametersView parameters, SpringParameterAdjointView parameter_adjoint);
} // namespace physica::deformables::cloth::cuda_detail

#endif
