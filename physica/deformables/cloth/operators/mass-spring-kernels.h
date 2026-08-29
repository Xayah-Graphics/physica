#ifndef PHYSICA_DEFORMABLES_CLOTH_OPERATORS_MASS_SPRING_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_OPERATORS_MASS_SPRING_KERNELS_H

#include <physica/field/device.cuh>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::deformables::cloth::kernels {
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

    void force_forward(::cuda::stream_ref stream, std::uint32_t particle_count, Vector3<float> gravity, field::VectorView<const float> positions, field::VectorView<const float> velocities, field::VectorView<const float> controls, const float* masses, SpringTopologyView stretch_topology, SpringParametersView stretch_parameters, SpringTopologyView bending_topology, SpringParametersView bending_parameters, field::VectorView<float> forces);
    void force_jvp(::cuda::stream_ref stream, std::uint32_t particle_count, Vector3<float> gravity, field::VectorView<const float> positions, field::VectorView<const float> velocities, field::VectorView<const float> control_tangent, field::VectorView<const float> position_tangent, field::VectorView<const float> velocity_tangent, const float* mass_tangent, SpringTopologyView stretch_topology, SpringParametersView stretch_parameters, SpringParametersView stretch_tangent, SpringTopologyView bending_topology, SpringParametersView bending_parameters, SpringParametersView bending_tangent, field::VectorView<float> force_tangent);
    void force_state_vjp(::cuda::stream_ref stream, std::uint32_t particle_count, Vector3<float> gravity, field::VectorView<const float> positions, field::VectorView<const float> velocities, field::VectorView<const double> force_adjoint, SpringTopologyView stretch_topology, SpringParametersView stretch_parameters, SpringTopologyView bending_topology, SpringParametersView bending_parameters, field::VectorView<double> position_adjoint, field::VectorView<double> velocity_adjoint, field::VectorView<double> control_adjoint, double* mass_adjoint);
    void force_parameter_vjp(::cuda::stream_ref stream, std::uint32_t spring_count, field::VectorView<const float> positions, field::VectorView<const float> velocities, field::VectorView<const double> force_adjoint, SpringTopologyView topology, SpringParametersView parameters, SpringParameterAdjointView parameter_adjoint);
} // namespace physica::deformables::cloth::kernels

#endif
