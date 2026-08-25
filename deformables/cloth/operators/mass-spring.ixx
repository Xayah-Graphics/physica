module;

#include <physica/cuda.h>

export module physica.deformables.cloth.operators.mass_spring;

import std;
import physica.deformables.cloth.domain;

export namespace physica::deformables::cloth::operators {
    struct MassSpringForce final {
        struct Configuration final {
            Vector3 gravity;
        };

        template <class Value>
        struct SpringParameters final {
            ScalarField<Value> stiffnesses;
            ScalarField<Value> dampings;
            ScalarField<Value> rest_lengths;
        };

        struct Parameters final {
            SpringParameters<float> stretch;
            SpringParameters<float> bending;
        };

        struct ParameterTangent final {
            SpringParameters<float> stretch;
            SpringParameters<float> bending;
        };

        struct ParameterAdjoint final {
            SpringParameters<double> stretch;
            SpringParameters<double> bending;
        };

        struct Cache final {};
        struct Workspace final {};
        struct TangentWorkspace final {};
        struct AdjointWorkspace final {};

        MassSpringForce(const Domain& domain, Configuration configuration);

        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>& external_forces, const ScalarField<float>& masses, const Parameters& parameters, VectorField<float>& forces, Cache& cache, Workspace& workspace) const;
        void jvp(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>& external_forces, const ScalarField<float>& masses, const Parameters& parameters, const VectorField<float>& forces, const Cache& cache, const VectorField<float>& position_tangent, const VectorField<float>& velocity_tangent, const VectorField<float>& external_force_tangent, const ScalarField<float>& mass_tangent, const ParameterTangent& parameter_tangent, VectorField<float>& force_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>& external_forces, const ScalarField<float>& masses, const Parameters& parameters, const VectorField<float>& forces, const Cache& cache, const VectorField<double>& force_adjoint, VectorField<double>& position_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& external_force_adjoint, ScalarField<double>& mass_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const;

    private:
        struct Spring final {
            std::uint32_t first;
            std::uint32_t second;
            float rest_length;
        };

        struct SpringTopology final {
            std::vector<Spring> springs;
            std::vector<std::uint32_t> offsets;
            std::vector<std::uint32_t> indices;
        };

        struct Topology final {
            SpringTopology stretch;
            SpringTopology bending;
        };

        struct DeviceSpringTopology final {
            ScalarField<std::uint32_t> first;
            ScalarField<std::uint32_t> second;
            ScalarField<std::uint32_t> offsets;
            ScalarField<std::uint32_t> indices;
        };

        struct DeviceTopology final {
            DeviceSpringTopology stretch;
            DeviceSpringTopology bending;
        };

        [[nodiscard]] static float distance(Vector3 first, Vector3 second);
        static void build_adjacency(SpringTopology& topology, std::size_t particle_count);
        [[nodiscard]] static Topology build_topology(const DomainConfiguration& configuration);
        [[nodiscard]] static DeviceSpringTopology allocate_device_topology(const Domain& domain, const SpringTopology& topology);
        static void upload_topology(const Domain& domain, const SpringTopology& topology, DeviceSpringTopology& destination);

        const Configuration configuration;
        const Topology topology;
        DeviceTopology device_topology;
    };
} // namespace physica::deformables::cloth::operators
