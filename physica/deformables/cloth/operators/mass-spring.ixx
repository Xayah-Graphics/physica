module;

#include <physica/cuda.h>

export module physica.deformables.cloth.operators.mass_spring;

import std;
import physica.deformables.cloth.model;

export namespace physica::deformables::cloth::operators {
    struct MassSpringForce final {
        struct Configuration final {
            Vector3<float> gravity;
        };

        template <class Value>
        struct SpringParameters final {
            simulation::ScalarField<Value> stiffnesses;
            simulation::ScalarField<Value> dampings;
            simulation::ScalarField<Value> rest_lengths;
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

        MassSpringForce(const Model& model, Configuration configuration);

        [[nodiscard]] Parameters allocate_parameters(const Model& model) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Model& model) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Model& model) const;
        [[nodiscard]] Cache allocate_cache(const Model& model) const;
        [[nodiscard]] Workspace allocate_workspace(const Model& model) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Model& model) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Model& model) const;

        void forward(const Model& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>& external_forces, const simulation::ScalarField<float>& masses, const Parameters& parameters, simulation::VectorField<float>& forces, Cache& cache, Workspace& workspace) const;
        void jvp(const Model& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>& external_forces, const simulation::ScalarField<float>& masses, const Parameters& parameters, const simulation::VectorField<float>& forces, const Cache& cache, const simulation::VectorField<float>& position_tangent, const simulation::VectorField<float>& velocity_tangent, const simulation::VectorField<float>& external_force_tangent, const simulation::ScalarField<float>& mass_tangent, const ParameterTangent& parameter_tangent, simulation::VectorField<float>& force_tangent, TangentWorkspace& workspace) const;
        void vjp(const Model& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>& external_forces, const simulation::ScalarField<float>& masses, const Parameters& parameters, const simulation::VectorField<float>& forces, const Cache& cache, const simulation::VectorField<double>& force_adjoint, simulation::VectorField<double>& position_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& external_force_adjoint, simulation::ScalarField<double>& mass_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const;

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
            simulation::ScalarField<std::uint32_t> first;
            simulation::ScalarField<std::uint32_t> second;
            simulation::ScalarField<std::uint32_t> offsets;
            simulation::ScalarField<std::uint32_t> indices;
        };

        struct DeviceTopology final {
            DeviceSpringTopology stretch;
            DeviceSpringTopology bending;
        };

        static void build_adjacency(SpringTopology& topology, std::size_t particle_count);
        [[nodiscard]] static Topology build_topology(const ModelConfiguration& configuration);
        [[nodiscard]] static DeviceSpringTopology allocate_device_topology(const Model& model, const SpringTopology& topology);
        static void upload_topology(const Model& model, const SpringTopology& topology, DeviceSpringTopology& destination);

        const Configuration configuration;
        const Topology topology;
        DeviceTopology device_topology;
    };
} // namespace physica::deformables::cloth::operators
