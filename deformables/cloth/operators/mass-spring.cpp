module;

#include "../detail/cuda/interop.h"
#include "mass-spring-kernels.h"
#include <physica/cuda.h>

module physica.deformables.cloth.operators.mass_spring;

import std;

namespace physica::deformables::cloth::operators {
    namespace {
        auto topology_view(const auto& topology) {
            return cuda_detail::SpringTopologyView{
                .first   = topology.first.values.data(),
                .second  = topology.second.values.data(),
                .offsets = topology.offsets.values.data(),
                .indices = topology.indices.values.data(),
            };
        }

        auto parameter_view(const auto& parameters) {
            return cuda_detail::SpringParametersView{
                .stiffnesses  = parameters.stiffnesses.values.data(),
                .dampings     = parameters.dampings.values.data(),
                .rest_lengths = parameters.rest_lengths.values.data(),
            };
        }

        auto parameter_adjoint_view(auto& parameters) {
            return cuda_detail::SpringParameterAdjointView{
                .stiffnesses  = parameters.stiffnesses.values.data(),
                .dampings     = parameters.dampings.values.data(),
                .rest_lengths = parameters.rest_lengths.values.data(),
            };
        }
    } // namespace

    MassSpringForce::MassSpringForce(const Domain& domain, Configuration next_configuration) : configuration(std::move(next_configuration)), topology(build_topology(domain.configuration)), device_topology{.stretch = allocate_device_topology(domain, topology.stretch), .bending = allocate_device_topology(domain, topology.bending)} {
        upload_topology(domain, topology.stretch, device_topology.stretch);
        upload_topology(domain, topology.bending, device_topology.bending);
        domain.stream.sync();
    }

    MassSpringForce::Parameters MassSpringForce::allocate_parameters(const Domain& domain) const {
        Parameters result{
            .stretch = {
                .stiffnesses  = domain.allocate_scalar_field<float>(topology.stretch.springs.size()),
                .dampings     = domain.allocate_scalar_field<float>(topology.stretch.springs.size()),
                .rest_lengths = domain.allocate_scalar_field<float>(topology.stretch.springs.size()),
            },
            .bending = {
                .stiffnesses  = domain.allocate_scalar_field<float>(topology.bending.springs.size()),
                .dampings     = domain.allocate_scalar_field<float>(topology.bending.springs.size()),
                .rest_lengths = domain.allocate_scalar_field<float>(topology.bending.springs.size()),
            },
        };
        domain.clear(result.stretch.stiffnesses);
        domain.clear(result.stretch.dampings);
        domain.clear(result.bending.stiffnesses);
        domain.clear(result.bending.dampings);

        std::vector<float> stretch_rest_lengths(topology.stretch.springs.size());
        std::vector<float> bending_rest_lengths(topology.bending.springs.size());
        for (std::size_t spring = 0uz; spring < topology.stretch.springs.size(); ++spring) stretch_rest_lengths[spring] = topology.stretch.springs[spring].rest_length;
        for (std::size_t spring = 0uz; spring < topology.bending.springs.size(); ++spring) bending_rest_lengths[spring] = topology.bending.springs[spring].rest_length;
        ::cuda::copy_bytes(domain.stream, stretch_rest_lengths, result.stretch.rest_lengths.values);
        ::cuda::copy_bytes(domain.stream, bending_rest_lengths, result.bending.rest_lengths.values);
        domain.stream.sync();
        return result;
    }

    MassSpringForce::ParameterTangent MassSpringForce::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent result{
            .stretch = {
                .stiffnesses  = domain.allocate_scalar_field<float>(topology.stretch.springs.size()),
                .dampings     = domain.allocate_scalar_field<float>(topology.stretch.springs.size()),
                .rest_lengths = domain.allocate_scalar_field<float>(topology.stretch.springs.size()),
            },
            .bending = {
                .stiffnesses  = domain.allocate_scalar_field<float>(topology.bending.springs.size()),
                .dampings     = domain.allocate_scalar_field<float>(topology.bending.springs.size()),
                .rest_lengths = domain.allocate_scalar_field<float>(topology.bending.springs.size()),
            },
        };
        domain.clear(result.stretch.stiffnesses);
        domain.clear(result.stretch.dampings);
        domain.clear(result.stretch.rest_lengths);
        domain.clear(result.bending.stiffnesses);
        domain.clear(result.bending.dampings);
        domain.clear(result.bending.rest_lengths);
        return result;
    }

    MassSpringForce::ParameterAdjoint MassSpringForce::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint result{
            .stretch = {
                .stiffnesses  = domain.allocate_scalar_field<double>(topology.stretch.springs.size()),
                .dampings     = domain.allocate_scalar_field<double>(topology.stretch.springs.size()),
                .rest_lengths = domain.allocate_scalar_field<double>(topology.stretch.springs.size()),
            },
            .bending = {
                .stiffnesses  = domain.allocate_scalar_field<double>(topology.bending.springs.size()),
                .dampings     = domain.allocate_scalar_field<double>(topology.bending.springs.size()),
                .rest_lengths = domain.allocate_scalar_field<double>(topology.bending.springs.size()),
            },
        };
        domain.clear(result.stretch.stiffnesses);
        domain.clear(result.stretch.dampings);
        domain.clear(result.stretch.rest_lengths);
        domain.clear(result.bending.stiffnesses);
        domain.clear(result.bending.dampings);
        domain.clear(result.bending.rest_lengths);
        return result;
    }

    MassSpringForce::Cache MassSpringForce::allocate_cache(const Domain&) const {
        return {};
    }

    MassSpringForce::Workspace MassSpringForce::allocate_workspace(const Domain&) const {
        return {};
    }

    MassSpringForce::TangentWorkspace MassSpringForce::allocate_tangent_workspace(const Domain&) const {
        return {};
    }

    MassSpringForce::AdjointWorkspace MassSpringForce::allocate_adjoint_workspace(const Domain&) const {
        return {};
    }

    void MassSpringForce::forward(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>& external_forces, const ScalarField<float>& masses, const Parameters& parameters, VectorField<float>& forces, Cache&, Workspace&) const {
        cuda_detail::force_forward(domain.stream, static_cast<std::uint32_t>(domain.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, cuda_detail::field<float>(positions), cuda_detail::field<float>(velocities), cuda_detail::field<float>(external_forces), masses.values.data(), topology_view(device_topology.stretch), parameter_view(parameters.stretch), topology_view(device_topology.bending), parameter_view(parameters.bending), cuda_detail::field<float>(forces));
    }

    void MassSpringForce::jvp(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>&, const ScalarField<float>&, const Parameters& parameters, const VectorField<float>&, const Cache&, const VectorField<float>& position_tangent, const VectorField<float>& velocity_tangent, const VectorField<float>& external_force_tangent, const ScalarField<float>& mass_tangent, const ParameterTangent& parameter_tangent, VectorField<float>& force_tangent, TangentWorkspace&) const {
        cuda_detail::force_jvp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, cuda_detail::field<float>(positions), cuda_detail::field<float>(velocities), cuda_detail::field<float>(external_force_tangent), cuda_detail::field<float>(position_tangent), cuda_detail::field<float>(velocity_tangent), mass_tangent.values.data(), topology_view(device_topology.stretch), parameter_view(parameters.stretch), parameter_view(parameter_tangent.stretch), topology_view(device_topology.bending), parameter_view(parameters.bending), parameter_view(parameter_tangent.bending), cuda_detail::field<float>(force_tangent));
    }

    void MassSpringForce::vjp(const Domain& domain, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>&, const ScalarField<float>&, const Parameters& parameters, const VectorField<float>&, const Cache&, const VectorField<double>& force_adjoint, VectorField<double>& position_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& external_force_adjoint, ScalarField<double>& mass_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace&) const {
        cuda_detail::force_state_vjp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, cuda_detail::field<float>(positions), cuda_detail::field<float>(velocities), cuda_detail::field<double>(force_adjoint), topology_view(device_topology.stretch), parameter_view(parameters.stretch), topology_view(device_topology.bending), parameter_view(parameters.bending), cuda_detail::field<double>(position_adjoint), cuda_detail::field<double>(velocity_adjoint), cuda_detail::field<double>(external_force_adjoint), mass_adjoint.values.data());
        cuda_detail::force_parameter_vjp(domain.stream, static_cast<std::uint32_t>(topology.stretch.springs.size()), cuda_detail::field<float>(positions), cuda_detail::field<float>(velocities), cuda_detail::field<double>(force_adjoint), topology_view(device_topology.stretch), parameter_view(parameters.stretch), parameter_adjoint_view(parameter_adjoint.stretch));
        cuda_detail::force_parameter_vjp(domain.stream, static_cast<std::uint32_t>(topology.bending.springs.size()), cuda_detail::field<float>(positions), cuda_detail::field<float>(velocities), cuda_detail::field<double>(force_adjoint), topology_view(device_topology.bending), parameter_view(parameters.bending), parameter_adjoint_view(parameter_adjoint.bending));
    }

    float MassSpringForce::distance(const Vector3 first, const Vector3 second) {
        const float x = second.x - first.x;
        const float y = second.y - first.y;
        const float z = second.z - first.z;
        return std::sqrt(x * x + y * y + z * z);
    }

    void MassSpringForce::build_adjacency(SpringTopology& spring_topology, const std::size_t particle_count) {
        spring_topology.offsets.assign(particle_count + 1uz, 0u);
        for (const Spring& spring : spring_topology.springs) {
            ++spring_topology.offsets[spring.first + 1u];
            ++spring_topology.offsets[spring.second + 1u];
        }
        for (std::size_t particle = 1uz; particle < spring_topology.offsets.size(); ++particle) spring_topology.offsets[particle] += spring_topology.offsets[particle - 1uz];
        spring_topology.indices.resize(spring_topology.springs.size() * 2uz);
        std::vector<std::uint32_t> cursors = spring_topology.offsets;
        for (std::uint32_t spring = 0u; spring < spring_topology.springs.size(); ++spring) {
            spring_topology.indices[cursors[spring_topology.springs[spring].first]++]  = spring;
            spring_topology.indices[cursors[spring_topology.springs[spring].second]++] = spring;
        }
    }

    MassSpringForce::Topology MassSpringForce::build_topology(const DomainConfiguration& domain_configuration) {
        std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::uint32_t>> edge_opposites;
        for (const Triangle triangle : domain_configuration.triangles) {
            const std::array vertices{triangle.first, triangle.second, triangle.third};
            for (std::size_t edge = 0uz; edge < 3uz; ++edge) {
                const std::uint32_t first  = vertices[edge] < vertices[(edge + 1uz) % 3uz] ? vertices[edge] : vertices[(edge + 1uz) % 3uz];
                const std::uint32_t second = vertices[edge] < vertices[(edge + 1uz) % 3uz] ? vertices[(edge + 1uz) % 3uz] : vertices[edge];
                edge_opposites[{first, second}].push_back(vertices[(edge + 2uz) % 3uz]);
            }
        }

        Topology result{};
        result.stretch.springs.reserve(edge_opposites.size());
        for (const auto& [edge, opposites] : edge_opposites) {
            result.stretch.springs.push_back({.first = edge.first, .second = edge.second, .rest_length = distance(domain_configuration.rest_positions[edge.first], domain_configuration.rest_positions[edge.second])});
            if (opposites.size() == 2uz) {
                const std::uint32_t first  = opposites[0] < opposites[1] ? opposites[0] : opposites[1];
                const std::uint32_t second = opposites[0] < opposites[1] ? opposites[1] : opposites[0];
                result.bending.springs.push_back({.first = first, .second = second, .rest_length = distance(domain_configuration.rest_positions[first], domain_configuration.rest_positions[second])});
            }
        }
        build_adjacency(result.stretch, domain_configuration.rest_positions.size());
        build_adjacency(result.bending, domain_configuration.rest_positions.size());
        return result;
    }

    MassSpringForce::DeviceSpringTopology MassSpringForce::allocate_device_topology(const Domain& domain, const SpringTopology& spring_topology) {
        return {
            .first   = domain.allocate_scalar_field<std::uint32_t>(spring_topology.springs.size()),
            .second  = domain.allocate_scalar_field<std::uint32_t>(spring_topology.springs.size()),
            .offsets = domain.allocate_scalar_field<std::uint32_t>(spring_topology.offsets.size()),
            .indices = domain.allocate_scalar_field<std::uint32_t>(spring_topology.indices.size()),
        };
    }

    void MassSpringForce::upload_topology(const Domain& domain, const SpringTopology& spring_topology, DeviceSpringTopology& destination) {
        std::vector<std::uint32_t> first(spring_topology.springs.size());
        std::vector<std::uint32_t> second(spring_topology.springs.size());
        for (std::size_t spring = 0uz; spring < spring_topology.springs.size(); ++spring) {
            first[spring]  = spring_topology.springs[spring].first;
            second[spring] = spring_topology.springs[spring].second;
        }
        ::cuda::copy_bytes(domain.stream, first, destination.first.values);
        ::cuda::copy_bytes(domain.stream, second, destination.second.values);
        ::cuda::copy_bytes(domain.stream, spring_topology.offsets, destination.offsets.values);
        ::cuda::copy_bytes(domain.stream, spring_topology.indices, destination.indices.values);
    }
} // namespace physica::deformables::cloth::operators
