module;

#include "../domain/interop.h"
#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>

module physica.deformables.cloth.forces;

import std;

namespace physica::deformables::cloth {
    namespace {
        float distance(const Vector3 first, const Vector3 second) {
            const float x = second.x - first.x;
            const float y = second.y - first.y;
            const float z = second.z - first.z;
            return std::sqrt(x * x + y * y + z * z);
        }

        void build_adjacency(SpringTopology& topology, const std::size_t particle_count) {
            topology.offsets.assign(particle_count + 1uz, 0u);
            for (const Spring& spring : topology.springs) {
                ++topology.offsets[spring.first + 1u];
                ++topology.offsets[spring.second + 1u];
            }
            for (std::size_t particle = 1uz; particle < topology.offsets.size(); ++particle) topology.offsets[particle] += topology.offsets[particle - 1uz];
            topology.indices.resize(topology.springs.size() * 2uz);
            std::vector<std::uint32_t> cursors = topology.offsets;
            for (std::uint32_t spring = 0u; spring < topology.springs.size(); ++spring) {
                topology.indices[cursors[topology.springs[spring].first]++] = spring;
                topology.indices[cursors[topology.springs[spring].second]++] = spring;
            }
        }

        MassSpringTopology build_topology(const DomainConfiguration& configuration) {
            std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::uint32_t>> edge_opposites;
            for (const Triangle triangle : configuration.triangles) {
                const std::array vertices{triangle.first, triangle.second, triangle.third};
                for (std::size_t edge = 0uz; edge < 3uz; ++edge) {
                    const std::uint32_t first = vertices[edge] < vertices[(edge + 1uz) % 3uz] ? vertices[edge] : vertices[(edge + 1uz) % 3uz];
                    const std::uint32_t second = vertices[edge] < vertices[(edge + 1uz) % 3uz] ? vertices[(edge + 1uz) % 3uz] : vertices[edge];
                    edge_opposites[{first, second}].push_back(vertices[(edge + 2uz) % 3uz]);
                }
            }

            MassSpringTopology result{};
            result.stretch.springs.reserve(edge_opposites.size());
            for (const auto& [edge, opposites] : edge_opposites) {
                result.stretch.springs.push_back({.first = edge.first, .second = edge.second, .rest_length = distance(configuration.rest_positions[edge.first], configuration.rest_positions[edge.second])});
                if (opposites.size() == 2uz) {
                    const std::uint32_t first = opposites[0] < opposites[1] ? opposites[0] : opposites[1];
                    const std::uint32_t second = opposites[0] < opposites[1] ? opposites[1] : opposites[0];
                    result.bending.springs.push_back({.first = first, .second = second, .rest_length = distance(configuration.rest_positions[first], configuration.rest_positions[second])});
                }
            }
            build_adjacency(result.stretch, configuration.rest_positions.size());
            build_adjacency(result.bending, configuration.rest_positions.size());
            return result;
        }

        DeviceSpringTopology allocate_device_topology(const Domain& domain, const SpringTopology& topology) {
            return {
                .first = domain.allocate_index_field(topology.springs.size()),
                .second = domain.allocate_index_field(topology.springs.size()),
                .offsets = domain.allocate_index_field(topology.offsets.size()),
                .indices = domain.allocate_index_field(topology.indices.size()),
            };
        }

        void upload_topology(const Domain& domain, const SpringTopology& topology, DeviceSpringTopology& destination) {
            std::vector<std::uint32_t> first(topology.springs.size());
            std::vector<std::uint32_t> second(topology.springs.size());
            for (std::size_t spring = 0uz; spring < topology.springs.size(); ++spring) {
                first[spring] = topology.springs[spring].first;
                second[spring] = topology.springs[spring].second;
            }
            ::cuda::copy_bytes(domain.stream, first, destination.first.values);
            ::cuda::copy_bytes(domain.stream, second, destination.second.values);
            ::cuda::copy_bytes(domain.stream, topology.offsets, destination.offsets.values);
            ::cuda::copy_bytes(domain.stream, topology.indices, destination.indices.values);
        }

        cuda_detail::SpringTopologyView view(const DeviceSpringTopology& topology) {
            return {
                .first = topology.first.values.data(),
                .second = topology.second.values.data(),
                .offsets = topology.offsets.values.data(),
                .indices = topology.indices.values.data(),
            };
        }

        cuda_detail::SpringParametersView view(const MassSpringForces::SpringParameters& parameters) {
            return {.stiffnesses = parameters.stiffnesses.values.data(), .dampings = parameters.dampings.values.data(), .rest_lengths = parameters.rest_lengths.values.data()};
        }

        cuda_detail::SpringParameterAdjointView view(MassSpringForces::SpringParameterAdjoint& parameters) {
            return {.stiffnesses = parameters.stiffnesses.values.data(), .dampings = parameters.dampings.values.data(), .rest_lengths = parameters.rest_lengths.values.data()};
        }

        MassSpringForces::SpringParameters allocate_spring_parameters(const Domain& domain, const std::size_t count) {
            return {.stiffnesses = domain.allocate_scalar_field(count), .dampings = domain.allocate_scalar_field(count), .rest_lengths = domain.allocate_scalar_field(count)};
        }

        MassSpringForces::SpringParameterAdjoint allocate_spring_parameter_adjoint(const Domain& domain, const std::size_t count) {
            return {.stiffnesses = domain.allocate_scalar_adjoint_field(count), .dampings = domain.allocate_scalar_adjoint_field(count), .rest_lengths = domain.allocate_scalar_adjoint_field(count)};
        }

        void clear(const Domain& domain, MassSpringForces::SpringParameters& parameters) {
            domain.clear(parameters.stiffnesses);
            domain.clear(parameters.dampings);
            domain.clear(parameters.rest_lengths);
        }

        void clear(const Domain& domain, MassSpringForces::SpringParameterAdjoint& parameters) {
            domain.clear(parameters.stiffnesses);
            domain.clear(parameters.dampings);
            domain.clear(parameters.rest_lengths);
        }
    } // namespace

    MassSpringForces::MassSpringForces(const Domain& domain, Configuration next_configuration, const ExecutionMode mode)
        : configuration(std::move(next_configuration)), topology(build_topology(domain.configuration)), device_topology{.stretch = allocate_device_topology(domain, topology.stretch), .bending = allocate_device_topology(domain, topology.bending)}, differentiation{} {
        upload_topology(domain, topology.stretch, device_topology.stretch);
        upload_topology(domain, topology.bending, device_topology.bending);
        if (mode == ExecutionMode::differentiable) differentiation.emplace(Differentiation{.tangent = domain.allocate_vector_field(), .adjoint = domain.allocate_vector_adjoint_field()});
        domain.stream.sync();
    }

    MassSpringForces::Parameters MassSpringForces::allocate_parameters(const Domain& domain) const {
        Parameters result{.stretch = allocate_spring_parameters(domain, topology.stretch.springs.size()), .bending = allocate_spring_parameters(domain, topology.bending.springs.size())};
        clear(domain, result.stretch);
        clear(domain, result.bending);
        std::vector<float> stretch_rest_lengths(topology.stretch.springs.size());
        std::vector<float> bending_rest_lengths(topology.bending.springs.size());
        for (std::size_t spring = 0uz; spring < topology.stretch.springs.size(); ++spring) stretch_rest_lengths[spring] = topology.stretch.springs[spring].rest_length;
        for (std::size_t spring = 0uz; spring < topology.bending.springs.size(); ++spring) bending_rest_lengths[spring] = topology.bending.springs[spring].rest_length;
        ::cuda::copy_bytes(domain.stream, stretch_rest_lengths, result.stretch.rest_lengths.values);
        ::cuda::copy_bytes(domain.stream, bending_rest_lengths, result.bending.rest_lengths.values);
        domain.stream.sync();
        return result;
    }

    MassSpringForces::ParameterTangent MassSpringForces::allocate_parameter_tangent(const Domain& domain) const {
        ParameterTangent result{.stretch = allocate_spring_parameters(domain, topology.stretch.springs.size()), .bending = allocate_spring_parameters(domain, topology.bending.springs.size())};
        clear(domain, result.stretch);
        clear(domain, result.bending);
        return result;
    }

    MassSpringForces::ParameterAdjoint MassSpringForces::allocate_parameter_adjoint(const Domain& domain) const {
        ParameterAdjoint result{.stretch = allocate_spring_parameter_adjoint(domain, topology.stretch.springs.size()), .bending = allocate_spring_parameter_adjoint(domain, topology.bending.springs.size())};
        clear(domain, result.stretch);
        clear(domain, result.bending);
        return result;
    }

    MassSpringForces::Cache MassSpringForces::allocate_cache(const Domain& domain) const {
        return {.values = domain.allocate_vector_field()};
    }

    void MassSpringForces::forward(const Domain& domain, const State& state, const Control& control, const ScalarField& masses, const Parameters& parameters, Cache& cache) const {
        cuda_detail::force_forward(domain.stream, static_cast<std::uint32_t>(domain.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, cuda_detail::field(state.positions), cuda_detail::field(state.velocities), cuda_detail::field(control.external_forces), masses.values.data(), view(device_topology.stretch), view(parameters.stretch), view(device_topology.bending), view(parameters.bending), cuda_detail::field(cache.values));
    }

    void MassSpringForces::jvp(const Domain& domain, const State& state, const ScalarField&, const Parameters& parameters, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ScalarField& mass_tangent, const ParameterTangent& parameter_tangent, const Cache&) {
        Differentiation& workspace = *differentiation;
        cuda_detail::force_jvp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, cuda_detail::field(state.positions), cuda_detail::field(state.velocities), cuda_detail::field(control_tangent.external_forces), cuda_detail::field(state_tangent.positions), cuda_detail::field(state_tangent.velocities), mass_tangent.values.data(), view(device_topology.stretch), view(parameters.stretch), view(parameter_tangent.stretch), view(device_topology.bending), view(parameters.bending), view(parameter_tangent.bending), cuda_detail::field(workspace.tangent));
    }

    void MassSpringForces::vjp(const Domain& domain, const State& state, const ScalarField&, const Parameters& parameters, const VectorAdjointField& force_adjoint, StateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, ScalarAdjointField& mass_adjoint, ParameterAdjoint& parameter_adjoint) const {
        cuda_detail::force_state_vjp(domain.stream, static_cast<std::uint32_t>(domain.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, cuda_detail::field(state.positions), cuda_detail::field(state.velocities), cuda_detail::adjoint_field(force_adjoint), view(device_topology.stretch), view(parameters.stretch), view(device_topology.bending), view(parameters.bending), cuda_detail::adjoint_field(state_adjoint.positions), cuda_detail::adjoint_field(state_adjoint.velocities), cuda_detail::adjoint_field(control_adjoint.external_forces), mass_adjoint.values.data());
        cuda_detail::force_parameter_vjp(domain.stream, static_cast<std::uint32_t>(topology.stretch.springs.size()), cuda_detail::field(state.positions), cuda_detail::field(state.velocities), cuda_detail::adjoint_field(force_adjoint), view(device_topology.stretch), view(parameters.stretch), view(parameter_adjoint.stretch));
        cuda_detail::force_parameter_vjp(domain.stream, static_cast<std::uint32_t>(topology.bending.springs.size()), cuda_detail::field(state.positions), cuda_detail::field(state.velocities), cuda_detail::adjoint_field(force_adjoint), view(device_topology.bending), view(parameters.bending), view(parameter_adjoint.bending));
    }
} // namespace physica::deformables::cloth
