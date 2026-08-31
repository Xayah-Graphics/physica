module;

#include "mass-spring-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.operators.mass_spring;

import std;

namespace physica::deformables::cloth::operators {
    namespace {
        auto topology_view(const auto& topology) {
            return kernels::SpringTopologyView{
                .first   = topology.first.values.data(),
                .second  = topology.second.values.data(),
                .offsets = topology.offsets.values.data(),
                .indices = topology.indices.values.data(),
            };
        }

        kernels::SpringTopologyView stretch_topology_view(const Model<float>& model) {
            return {
                .first   = model.topology.device.edges.first.values.data(),
                .second  = model.topology.device.edges.second.values.data(),
                .offsets = model.topology.device.vertex_edges.offsets.values.data(),
                .indices = model.topology.device.vertex_edges.indices.values.data(),
            };
        }

        auto parameter_view(const auto& parameters) {
            return kernels::SpringParametersView{
                .stiffnesses  = parameters.stiffnesses.values.data(),
                .dampings     = parameters.dampings.values.data(),
                .rest_lengths = parameters.rest_lengths.values.data(),
            };
        }

        auto parameter_adjoint_view(auto& parameters) {
            return kernels::SpringParameterAdjointView{
                .stiffnesses  = parameters.stiffnesses.values.data(),
                .dampings     = parameters.dampings.values.data(),
                .rest_lengths = parameters.rest_lengths.values.data(),
            };
        }
    } // namespace

    MassSpringForce::MassSpringForce(const Model<float>& model, Configuration next_configuration) : configuration(std::move(next_configuration)), bending_topology(build_bending_topology(model)), device_bending_topology(allocate_device_topology(model, bending_topology)) {
        upload_topology(model, bending_topology, device_bending_topology);
        model.stream.sync();
    }

    MassSpringForce::Parameters MassSpringForce::allocate_parameters(const Model<float>& model) const {
        Parameters result{
            .stretch =
                {
                    .stiffnesses  = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
                    .dampings     = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
                    .rest_lengths = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
                },
            .bending =
                {
                    .stiffnesses  = simulation::ScalarField<float>(model.stream, bending_topology.springs.size()),
                    .dampings     = simulation::ScalarField<float>(model.stream, bending_topology.springs.size()),
                    .rest_lengths = simulation::ScalarField<float>(model.stream, bending_topology.springs.size()),
                },
        };
        simulation::clear(model.stream, result.stretch.stiffnesses);
        simulation::clear(model.stream, result.stretch.dampings);
        simulation::clear(model.stream, result.bending.stiffnesses);
        simulation::clear(model.stream, result.bending.dampings);

        std::vector<float> stretch_rest_lengths(model.topology.edges.size());
        std::vector<float> bending_rest_lengths(bending_topology.springs.size());
        for (std::size_t spring = 0uz; spring < model.topology.edges.size(); ++spring) stretch_rest_lengths[spring] = length(model.configuration.rest_positions[model.topology.edges[spring].second] - model.configuration.rest_positions[model.topology.edges[spring].first]);
        for (std::size_t spring = 0uz; spring < bending_topology.springs.size(); ++spring) bending_rest_lengths[spring] = bending_topology.springs[spring].rest_length;
        ::cuda::copy_bytes(model.stream, stretch_rest_lengths, result.stretch.rest_lengths.values);
        ::cuda::copy_bytes(model.stream, bending_rest_lengths, result.bending.rest_lengths.values);
        model.stream.sync();
        return result;
    }

    MassSpringForce::ParameterTangent MassSpringForce::allocate_parameter_tangent(const Model<float>& model) const {
        ParameterTangent result{
            .stretch =
                {
                    .stiffnesses  = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
                    .dampings     = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
                    .rest_lengths = simulation::ScalarField<float>(model.stream, model.topology.edges.size()),
                },
            .bending =
                {
                    .stiffnesses  = simulation::ScalarField<float>(model.stream, bending_topology.springs.size()),
                    .dampings     = simulation::ScalarField<float>(model.stream, bending_topology.springs.size()),
                    .rest_lengths = simulation::ScalarField<float>(model.stream, bending_topology.springs.size()),
                },
        };
        simulation::clear(model.stream, result.stretch.stiffnesses);
        simulation::clear(model.stream, result.stretch.dampings);
        simulation::clear(model.stream, result.stretch.rest_lengths);
        simulation::clear(model.stream, result.bending.stiffnesses);
        simulation::clear(model.stream, result.bending.dampings);
        simulation::clear(model.stream, result.bending.rest_lengths);
        return result;
    }

    MassSpringForce::ParameterAdjoint MassSpringForce::allocate_parameter_adjoint(const Model<float>& model) const {
        ParameterAdjoint result{
            .stretch =
                {
                    .stiffnesses  = simulation::ScalarField<double>(model.stream, model.topology.edges.size()),
                    .dampings     = simulation::ScalarField<double>(model.stream, model.topology.edges.size()),
                    .rest_lengths = simulation::ScalarField<double>(model.stream, model.topology.edges.size()),
                },
            .bending =
                {
                    .stiffnesses  = simulation::ScalarField<double>(model.stream, bending_topology.springs.size()),
                    .dampings     = simulation::ScalarField<double>(model.stream, bending_topology.springs.size()),
                    .rest_lengths = simulation::ScalarField<double>(model.stream, bending_topology.springs.size()),
                },
        };
        simulation::clear(model.stream, result.stretch.stiffnesses);
        simulation::clear(model.stream, result.stretch.dampings);
        simulation::clear(model.stream, result.stretch.rest_lengths);
        simulation::clear(model.stream, result.bending.stiffnesses);
        simulation::clear(model.stream, result.bending.dampings);
        simulation::clear(model.stream, result.bending.rest_lengths);
        return result;
    }

    MassSpringForce::Cache MassSpringForce::allocate_cache(const Model<float>&) const {
        return {};
    }

    MassSpringForce::Workspace MassSpringForce::allocate_workspace(const Model<float>&) const {
        return {};
    }

    MassSpringForce::TangentWorkspace MassSpringForce::allocate_tangent_workspace(const Model<float>&) const {
        return {};
    }

    MassSpringForce::AdjointWorkspace MassSpringForce::allocate_adjoint_workspace(const Model<float>&) const {
        return {};
    }

    void MassSpringForce::forward(const Model<float>& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>& external_forces, const simulation::ScalarField<float>& masses, const Parameters& parameters, simulation::VectorField<float>& forces, Cache&, Workspace&) const {
        kernels::force_forward(model.stream, static_cast<std::uint32_t>(model.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, simulation::view(positions), simulation::view(velocities), simulation::view(external_forces), masses.values.data(), stretch_topology_view(model), parameter_view(parameters.stretch), topology_view(device_bending_topology), parameter_view(parameters.bending), simulation::view(forces));
    }

    void MassSpringForce::jvp(const Model<float>& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>&, const simulation::ScalarField<float>&, const Parameters& parameters, const simulation::VectorField<float>&, const Cache&, const simulation::VectorField<float>& position_tangent, const simulation::VectorField<float>& velocity_tangent, const simulation::VectorField<float>& external_force_tangent, const simulation::ScalarField<float>& mass_tangent, const ParameterTangent& parameter_tangent, simulation::VectorField<float>& force_tangent, TangentWorkspace&) const {
        kernels::force_jvp(model.stream, static_cast<std::uint32_t>(model.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, simulation::view(positions), simulation::view(velocities), simulation::view(external_force_tangent), simulation::view(position_tangent), simulation::view(velocity_tangent), mass_tangent.values.data(), stretch_topology_view(model), parameter_view(parameters.stretch), parameter_view(parameter_tangent.stretch), topology_view(device_bending_topology), parameter_view(parameters.bending), parameter_view(parameter_tangent.bending), simulation::view(force_tangent));
    }

    void MassSpringForce::vjp(const Model<float>& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::VectorField<float>&, const simulation::ScalarField<float>&, const Parameters& parameters, const simulation::VectorField<float>&, const Cache&, const simulation::VectorField<double>& force_adjoint, simulation::VectorField<double>& position_adjoint, simulation::VectorField<double>& velocity_adjoint, simulation::VectorField<double>& external_force_adjoint, simulation::ScalarField<double>& mass_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace&) const {
        kernels::force_state_vjp(model.stream, static_cast<std::uint32_t>(model.particle_count), {.x = configuration.gravity.x, .y = configuration.gravity.y, .z = configuration.gravity.z}, simulation::view(positions), simulation::view(velocities), simulation::view(force_adjoint), stretch_topology_view(model), parameter_view(parameters.stretch), topology_view(device_bending_topology), parameter_view(parameters.bending), simulation::view(position_adjoint), simulation::view(velocity_adjoint), simulation::view(external_force_adjoint), mass_adjoint.values.data());
        kernels::force_parameter_vjp(model.stream, static_cast<std::uint32_t>(model.topology.edges.size()), simulation::view(positions), simulation::view(velocities), simulation::view(force_adjoint), stretch_topology_view(model), parameter_view(parameters.stretch), parameter_adjoint_view(parameter_adjoint.stretch));
        kernels::force_parameter_vjp(model.stream, static_cast<std::uint32_t>(bending_topology.springs.size()), simulation::view(positions), simulation::view(velocities), simulation::view(force_adjoint), topology_view(device_bending_topology), parameter_view(parameters.bending), parameter_adjoint_view(parameter_adjoint.bending));
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

    MassSpringForce::SpringTopology MassSpringForce::build_bending_topology(const Model<float>& model) {
        SpringTopology result{};
        result.springs.reserve(model.topology.hinges.size());
        for (const Hinge hinge : model.topology.hinges) {
            const std::uint32_t first  = hinge.first_opposite < hinge.second_opposite ? hinge.first_opposite : hinge.second_opposite;
            const std::uint32_t second = hinge.first_opposite < hinge.second_opposite ? hinge.second_opposite : hinge.first_opposite;
            result.springs.push_back({.first = first, .second = second, .rest_length = length(model.configuration.rest_positions[second] - model.configuration.rest_positions[first])});
        }
        build_adjacency(result, model.particle_count);
        return result;
    }

    MassSpringForce::DeviceSpringTopology MassSpringForce::allocate_device_topology(const Model<float>& model, const SpringTopology& spring_topology) {
        return {
            .first   = simulation::ScalarField<std::uint32_t>(model.stream, spring_topology.springs.size()),
            .second  = simulation::ScalarField<std::uint32_t>(model.stream, spring_topology.springs.size()),
            .offsets = simulation::ScalarField<std::uint32_t>(model.stream, spring_topology.offsets.size()),
            .indices = simulation::ScalarField<std::uint32_t>(model.stream, spring_topology.indices.size()),
        };
    }

    void MassSpringForce::upload_topology(const Model<float>& model, const SpringTopology& spring_topology, DeviceSpringTopology& destination) {
        std::vector<std::uint32_t> first(spring_topology.springs.size());
        std::vector<std::uint32_t> second(spring_topology.springs.size());
        for (std::size_t spring = 0uz; spring < spring_topology.springs.size(); ++spring) {
            first[spring]  = spring_topology.springs[spring].first;
            second[spring] = spring_topology.springs[spring].second;
        }
        ::cuda::copy_bytes(model.stream, first, destination.first.values);
        ::cuda::copy_bytes(model.stream, second, destination.second.values);
        ::cuda::copy_bytes(model.stream, spring_topology.offsets, destination.offsets.values);
        ::cuda::copy_bytes(model.stream, spring_topology.indices, destination.indices.values);
    }
} // namespace physica::deformables::cloth::operators
