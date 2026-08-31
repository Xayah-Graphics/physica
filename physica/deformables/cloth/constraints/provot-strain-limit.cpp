module;

#include "provot-strain-limit-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.constraints.provot_strain_limit;

import std;

namespace physica::deformables::cloth::constraints {
    ProvotStrainLimitConstraint::ProvotStrainLimitConstraint(const Model<float>& model, Configuration configuration)
        : iteration_count(configuration.iteration_count), coloring(build_edge_coloring(model.topology.edges, model.particle_count)), colored_edges(model.stream, coloring.edges.size()), maximum_lengths(model.stream, model.topology.edges.size()), fixed_vertex_mask(model.stream, model.particle_count), fixed_positions(model.stream, model.particle_count) {
        std::vector<float> host_maximum_lengths(model.topology.edges.size());
        for (std::size_t edge_index = 0uz; edge_index < model.topology.edges.size(); ++edge_index) {
            const Edge edge                  = model.topology.edges[edge_index];
            host_maximum_lengths[edge_index] = configuration.maximum_stretch_ratio * length(model.configuration.rest_positions[edge.second] - model.configuration.rest_positions[edge.first]);
        }

        std::vector<std::uint32_t> host_fixed_vertex_mask(model.particle_count);
        std::vector<Vector3<float>> host_fixed_positions = model.configuration.rest_positions;
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            host_fixed_vertex_mask[fixed_vertex.particle] = 1u;
            host_fixed_positions[fixed_vertex.particle]    = fixed_vertex.position;
        }

        ::cuda::copy_bytes(model.stream, coloring.edges, colored_edges.values);
        ::cuda::copy_bytes(model.stream, host_maximum_lengths, maximum_lengths.values);
        ::cuda::copy_bytes(model.stream, host_fixed_vertex_mask, fixed_vertex_mask.values);
        simulation::upload(model.stream, host_fixed_positions, fixed_positions);
        model.stream.sync();
    }

    ProvotStrainLimitConstraint::Cache ProvotStrainLimitConstraint::allocate_cache(const Model<float>&) const {
        return {};
    }

    ProvotStrainLimitConstraint::Workspace ProvotStrainLimitConstraint::allocate_workspace(const Model<float>&) const {
        return {};
    }

    void ProvotStrainLimitConstraint::forward(const Model<float>& model, const simulation::VectorField<float>& previous_positions, const simulation::VectorField<float>&, const simulation::VectorField<float>& integrated_positions, const simulation::VectorField<float>&, const simulation::ScalarField<float>& masses, const float time_step, simulation::VectorField<float>& projected_positions, simulation::VectorField<float>& reconstructed_velocities, Cache&, Workspace&) const {
        kernels::provot_strain_limit_initialize(model.stream, static_cast<std::uint32_t>(model.particle_count), fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(integrated_positions), simulation::view(projected_positions));
        for (std::uint32_t iteration = 0u; iteration < iteration_count; ++iteration) {
            for (std::size_t color = 0uz; color + 1uz < coloring.offsets.size(); ++color) {
                const std::uint32_t color_offset = coloring.offsets[color];
                const std::uint32_t edge_count   = coloring.offsets[color + 1uz] - color_offset;
                kernels::provot_strain_limit_project(model.stream, edge_count, color_offset, colored_edges.values.data(), model.topology.device.edges.first.values.data(), model.topology.device.edges.second.values.data(), maximum_lengths.values.data(), fixed_vertex_mask.values.data(), masses.values.data(), simulation::view(projected_positions));
            }
        }
        kernels::provot_strain_limit_reconstruct_velocities(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(previous_positions), simulation::view(projected_positions), simulation::view(reconstructed_velocities));
    }

} // namespace physica::deformables::cloth::constraints
