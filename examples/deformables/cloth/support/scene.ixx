module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.support.scene;

import std;
import physica.deformables.cloth.constraints.fixed_position;
import physica.deformables.cloth.model;
import physica.deformables.cloth.state;

export namespace physica::examples::cloth::support {
    struct Grid final {
        std::uint32_t rows;
        std::uint32_t columns;
        float width;
        float height;
    };

    struct MassSpringMaterial final {
        float mass;
        float stretch_stiffness;
        float stretch_damping;
        float bending_stiffness;
        float bending_damping;
    };

    [[nodiscard]] deformables::cloth::ModelConfiguration<float> create_grid(Grid grid);
    [[nodiscard]] deformables::cloth::constraints::FixedPositionConstraint::Configuration create_anchors(const deformables::cloth::ModelConfiguration<float>& configuration, std::span<const std::uint32_t> particles);

    template <class Parameters>
    void set_mass_spring_parameters(const ::cuda::stream_ref stream, Parameters& parameters, const MassSpringMaterial material) {
        const std::vector<float> masses(parameters.masses.values.size(), material.mass);
        const std::vector<float> stretch_stiffnesses(parameters.force.stretch.stiffnesses.values.size(), material.stretch_stiffness);
        const std::vector<float> stretch_dampings(parameters.force.stretch.dampings.values.size(), material.stretch_damping);
        const std::vector<float> bending_stiffnesses(parameters.force.bending.stiffnesses.values.size(), material.bending_stiffness);
        const std::vector<float> bending_dampings(parameters.force.bending.dampings.values.size(), material.bending_damping);
        ::cuda::copy_bytes(stream, masses, parameters.masses.values);
        ::cuda::copy_bytes(stream, stretch_stiffnesses, parameters.force.stretch.stiffnesses.values);
        ::cuda::copy_bytes(stream, stretch_dampings, parameters.force.stretch.dampings.values);
        ::cuda::copy_bytes(stream, bending_stiffnesses, parameters.force.bending.stiffnesses.values);
        ::cuda::copy_bytes(stream, bending_dampings, parameters.force.bending.dampings.values);
        stream.sync();
    }

    void initialize(const deformables::cloth::Model<float>& model, deformables::cloth::State<float>& current_state, deformables::cloth::State<float>& next_state, deformables::cloth::Control<float>& control);
} // namespace physica::examples::cloth::support
