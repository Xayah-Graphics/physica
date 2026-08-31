module;

#include <physica/cuda.h>

module physica.example.deformables.cloth.support.scene;

import std;

namespace physica::examples::cloth::support {
    deformables::cloth::ModelConfiguration<float> create_grid(const Grid grid) {
        deformables::cloth::ModelConfiguration<float> result{
            .rest_positions       = std::vector<Vector3<float>>(static_cast<std::size_t>(grid.rows) * grid.columns),
            .triangles            = {},
            .material_coordinates = {},
        };
        std::vector<deformables::cloth::MaterialCoordinate<float>> vertex_material_coordinates(result.rest_positions.size());
        const float spacing_x = grid.width / static_cast<float>(grid.columns - 1u);
        const float spacing_y = grid.height / static_cast<float>(grid.rows - 1u);
        for (std::uint32_t row = 0u; row < grid.rows; ++row) {
            for (std::uint32_t column = 0u; column < grid.columns; ++column) {
                const std::uint32_t particle    = row * grid.columns + column;
                result.rest_positions[particle] = {.x = static_cast<float>(column) * spacing_x, .y = -static_cast<float>(row) * spacing_y, .z = 0.0F};
                vertex_material_coordinates[particle] = {.u = static_cast<float>(column) * spacing_x, .v = static_cast<float>(row) * spacing_y};
            }
        }
        result.triangles.reserve(static_cast<std::size_t>(grid.rows - 1u) * (grid.columns - 1u) * 2uz);
        result.material_coordinates.reserve(result.triangles.capacity());
        const auto append_triangle = [&](const std::uint32_t first, const std::uint32_t second, const std::uint32_t third) {
            result.triangles.push_back({.first = first, .second = second, .third = third});
            result.material_coordinates.push_back({.first = vertex_material_coordinates[first], .second = vertex_material_coordinates[second], .third = vertex_material_coordinates[third]});
        };
        for (std::uint32_t row = 0u; row + 1u < grid.rows; ++row) {
            for (std::uint32_t column = 0u; column + 1u < grid.columns; ++column) {
                const std::uint32_t top_left     = row * grid.columns + column;
                const std::uint32_t top_right    = top_left + 1u;
                const std::uint32_t bottom_left  = top_left + grid.columns;
                const std::uint32_t bottom_right = bottom_left + 1u;
                if ((row + column) % 2u == 0u) {
                    append_triangle(top_left, top_right, bottom_right);
                    append_triangle(top_left, bottom_right, bottom_left);
                } else {
                    append_triangle(top_left, top_right, bottom_left);
                    append_triangle(top_right, bottom_right, bottom_left);
                }
            }
        }
        return result;
    }

    deformables::cloth::constraints::FixedPositionConstraint::Configuration create_anchors(const deformables::cloth::ModelConfiguration<float>& configuration, const std::span<const std::uint32_t> particles) {
        deformables::cloth::constraints::FixedPositionConstraint::Configuration result{.anchors = {}};
        result.anchors.reserve(particles.size());
        for (const std::uint32_t particle : particles) result.anchors.push_back({.particle = particle, .position = configuration.rest_positions[particle]});
        return result;
    }

    void initialize(const deformables::cloth::Model<float>& model, deformables::cloth::State<float>& current_state, deformables::cloth::State<float>& next_state, deformables::cloth::Control<float>& control) {
        simulation::upload(model.stream, model.configuration.rest_positions, current_state.positions);
        simulation::clear(model.stream, current_state.velocities);
        simulation::upload(model.stream, model.configuration.rest_positions, next_state.positions);
        simulation::clear(model.stream, next_state.velocities);
        simulation::clear(model.stream, control.external_forces);
    }
} // namespace physica::examples::cloth::support
