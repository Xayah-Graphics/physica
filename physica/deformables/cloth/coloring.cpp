module physica.deformables.cloth.coloring;

import std;

namespace physica::deformables::cloth {
    EdgeColoring build_edge_coloring(const std::span<const Edge> edges, const std::size_t particle_count) {
        EdgeColoring result{.offsets = {0u}, .edges = {}};
        result.edges.reserve(edges.size());
        std::vector<std::uint32_t> remaining_edges(edges.size());
        for (std::size_t edge = 0uz; edge < remaining_edges.size(); ++edge) remaining_edges[edge] = static_cast<std::uint32_t>(edge);

        while (!remaining_edges.empty()) {
            std::vector<std::uint8_t> occupied_vertices(particle_count);
            std::vector<std::uint32_t> next_remaining_edges;
            next_remaining_edges.reserve(remaining_edges.size());
            for (const std::uint32_t edge_index : remaining_edges) {
                const Edge edge = edges[edge_index];
                if (occupied_vertices[edge.first] != 0u || occupied_vertices[edge.second] != 0u) {
                    next_remaining_edges.push_back(edge_index);
                    continue;
                }
                result.edges.push_back(edge_index);
                occupied_vertices[edge.first]  = 1u;
                occupied_vertices[edge.second] = 1u;
            }
            result.offsets.push_back(static_cast<std::uint32_t>(result.edges.size()));
            remaining_edges = std::move(next_remaining_edges);
        }
        return result;
    }

    TriangleColoring build_triangle_coloring(const std::span<const Triangle> triangles, const std::size_t particle_count) {
        TriangleColoring result{.offsets = {0u}, .triangles = {}};
        result.triangles.reserve(triangles.size());
        std::vector<std::uint32_t> remaining_triangles(triangles.size());
        for (std::size_t triangle = 0uz; triangle < remaining_triangles.size(); ++triangle) remaining_triangles[triangle] = static_cast<std::uint32_t>(triangle);

        while (!remaining_triangles.empty()) {
            std::vector<std::uint8_t> occupied_vertices(particle_count);
            std::vector<std::uint32_t> next_remaining_triangles;
            next_remaining_triangles.reserve(remaining_triangles.size());
            for (const std::uint32_t triangle_index : remaining_triangles) {
                const Triangle triangle = triangles[triangle_index];
                if (occupied_vertices[triangle.first] != 0u || occupied_vertices[triangle.second] != 0u || occupied_vertices[triangle.third] != 0u) {
                    next_remaining_triangles.push_back(triangle_index);
                    continue;
                }
                result.triangles.push_back(triangle_index);
                occupied_vertices[triangle.first]  = 1u;
                occupied_vertices[triangle.second] = 1u;
                occupied_vertices[triangle.third]  = 1u;
            }
            result.offsets.push_back(static_cast<std::uint32_t>(result.triangles.size()));
            remaining_triangles = std::move(next_remaining_triangles);
        }
        return result;
    }
} // namespace physica::deformables::cloth
