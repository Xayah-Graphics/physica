export module physica.deformables.cloth.coloring;

import std;
import physica.deformables.cloth.topology;

export namespace physica::deformables::cloth {
    struct EdgeColoring final {
        std::vector<std::uint32_t> offsets;
        std::vector<std::uint32_t> edges;
    };

    struct TriangleColoring final {
        std::vector<std::uint32_t> offsets;
        std::vector<std::uint32_t> triangles;
    };

    [[nodiscard]] EdgeColoring build_edge_coloring(std::span<const Edge> edges, std::size_t particle_count);
    [[nodiscard]] TriangleColoring build_triangle_coloring(std::span<const Triangle> triangles, std::size_t particle_count);
} // namespace physica::deformables::cloth
