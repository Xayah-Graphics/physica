module;

#include <physica/cuda.h>

module physica.deformables.cloth.topology;

import std;

namespace physica::deformables::cloth {
    namespace {
        struct IncidentTriangle final {
            std::uint32_t edge_first;
            std::uint32_t edge_second;
            std::uint32_t opposite;
            std::uint32_t triangle;
        };
    } // namespace

    Topology::Topology(const std::span<const Triangle> triangles, const std::size_t particle_count, const ::cuda::stream_ref stream) : Topology(build(triangles, particle_count), triangles, stream) {}

    template <class Element, class Vertices>
    Topology::HostAdjacency Topology::build_adjacency(const std::span<const Element> elements, const std::size_t particle_count, Vertices vertices) {
        HostAdjacency result{.offsets = std::vector<std::uint32_t>(particle_count + 1uz), .indices = {}};
        for (const Element& element : elements)
            for (const std::uint32_t vertex : vertices(element)) ++result.offsets[vertex + 1u];
        for (std::size_t particle = 1uz; particle < result.offsets.size(); ++particle) result.offsets[particle] += result.offsets[particle - 1uz];
        result.indices.resize(result.offsets.back());
        std::vector<std::uint32_t> cursors = result.offsets;
        for (std::uint32_t index = 0u; index < elements.size(); ++index)
            for (const std::uint32_t vertex : vertices(elements[index])) result.indices[cursors[vertex]++] = index;
        return result;
    }

    Topology::HostTopology Topology::build(const std::span<const Triangle> triangles, const std::size_t particle_count) {
        std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<IncidentTriangle>> incidents;
        for (std::uint32_t triangle_index = 0u; triangle_index < triangles.size(); ++triangle_index) {
            const Triangle triangle = triangles[triangle_index];
            const std::array vertices{triangle.first, triangle.second, triangle.third};
            for (std::size_t edge = 0uz; edge < vertices.size(); ++edge) {
                const std::uint32_t edge_first  = vertices[edge];
                const std::uint32_t edge_second = vertices[(edge + 1uz) % vertices.size()];
                const std::uint32_t first       = edge_first < edge_second ? edge_first : edge_second;
                const std::uint32_t second      = edge_first < edge_second ? edge_second : edge_first;
                incidents[{first, second}].push_back({.edge_first = edge_first, .edge_second = edge_second, .opposite = vertices[(edge + 2uz) % vertices.size()], .triangle = triangle_index});
            }
        }

        HostTopology result{};
        result.edges.reserve(incidents.size());
        result.hinges.reserve(incidents.size());
        for (const auto& [edge, edge_incidents] : incidents) {
            const std::uint32_t edge_index = static_cast<std::uint32_t>(result.edges.size());
            result.edges.push_back({.first = edge.first, .second = edge.second});
            if (edge_incidents.size() == 1uz) continue;
            result.hinges.push_back({
                .edge            = edge_index,
                .edge_first      = edge_incidents[0].edge_first,
                .edge_second     = edge_incidents[0].edge_second,
                .first_opposite  = edge_incidents[0].opposite,
                .second_opposite = edge_incidents[1].opposite,
                .first_triangle  = edge_incidents[0].triangle,
                .second_triangle = edge_incidents[1].triangle,
            });
        }

        result.vertex_triangles = build_adjacency<Triangle>(triangles, particle_count, [](const Triangle triangle) { return std::array{triangle.first, triangle.second, triangle.third}; });
        result.vertex_edges     = build_adjacency<Edge>(result.edges, particle_count, [](const Edge edge) { return std::array{edge.first, edge.second}; });
        result.vertex_hinges    = build_adjacency<Hinge>(result.hinges, particle_count, [](const Hinge hinge) { return std::array{hinge.edge_first, hinge.edge_second, hinge.first_opposite, hinge.second_opposite}; });
        return result;
    }

    Topology::Topology(HostTopology topology, const std::span<const Triangle> triangles, const ::cuda::stream_ref stream)
        : edges(std::move(topology.edges)), hinges(std::move(topology.hinges)),
          device{
              .triangles =
                  {
                      .first  = simulation::ScalarField<std::uint32_t>(stream, triangles.size()),
                      .second = simulation::ScalarField<std::uint32_t>(stream, triangles.size()),
                      .third  = simulation::ScalarField<std::uint32_t>(stream, triangles.size()),
                  },
              .edges =
                  {
                      .first  = simulation::ScalarField<std::uint32_t>(stream, edges.size()),
                      .second = simulation::ScalarField<std::uint32_t>(stream, edges.size()),
                  },
              .hinges =
                  {
                      .edge_first       = simulation::ScalarField<std::uint32_t>(stream, hinges.size()),
                      .edge_second      = simulation::ScalarField<std::uint32_t>(stream, hinges.size()),
                      .first_opposite   = simulation::ScalarField<std::uint32_t>(stream, hinges.size()),
                      .second_opposite  = simulation::ScalarField<std::uint32_t>(stream, hinges.size()),
                  },
              .vertex_triangles =
                  {
                      .offsets = simulation::ScalarField<std::uint32_t>(stream, topology.vertex_triangles.offsets.size()),
                      .indices = simulation::ScalarField<std::uint32_t>(stream, topology.vertex_triangles.indices.size()),
                  },
              .vertex_edges =
                  {
                      .offsets = simulation::ScalarField<std::uint32_t>(stream, topology.vertex_edges.offsets.size()),
                      .indices = simulation::ScalarField<std::uint32_t>(stream, topology.vertex_edges.indices.size()),
                  },
              .vertex_hinges =
                  {
                      .offsets = simulation::ScalarField<std::uint32_t>(stream, topology.vertex_hinges.offsets.size()),
                      .indices = simulation::ScalarField<std::uint32_t>(stream, topology.vertex_hinges.indices.size()),
                  },
          } {
        std::vector<std::uint32_t> triangle_first(triangles.size());
        std::vector<std::uint32_t> triangle_second(triangles.size());
        std::vector<std::uint32_t> triangle_third(triangles.size());
        for (std::size_t index = 0uz; index < triangles.size(); ++index) {
            triangle_first[index]  = triangles[index].first;
            triangle_second[index] = triangles[index].second;
            triangle_third[index]  = triangles[index].third;
        }
        std::vector<std::uint32_t> edge_first(edges.size());
        std::vector<std::uint32_t> edge_second(edges.size());
        for (std::size_t index = 0uz; index < edges.size(); ++index) {
            edge_first[index]  = edges[index].first;
            edge_second[index] = edges[index].second;
        }
        std::vector<std::uint32_t> hinge_edge_first(hinges.size());
        std::vector<std::uint32_t> hinge_edge_second(hinges.size());
        std::vector<std::uint32_t> hinge_first_opposite(hinges.size());
        std::vector<std::uint32_t> hinge_second_opposite(hinges.size());
        for (std::size_t index = 0uz; index < hinges.size(); ++index) {
            hinge_edge_first[index]      = hinges[index].edge_first;
            hinge_edge_second[index]     = hinges[index].edge_second;
            hinge_first_opposite[index]  = hinges[index].first_opposite;
            hinge_second_opposite[index] = hinges[index].second_opposite;
        }

        ::cuda::copy_bytes(stream, triangle_first, device.triangles.first.values);
        ::cuda::copy_bytes(stream, triangle_second, device.triangles.second.values);
        ::cuda::copy_bytes(stream, triangle_third, device.triangles.third.values);
        ::cuda::copy_bytes(stream, edge_first, device.edges.first.values);
        ::cuda::copy_bytes(stream, edge_second, device.edges.second.values);
        ::cuda::copy_bytes(stream, hinge_edge_first, device.hinges.edge_first.values);
        ::cuda::copy_bytes(stream, hinge_edge_second, device.hinges.edge_second.values);
        ::cuda::copy_bytes(stream, hinge_first_opposite, device.hinges.first_opposite.values);
        ::cuda::copy_bytes(stream, hinge_second_opposite, device.hinges.second_opposite.values);
        ::cuda::copy_bytes(stream, topology.vertex_triangles.offsets, device.vertex_triangles.offsets.values);
        ::cuda::copy_bytes(stream, topology.vertex_triangles.indices, device.vertex_triangles.indices.values);
        ::cuda::copy_bytes(stream, topology.vertex_edges.offsets, device.vertex_edges.offsets.values);
        ::cuda::copy_bytes(stream, topology.vertex_edges.indices, device.vertex_edges.indices.values);
        ::cuda::copy_bytes(stream, topology.vertex_hinges.offsets, device.vertex_hinges.offsets.values);
        ::cuda::copy_bytes(stream, topology.vertex_hinges.indices, device.vertex_hinges.indices.values);
        stream.sync();
    }
} // namespace physica::deformables::cloth
