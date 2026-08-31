module;

#include <physica/cuda.h>

export module physica.deformables.cloth.topology;

import std;
export import physica.simulation.field;

export namespace physica::deformables::cloth {
    struct Triangle final {
        std::uint32_t first;
        std::uint32_t second;
        std::uint32_t third;

        constexpr bool operator==(const Triangle&) const noexcept = default;
    };

    struct Edge final {
        std::uint32_t first;
        std::uint32_t second;

        constexpr bool operator==(const Edge&) const noexcept = default;
    };

    struct Hinge final {
        std::uint32_t edge;
        std::uint32_t edge_first;
        std::uint32_t edge_second;
        std::uint32_t first_opposite;
        std::uint32_t second_opposite;
        std::uint32_t first_triangle;
        std::uint32_t second_triangle;

        constexpr bool operator==(const Hinge&) const noexcept = default;
    };

    struct DeviceTriangleTopology final {
        simulation::ScalarField<std::uint32_t> first;
        simulation::ScalarField<std::uint32_t> second;
        simulation::ScalarField<std::uint32_t> third;
    };

    struct DeviceEdgeTopology final {
        simulation::ScalarField<std::uint32_t> first;
        simulation::ScalarField<std::uint32_t> second;
    };

    struct DeviceHingeTopology final {
        simulation::ScalarField<std::uint32_t> edge_first;
        simulation::ScalarField<std::uint32_t> edge_second;
        simulation::ScalarField<std::uint32_t> first_opposite;
        simulation::ScalarField<std::uint32_t> second_opposite;
    };

    struct DeviceAdjacency final {
        simulation::ScalarField<std::uint32_t> offsets;
        simulation::ScalarField<std::uint32_t> indices;
    };

    struct DeviceTopology final {
        DeviceTriangleTopology triangles;
        DeviceEdgeTopology edges;
        DeviceHingeTopology hinges;
        DeviceAdjacency vertex_triangles;
        DeviceAdjacency vertex_edges;
        DeviceAdjacency vertex_hinges;
    };

    struct Topology final {
        std::vector<Edge> edges;
        std::vector<Hinge> hinges;
        DeviceTopology device;

        Topology(std::span<const Triangle> triangles, std::size_t particle_count, ::cuda::stream_ref stream);

        Topology(const Topology&)            = delete;
        Topology& operator=(const Topology&) = delete;
        Topology(Topology&&)                 = delete;
        Topology& operator=(Topology&&)      = delete;

    private:
        struct HostAdjacency final {
            std::vector<std::uint32_t> offsets;
            std::vector<std::uint32_t> indices;
        };

        struct HostTopology final {
            std::vector<Edge> edges;
            std::vector<Hinge> hinges;
            HostAdjacency vertex_triangles;
            HostAdjacency vertex_edges;
            HostAdjacency vertex_hinges;
        };

        template <class Element, class Vertices>
        [[nodiscard]] static HostAdjacency build_adjacency(std::span<const Element> elements, std::size_t particle_count, Vertices vertices);
        [[nodiscard]] static HostTopology build(std::span<const Triangle> triangles, std::size_t particle_count);
        Topology(HostTopology topology, std::span<const Triangle> triangles, ::cuda::stream_ref stream);
    };
} // namespace physica::deformables::cloth
