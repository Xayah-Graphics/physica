module;

#include <physica/cuda.h>

export module physica.deformables.cloth.model;

import std;
export import physica.deformables.cloth.topology;

export namespace physica::deformables::cloth {
    template <class Value>
    struct MaterialCoordinate final {
        Value u;
        Value v;
    };

    template <class Value>
    struct TriangleMaterialCoordinates final {
        MaterialCoordinate<Value> first;
        MaterialCoordinate<Value> second;
        MaterialCoordinate<Value> third;
    };

    template <class Value>
    struct ModelConfiguration final {
        std::vector<Vector3<Value>> rest_positions;
        std::vector<Triangle> triangles;
        std::vector<TriangleMaterialCoordinates<Value>> material_coordinates;
    };

    template <class Value>
    struct Model final {
        const ModelConfiguration<Value> configuration;
        const ::cuda::stream_ref stream;
        const std::size_t particle_count;
        const Topology topology;

        Model(ModelConfiguration<Value> configuration, const ::cuda::stream_ref stream) : configuration(std::move(configuration)), stream(stream), particle_count(this->configuration.rest_positions.size()), topology(this->configuration.triangles, particle_count, stream) {}

        Model(const Model&)            = delete;
        Model& operator=(const Model&) = delete;
        Model(Model&&)                 = delete;
        Model& operator=(Model&&)      = delete;
    };
} // namespace physica::deformables::cloth
