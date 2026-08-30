module;

#include <physica/cuda.h>

export module physica.deformables.cloth.model;

import std;
export import physica.simulation.field;

export namespace physica::deformables::cloth {
    struct Triangle final {
        std::uint32_t first;
        std::uint32_t second;
        std::uint32_t third;
    };

    struct ModelConfiguration final {
        std::vector<Vector3<float>> rest_positions;
        std::vector<Triangle> triangles;
    };

    struct Model final {
        const ModelConfiguration configuration;
        const ::cuda::stream_ref stream;
        const std::size_t particle_count;

        Model(ModelConfiguration configuration, ::cuda::stream_ref stream);

        Model(const Model&)            = delete;
        Model& operator=(const Model&) = delete;
        Model(Model&&)                 = delete;
        Model& operator=(Model&&)      = delete;
    };
}
