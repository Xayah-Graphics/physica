module;

#include <physica/cuda.h>

export module physica.fluids.liquid.meshfree;

import std;
export import physica.field;

export namespace physica::fluids::liquid::meshfree {
    struct BoxBoundary final {
        AxisAlignedBox3<float> bounds;
        Vector3<float> velocity;
        bool no_slip{true};
    };

    struct BoundaryParticle final {
        Vector3<float> position;
        Vector3<float> velocity;
        float volume;
    };

    struct Configuration final {
        std::uint32_t particle_count;
        float time_step;
        float support_radius;
        float particle_radius;
        BoxBoundary boundary;
        std::vector<BoundaryParticle> boundary_particles;
    };

    struct BoundaryFields final {
        VectorField<float> positions;
        VectorField<float> velocities;
        ScalarField<float> volumes;
    };

    struct Model final {
        const Configuration configuration;
        FieldContext fields;
        BoundaryFields boundary;

        Model(Configuration configuration, ::cuda::stream_ref stream);

        Model(const Model&)            = delete;
        Model& operator=(const Model&) = delete;
        Model(Model&&)                 = delete;
        Model& operator=(Model&&)      = delete;
    };
}
