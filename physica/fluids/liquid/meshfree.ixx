module;

#include <physica/cuda.h>

export module physica.fluids.liquid.meshfree;

import std;
export import physica.simulation.field;

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
        simulation::VectorField<float> positions;
        simulation::VectorField<float> velocities;
        simulation::ScalarField<float> volumes;
    };

    struct Model final {
        const Configuration configuration;
        const ::cuda::stream_ref stream;
        BoundaryFields boundary;

        Model(Configuration configuration, ::cuda::stream_ref stream);

        Model(const Model&)            = delete;
        Model& operator=(const Model&) = delete;
        Model(Model&&)                 = delete;
        Model& operator=(Model&&)      = delete;
    };
} // namespace physica::fluids::liquid::meshfree
