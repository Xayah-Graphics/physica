module;

#include <physica/cuda.h>

export module physica.fluids.liquid.solvers.pic.model;

import std;
export import physica.fluids.grid;

export namespace physica::fluids::liquid::solvers::pic {
    struct ModelConfiguration final {
        grid::Configuration grid;
        std::uint32_t maximum_particle_count;
        float particle_radius;
        bool no_slip{true};
    };

    struct Model final {
        grid::Grid grid;
        const std::uint32_t maximum_particle_count;
        const float particle_radius;
        const bool no_slip;

        Model(ModelConfiguration next_configuration, const ::cuda::stream_ref stream)
            : grid(std::move(next_configuration.grid), stream), maximum_particle_count(next_configuration.maximum_particle_count), particle_radius(next_configuration.particle_radius), no_slip(next_configuration.no_slip) {}

        Model(const Model&)            = delete;
        Model& operator=(const Model&) = delete;
        Model(Model&&)                 = delete;
        Model& operator=(Model&&)      = delete;
    };
} // namespace physica::fluids::liquid::solvers::pic
