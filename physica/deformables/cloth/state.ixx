module;

#include <physica/cuda.h>

export module physica.deformables.cloth.state;

import std;
export import physica.simulation.field;

export namespace physica::deformables::cloth {
    template <class Value>
    struct State final {
        simulation::VectorField<Value> positions;
        simulation::VectorField<Value> velocities;

        State(const ::cuda::stream_ref stream, const std::size_t particle_count) : positions(stream, particle_count), velocities(stream, particle_count) {}

        State(const State&)            = delete;
        State& operator=(const State&) = delete;
        State(State&&)                 = default;
        State& operator=(State&&)      = default;
    };

    template <class Value>
    struct Control final {
        simulation::VectorField<Value> external_forces;

        Control(const ::cuda::stream_ref stream, const std::size_t particle_count) : external_forces(stream, particle_count) {}

        Control(const Control&)            = delete;
        Control& operator=(const Control&) = delete;
        Control(Control&&)                 = default;
        Control& operator=(Control&&)      = default;
    };
} // namespace physica::deformables::cloth
