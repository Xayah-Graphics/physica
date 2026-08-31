module;

#include "forward-euler-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.integrators.forward_euler;

import std;

namespace physica::deformables::cloth::integrators {
    ForwardEuler::ForwardEuler(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ForwardEuler::Cache ForwardEuler::allocate_cache(const Model<float>&) const {
        return {};
    }

    ForwardEuler::Workspace ForwardEuler::allocate_workspace(const Model<float>&) const {
        return {};
    }

    void ForwardEuler::forward(const Model<float>& model, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, const simulation::ScalarField<float>& masses, const simulation::VectorField<float>& forces, simulation::VectorField<float>& integrated_positions, simulation::VectorField<float>& integrated_velocities, Cache&, Workspace&) const {
        kernels::forward_euler_forward(model.stream, static_cast<std::uint32_t>(model.particle_count), configuration.time_step, simulation::view(positions), simulation::view(velocities), simulation::view(forces), masses.values.data(), simulation::view(integrated_positions), simulation::view(integrated_velocities));
    }
} // namespace physica::deformables::cloth::integrators
