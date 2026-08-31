module;

#include "analytic-collision-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.constraints.analytic_collision;

import std;

namespace physica::deformables::cloth::constraints {
    AnalyticCollisionConstraint::AnalyticCollisionConstraint(const Model<float>&, Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    AnalyticCollisionConstraint::Cache AnalyticCollisionConstraint::allocate_cache(const Model<float>&) const {
        return {};
    }

    AnalyticCollisionConstraint::Workspace AnalyticCollisionConstraint::allocate_workspace(const Model<float>&) const {
        return {};
    }

    void AnalyticCollisionConstraint::forward(const Model<float>& model, const simulation::VectorField<float>&, const simulation::VectorField<float>&, const simulation::VectorField<float>& integrated_positions, const simulation::VectorField<float>& integrated_velocities, const simulation::ScalarField<float>&, const float, simulation::VectorField<float>& constrained_positions, simulation::VectorField<float>& constrained_velocities, Cache&, Workspace&) const {
        kernels::analytic_collision_initialize(model.stream, static_cast<std::uint32_t>(model.particle_count), simulation::view(integrated_positions), simulation::view(integrated_velocities), simulation::view(constrained_positions), simulation::view(constrained_velocities));
        for (const Plane plane : configuration.planes) kernels::analytic_collision_plane(model.stream, static_cast<std::uint32_t>(model.particle_count), plane.normal, plane.offset, configuration.thickness, plane.restitution, plane.friction, simulation::view(constrained_positions), simulation::view(constrained_velocities));
        for (const Sphere sphere : configuration.spheres) kernels::analytic_collision_sphere(model.stream, static_cast<std::uint32_t>(model.particle_count), sphere.center, sphere.radius, configuration.thickness, sphere.restitution, sphere.friction, simulation::view(constrained_positions), simulation::view(constrained_velocities));
    }
} // namespace physica::deformables::cloth::constraints
