module;

#include "strain-based-dynamics-kernels.h"
#include "../position-dynamics-kernels.h"
#include <physica/cuda.h>
#include <simulation/field/device.cuh>

module physica.deformables.cloth.solvers.strain_based_dynamics;

import std;

namespace physica::deformables::cloth::solvers::strain_based_dynamics {
    Solver::Solver(const Model<float>& model, Configuration configuration)
        : time_step(configuration.time_step), iteration_count(configuration.iteration_count), gravity(configuration.gravity), stretch_stiffness_u(configuration.stretch_stiffness_u), stretch_stiffness_v(configuration.stretch_stiffness_v), shear_stiffness(configuration.shear_stiffness), coloring(build_triangle_coloring(model.configuration.triangles, model.particle_count)), colored_triangles(model.stream, coloring.triangles.size()), inverse_rest_00(model.stream, model.configuration.triangles.size()), inverse_rest_01(model.stream, model.configuration.triangles.size()), inverse_rest_10(model.stream, model.configuration.triangles.size()), inverse_rest_11(model.stream, model.configuration.triangles.size()), fixed_vertex_mask(model.stream, model.particle_count), fixed_positions(model.stream, model.particle_count) {
        std::vector<float> host_inverse_rest_00(model.configuration.triangles.size());
        std::vector<float> host_inverse_rest_01(model.configuration.triangles.size());
        std::vector<float> host_inverse_rest_10(model.configuration.triangles.size());
        std::vector<float> host_inverse_rest_11(model.configuration.triangles.size());
        for (std::size_t triangle = 0uz; triangle < model.configuration.triangles.size(); ++triangle) {
            const TriangleMaterialCoordinates<float> coordinates = model.configuration.material_coordinates[triangle];
            const float rest_00 = coordinates.second.u - coordinates.first.u;
            const float rest_01 = coordinates.third.u - coordinates.first.u;
            const float rest_10 = coordinates.second.v - coordinates.first.v;
            const float rest_11 = coordinates.third.v - coordinates.first.v;
            const float inverse_determinant = 1.0F / (rest_00 * rest_11 - rest_01 * rest_10);
            host_inverse_rest_00[triangle] = rest_11 * inverse_determinant;
            host_inverse_rest_01[triangle] = -rest_01 * inverse_determinant;
            host_inverse_rest_10[triangle] = -rest_10 * inverse_determinant;
            host_inverse_rest_11[triangle] = rest_00 * inverse_determinant;
        }

        std::vector<std::uint32_t> host_fixed_vertex_mask(model.particle_count);
        std::vector<Vector3<float>> host_fixed_positions = model.configuration.rest_positions;
        for (const FixedVertex fixed_vertex : configuration.fixed_vertices) {
            host_fixed_vertex_mask[fixed_vertex.particle] = 1u;
            host_fixed_positions[fixed_vertex.particle]    = fixed_vertex.position;
        }

        ::cuda::copy_bytes(model.stream, coloring.triangles, colored_triangles.values);
        ::cuda::copy_bytes(model.stream, host_inverse_rest_00, inverse_rest_00.values);
        ::cuda::copy_bytes(model.stream, host_inverse_rest_01, inverse_rest_01.values);
        ::cuda::copy_bytes(model.stream, host_inverse_rest_10, inverse_rest_10.values);
        ::cuda::copy_bytes(model.stream, host_inverse_rest_11, inverse_rest_11.values);
        ::cuda::copy_bytes(model.stream, host_fixed_vertex_mask, fixed_vertex_mask.values);
        simulation::upload(model.stream, host_fixed_positions, fixed_positions);
        model.stream.sync();
    }

    State<float> Solver::allocate_state(const Model<float>& model) const {
        State<float> result(model.stream, model.particle_count);
        simulation::clear(model.stream, result.positions);
        simulation::clear(model.stream, result.velocities);
        return result;
    }

    Control<float> Solver::allocate_control(const Model<float>& model) const {
        Control<float> result(model.stream, model.particle_count);
        simulation::clear(model.stream, result.external_forces);
        return result;
    }

    Solver::Parameters Solver::allocate_parameters(const Model<float>& model) const {
        Parameters result{.masses = simulation::ScalarField<float>(model.stream, model.particle_count)};
        simulation::clear(model.stream, result.masses);
        return result;
    }

    Solver::StepCache Solver::allocate_step_cache(const Model<float>&) const {
        return {};
    }

    Solver::Workspace Solver::allocate_workspace(const Model<float>&) const {
        return {};
    }

    void Solver::forward(const Model<float>& model, const State<float>& state, const Control<float>& control, const Parameters& parameters, State<float>& next_state, StepCache&, Workspace&) const {
        position_dynamics::kernels::predict(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, gravity, fixed_vertex_mask.values.data(), simulation::view(fixed_positions), simulation::view(state.positions), simulation::view(state.velocities), simulation::view(control.external_forces), parameters.masses.values.data(), simulation::view(next_state.positions));
        for (std::uint32_t iteration = 0u; iteration < iteration_count; ++iteration) {
            for (std::size_t color = 0uz; color + 1uz < coloring.offsets.size(); ++color) {
                const std::uint32_t color_offset   = coloring.offsets[color];
                const std::uint32_t triangle_count = coloring.offsets[color + 1uz] - color_offset;
                kernels::project_strain(model.stream, triangle_count, color_offset, colored_triangles.values.data(), model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), inverse_rest_00.values.data(), inverse_rest_01.values.data(), inverse_rest_10.values.data(), inverse_rest_11.values.data(), fixed_vertex_mask.values.data(), parameters.masses.values.data(), stretch_stiffness_u, stretch_stiffness_v, shear_stiffness, simulation::view(next_state.positions));
            }
        }
        position_dynamics::kernels::reconstruct_velocities(model.stream, static_cast<std::uint32_t>(model.particle_count), time_step, simulation::view(state.positions), simulation::view(next_state.positions), simulation::view(next_state.velocities));
    }
} // namespace physica::deformables::cloth::solvers::strain_based_dynamics
