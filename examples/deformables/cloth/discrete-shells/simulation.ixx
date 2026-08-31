module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.discrete_shells;

import std;
import physica.deformables.cloth.model;
import physica.deformables.cloth.solvers.discrete_shells;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::discrete_shells {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        std::uint32_t curved_rest_hinge_count;
        float maximum_absolute_rest_dihedral;
        float maximum_relative_edge_length_error;
        float maximum_relative_triangle_area_error;
        float maximum_absolute_dihedral_delta;
        double total_membrane_energy;
        double total_bending_energy;
        double total_damping_potential;
        float regularization_shift;
        float accepted_step_size;
        std::uint32_t accepted_line_search_candidate;
        float maximum_fixed_position_error;
        float maximum_position_magnitude;
        float maximum_velocity_magnitude;
        float probe_displacement;
        Vector3<float> probe_position;
        Vector3<float> probe_velocity;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows                        = 5u;
        inline static constexpr std::uint32_t columns                     = 16u;
        inline static constexpr float beam_length                         = 1.6F;
        inline static constexpr float beam_width                          = 0.3F;
        inline static constexpr float crease_depth                        = 0.08F;
        inline static constexpr float time_step                           = 1.0F / 240.0F;
        inline static constexpr std::uint32_t frame_count                 = 120u;
        inline static constexpr std::uint32_t newton_iteration_count      = 6u;
        inline static constexpr std::uint32_t pcg_iteration_count         = 128u;
        inline static constexpr std::uint32_t line_search_candidate_count = 14u;
        inline static constexpr float surface_density                     = 0.2F;
        inline static constexpr float length_stiffness                    = 2.0e4F;
        inline static constexpr float area_stiffness                      = 2.0e4F;
        inline static constexpr float bending_stiffness                   = 0.45F;
        inline static constexpr float bending_damping                     = 0.015F;
        inline static constexpr float gravity_z                           = -9.81F;
        inline static constexpr float initial_tip_velocity                = -0.04F;
        inline static constexpr std::uint32_t probe_particle              = (rows / 2u) * columns + columns - 1u;
        inline static constexpr std::array<std::uint32_t, rows> fixed_particles{0u, columns, 2u * columns, 3u * columns, 4u * columns};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::discrete_shells::Solver solver;
        deformables::cloth::solvers::discrete_shells::State current_state;
        deformables::cloth::solvers::discrete_shells::State next_state;
        deformables::cloth::Control<float> control;
        decltype(solver)::Parameters parameters;
        decltype(solver)::StepCache step_cache;
        decltype(solver)::Workspace workspace;

    public:
        Simulation();

        Simulation(const Simulation&)            = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&)                 = delete;
        Simulation& operator=(Simulation&&)      = delete;

        [[nodiscard]] Summary run();

    private:
        [[nodiscard]] static deformables::cloth::ModelConfiguration<float> create_configuration();
        void step();
        [[nodiscard]] Summary summarize();
    };

    Simulation::Simulation()
        : stream{::cuda::devices[0]},
          model(create_configuration(), stream),
          solver(
              model,
              {
                  .time_step = time_step,
                  .newton_iteration_count = newton_iteration_count,
                  .pcg_iteration_count = pcg_iteration_count,
                  .line_search_candidate_count = line_search_candidate_count,
                  .gravity = {.x = 0.0F, .y = 0.0F, .z = gravity_z},
                  .length_stiffness = length_stiffness,
                  .area_stiffness = area_stiffness,
                  .bending_stiffness = bending_stiffness,
                  .bending_damping = bending_damping,
                  .hessian_positive_margin = 1.0e-3F,
                  .armijo_coefficient = 1.0e-4F,
                  .line_search_contraction = 0.5F,
                  .fixed_vertices =
                      {
                          {.particle = fixed_particles[0], .position = model.configuration.rest_positions[fixed_particles[0]]},
                          {.particle = fixed_particles[1], .position = model.configuration.rest_positions[fixed_particles[1]]},
                          {.particle = fixed_particles[2], .position = model.configuration.rest_positions[fixed_particles[2]]},
                          {.particle = fixed_particles[3], .position = model.configuration.rest_positions[fixed_particles[3]]},
                          {.particle = fixed_particles[4], .position = model.configuration.rest_positions[fixed_particles[4]]},
                      },
              }),
          current_state(solver.allocate_state(model)),
          next_state(solver.allocate_state(model)),
          control(solver.allocate_control(model)),
          parameters(solver.allocate_parameters(model)),
          step_cache(solver.allocate_step_cache(model)),
          workspace(solver.allocate_workspace(model)) {
        std::vector<Vector3<float>> velocities(model.particle_count);
        std::vector<Vector3<float>> accelerations(model.particle_count, {.x = 0.0F, .y = 0.0F, .z = gravity_z});
        for (std::uint32_t row = 0u; row < rows; ++row) {
            accelerations[row * columns] = {};
            for (std::uint32_t column = 1u; column < columns; ++column) velocities[row * columns + column].z = initial_tip_velocity * static_cast<float>(column) / static_cast<float>(columns - 1u);
        }
        simulation::upload(stream, model.configuration.rest_positions, current_state.positions);
        simulation::upload(stream, velocities, current_state.velocities);
        simulation::upload(stream, accelerations, current_state.accelerations);
        simulation::upload(stream, model.configuration.rest_positions, next_state.positions);
        simulation::upload(stream, velocities, next_state.velocities);
        simulation::upload(stream, accelerations, next_state.accelerations);
        simulation::clear(stream, control.external_forces);

        std::vector<float> masses(model.particle_count);
        for (const deformables::cloth::Triangle triangle : model.configuration.triangles) {
            const Vector3<float> first = model.configuration.rest_positions[triangle.first];
            const Vector3<float> second = model.configuration.rest_positions[triangle.second];
            const Vector3<float> third = model.configuration.rest_positions[triangle.third];
            const float share = surface_density * length(cross(second - first, third - first)) / 6.0F;
            masses[triangle.first] += share;
            masses[triangle.second] += share;
            masses[triangle.third] += share;
        }
        ::cuda::copy_bytes(stream, masses, parameters.masses.values);
        stream.sync();
    }

    Summary Simulation::run() {
        for (std::uint32_t frame = 0u; frame < frame_count; ++frame) step();
        return summarize();
    }

    deformables::cloth::ModelConfiguration<float> Simulation::create_configuration() {
        deformables::cloth::ModelConfiguration<float> result = support::create_grid({.rows = rows, .columns = columns, .width = beam_length, .height = beam_width});
        for (std::uint32_t row = 0u; row < rows; ++row) {
            const float cross_coordinate = static_cast<float>(row) / static_cast<float>(rows - 1u);
            const float normalized_distance = std::abs(2.0F * cross_coordinate - 1.0F);
            for (std::uint32_t column = 0u; column < columns; ++column) {
                const std::uint32_t particle = row * columns + column;
                result.rest_positions[particle].y = (cross_coordinate - 0.5F) * beam_width;
                result.rest_positions[particle].z = -crease_depth * (1.0F - normalized_distance);
            }
        }
        return result;
    }

    void Simulation::step() {
        solver.forward(model, current_state, control, parameters, next_state, step_cache, workspace);
        std::swap(current_state, next_state);
    }

    Summary Simulation::summarize() {
        const std::size_t particle_count = model.particle_count;
        const std::size_t edge_count = model.topology.edges.size();
        const std::size_t triangle_count = model.configuration.triangles.size();
        const std::size_t hinge_count = model.topology.hinges.size();
        std::array<std::vector<float>, 9uz> state{
            std::vector<float>(particle_count), std::vector<float>(particle_count), std::vector<float>(particle_count),
            std::vector<float>(particle_count), std::vector<float>(particle_count), std::vector<float>(particle_count),
            std::vector<float>(particle_count), std::vector<float>(particle_count), std::vector<float>(particle_count),
        };
        std::vector<float> edge_conditions(edge_count);
        std::vector<float> edge_energies(edge_count);
        std::vector<float> triangle_conditions(triangle_count);
        std::vector<float> triangle_energies(triangle_count);
        std::vector<float> hinge_deltas(hinge_count);
        std::vector<float> hinge_energies(hinge_count);
        std::vector<float> damping_potentials(hinge_count);
        float regularization_shift{};
        float accepted_step_size{};
        std::uint32_t accepted_candidate{};
        ::cuda::copy_bytes(stream, current_state.positions.x, ::cuda::std::span<float>{state[0].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.y, ::cuda::std::span<float>{state[1].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.z, ::cuda::std::span<float>{state[2].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.x, ::cuda::std::span<float>{state[3].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.y, ::cuda::std::span<float>{state[4].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.z, ::cuda::std::span<float>{state[5].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.accelerations.x, ::cuda::std::span<float>{state[6].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.accelerations.y, ::cuda::std::span<float>{state[7].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.accelerations.z, ::cuda::std::span<float>{state[8].data(), particle_count});
        ::cuda::copy_bytes(stream, step_cache.edge_length_conditions.values, ::cuda::std::span<float>{edge_conditions.data(), edge_count});
        ::cuda::copy_bytes(stream, step_cache.edge_energies.values, ::cuda::std::span<float>{edge_energies.data(), edge_count});
        ::cuda::copy_bytes(stream, step_cache.triangle_area_conditions.values, ::cuda::std::span<float>{triangle_conditions.data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.triangle_energies.values, ::cuda::std::span<float>{triangle_energies.data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.hinge_angle_deltas.values, ::cuda::std::span<float>{hinge_deltas.data(), hinge_count});
        ::cuda::copy_bytes(stream, step_cache.hinge_energies.values, ::cuda::std::span<float>{hinge_energies.data(), hinge_count});
        ::cuda::copy_bytes(stream, step_cache.hinge_damping_potentials.values, ::cuda::std::span<float>{damping_potentials.data(), hinge_count});
        ::cuda::copy_bytes(stream, step_cache.regularization_shift.values, ::cuda::std::span<float>{&regularization_shift, 1uz});
        ::cuda::copy_bytes(stream, step_cache.accepted_step_size.values, ::cuda::std::span<float>{&accepted_step_size, 1uz});
        ::cuda::copy_bytes(stream, step_cache.accepted_candidate.values, ::cuda::std::span<std::uint32_t>{&accepted_candidate, 1uz});
        stream.sync();

        std::uint32_t curved_rest_hinge_count = 0u;
        float maximum_absolute_rest_dihedral = 0.0F;
        for (const deformables::cloth::Hinge hinge : model.topology.hinges) {
            const Vector3<float> edge_first = model.configuration.rest_positions[hinge.edge_first];
            const Vector3<float> edge_second = model.configuration.rest_positions[hinge.edge_second];
            const Vector3<float> first_opposite = model.configuration.rest_positions[hinge.first_opposite];
            const Vector3<float> second_opposite = model.configuration.rest_positions[hinge.second_opposite];
            const Vector3<float> edge = normalized(edge_second - edge_first);
            const Vector3<float> first_normal = normalized(cross(edge_second - edge_first, first_opposite - edge_first));
            const Vector3<float> second_normal = normalized(cross(edge_first - edge_second, second_opposite - edge_second));
            const float angle = std::atan2(dot(cross(first_normal, second_normal), edge), dot(first_normal, second_normal));
            maximum_absolute_rest_dihedral = std::max(maximum_absolute_rest_dihedral, std::abs(angle));
            if (std::abs(angle) > 1.0e-4F) ++curved_rest_hinge_count;
        }

        float maximum_relative_edge_length_error = 0.0F;
        float maximum_relative_triangle_area_error = 0.0F;
        float maximum_absolute_dihedral_delta = 0.0F;
        double total_membrane_energy = 0.0;
        double total_bending_energy = 0.0;
        double total_damping_potential = 0.0;
        for (std::size_t edge = 0uz; edge < edge_count; ++edge) {
            maximum_relative_edge_length_error = std::max(maximum_relative_edge_length_error, std::abs(edge_conditions[edge]));
            total_membrane_energy += edge_energies[edge];
        }
        for (std::size_t triangle = 0uz; triangle < triangle_count; ++triangle) {
            maximum_relative_triangle_area_error = std::max(maximum_relative_triangle_area_error, std::abs(triangle_conditions[triangle]));
            total_membrane_energy += triangle_energies[triangle];
        }
        for (std::size_t hinge = 0uz; hinge < hinge_count; ++hinge) {
            maximum_absolute_dihedral_delta = std::max(maximum_absolute_dihedral_delta, std::abs(hinge_deltas[hinge]));
            total_bending_energy += hinge_energies[hinge];
            total_damping_potential += damping_potentials[hinge];
        }

        float maximum_fixed_position_error = 0.0F;
        for (const std::uint32_t particle : fixed_particles) {
            const Vector3<float> position{.x = state[0][particle], .y = state[1][particle], .z = state[2][particle]};
            maximum_fixed_position_error = std::max(maximum_fixed_position_error, length(position - model.configuration.rest_positions[particle]));
        }
        float maximum_position_magnitude = 0.0F;
        float maximum_velocity_magnitude = 0.0F;
        for (std::size_t particle = 0uz; particle < particle_count; ++particle) {
            maximum_position_magnitude = std::max(maximum_position_magnitude, length(Vector3<float>{.x = state[0][particle], .y = state[1][particle], .z = state[2][particle]}));
            maximum_velocity_magnitude = std::max(maximum_velocity_magnitude, length(Vector3<float>{.x = state[3][particle], .y = state[4][particle], .z = state[5][particle]}));
        }
        const Vector3<float> probe_position{.x = state[0][probe_particle], .y = state[1][probe_particle], .z = state[2][probe_particle]};
        const Vector3<float> probe_velocity{.x = state[3][probe_particle], .y = state[4][probe_particle], .z = state[5][probe_particle]};

        return {
            .frames = frame_count,
            .physical_time = static_cast<double>(frame_count) * time_step,
            .curved_rest_hinge_count = curved_rest_hinge_count,
            .maximum_absolute_rest_dihedral = maximum_absolute_rest_dihedral,
            .maximum_relative_edge_length_error = maximum_relative_edge_length_error,
            .maximum_relative_triangle_area_error = maximum_relative_triangle_area_error,
            .maximum_absolute_dihedral_delta = maximum_absolute_dihedral_delta,
            .total_membrane_energy = total_membrane_energy,
            .total_bending_energy = total_bending_energy,
            .total_damping_potential = total_damping_potential,
            .regularization_shift = regularization_shift,
            .accepted_step_size = accepted_step_size,
            .accepted_line_search_candidate = accepted_candidate,
            .maximum_fixed_position_error = maximum_fixed_position_error,
            .maximum_position_magnitude = maximum_position_magnitude,
            .maximum_velocity_magnitude = maximum_velocity_magnitude,
            .probe_displacement = length(probe_position - model.configuration.rest_positions[probe_particle]),
            .probe_position = probe_position,
            .probe_velocity = probe_velocity,
        };
    }
} // namespace physica::examples::cloth::discrete_shells
