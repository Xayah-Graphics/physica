module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.choi_ko;

import std;
import physica.deformables.cloth.model;
import physica.deformables.cloth.solvers.choi_ko;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::choi_ko {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        std::array<float, 4uz> maximum_tension_conditions;
        std::uint32_t active_tension_condition_count;
        std::uint32_t buckled_hinge_count;
        float maximum_hinge_curvature;
        float maximum_fixed_position_error;
        float maximum_position_magnitude;
        float maximum_velocity_magnitude;
        float probe_displacement;
        Vector3<float> probe_position;
        Vector3<float> probe_velocity;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows                    = 8u;
        inline static constexpr std::uint32_t columns                 = 12u;
        inline static constexpr float width                           = 1.4F;
        inline static constexpr float height                          = 0.9F;
        inline static constexpr float time_step                       = 1.0F / 480.0F;
        inline static constexpr std::uint32_t frame_count             = 120u;
        inline static constexpr std::uint32_t pcg_iteration_count     = 160u;
        inline static constexpr float gravity_y                       = -9.81F;
        inline static constexpr float mass                            = 0.04F;
        inline static constexpr float stretch_u_stiffness             = 2.0e5F;
        inline static constexpr float stretch_v_stiffness             = 1.6e5F;
        inline static constexpr float diagonal_u_stiffness            = 7.0e4F;
        inline static constexpr float diagonal_v_stiffness            = 5.0e4F;
        inline static constexpr float bend_u_stiffness                = 0.005F;
        inline static constexpr float bend_v_stiffness                = 0.01F;
        inline static constexpr float imperfection_stiffness          = 0.2F;
        inline static constexpr float stretch_u_damping               = 300.0F;
        inline static constexpr float stretch_v_damping               = 240.0F;
        inline static constexpr float diagonal_u_damping              = 120.0F;
        inline static constexpr float diagonal_v_damping              = 90.0F;
        inline static constexpr float bending_damping                 = 0.02F;
        inline static constexpr float initial_in_plane_scale          = 0.9F;
        inline static constexpr float initial_perturbation            = 0.012F;
        inline static constexpr float initial_velocity_scale          = 0.08F;
        inline static constexpr std::uint32_t probe_particle          = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2uz> fixed_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::choi_ko::Solver solver;
        deformables::cloth::solvers::choi_ko::State current_state;
        deformables::cloth::solvers::choi_ko::State next_state;
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
        void step();
        [[nodiscard]] Summary summarize();
    };

    Simulation::Simulation()
        : stream{::cuda::devices[0]},
          model(support::create_grid({.rows = rows, .columns = columns, .width = width, .height = height}), stream),
          solver(
              model,
              {
                  .time_step                    = time_step,
                  .pcg_iteration_count          = pcg_iteration_count,
                  .gravity                      = {.x = 0.0F, .y = gravity_y, .z = 0.0F},
                  .stretch_u_stiffness          = stretch_u_stiffness,
                  .stretch_v_stiffness          = stretch_v_stiffness,
                  .diagonal_u_stiffness         = diagonal_u_stiffness,
                  .diagonal_v_stiffness         = diagonal_v_stiffness,
                  .bend_u_stiffness             = bend_u_stiffness,
                  .bend_v_stiffness             = bend_v_stiffness,
                  .imperfection_stiffness       = imperfection_stiffness,
                  .stretch_u_damping            = stretch_u_damping,
                  .stretch_v_damping            = stretch_v_damping,
                  .diagonal_u_damping           = diagonal_u_damping,
                  .diagonal_v_damping           = diagonal_v_damping,
                  .bending_damping              = bending_damping,
                  .fixed_vertices =
                      {
                          {.particle = fixed_particles[0], .position = model.configuration.rest_positions[fixed_particles[0]]},
                          {.particle = fixed_particles[1], .position = model.configuration.rest_positions[fixed_particles[1]]},
                      },
              }),
          current_state(solver.allocate_state(model)),
          next_state(solver.allocate_state(model)),
          control(solver.allocate_control(model)),
          parameters(solver.allocate_parameters(model)),
          step_cache(solver.allocate_step_cache(model)),
          workspace(solver.allocate_workspace(model)) {
        std::vector<Vector3<float>> positions = model.configuration.rest_positions;
        std::vector<Vector3<float>> velocities(model.particle_count);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t column = 0u; column < columns; ++column) {
                const std::uint32_t particle = row * columns + column;
                if (particle == fixed_particles[0] || particle == fixed_particles[1]) continue;
                const float row_fraction    = static_cast<float>(row) / static_cast<float>(rows - 1u);
                const float row_phase       = std::numbers::pi_v<float> * row_fraction;
                const float column_phase    = 2.0F * std::numbers::pi_v<float> * static_cast<float>(column) / static_cast<float>(columns - 1u);
                const float in_plane_scale  = 1.0F - (1.0F - initial_in_plane_scale) * row_fraction;
                positions[particle].x       = 0.5F * width + in_plane_scale * (positions[particle].x - 0.5F * width);
                positions[particle].y       = in_plane_scale * positions[particle].y;
                positions[particle].z       = initial_perturbation * std::sin(row_phase) * std::sin(column_phase);
                velocities[particle].y      = -initial_velocity_scale * row_fraction;
            }
        }
        std::vector<Vector3<float>> previous_positions(positions.size());
        for (std::size_t particle = 0uz; particle < positions.size(); ++particle) previous_positions[particle] = positions[particle] - time_step * velocities[particle];

        simulation::upload(stream, positions, current_state.positions);
        simulation::upload(stream, velocities, current_state.velocities);
        simulation::upload(stream, previous_positions, current_state.previous_positions);
        simulation::upload(stream, velocities, current_state.previous_velocities);
        simulation::upload(stream, positions, next_state.positions);
        simulation::upload(stream, velocities, next_state.velocities);
        simulation::upload(stream, previous_positions, next_state.previous_positions);
        simulation::upload(stream, velocities, next_state.previous_velocities);
        simulation::clear(stream, control.external_forces);
        const std::vector<float> masses(model.particle_count, mass);
        ::cuda::copy_bytes(stream, masses, parameters.masses.values);
        stream.sync();
    }

    Summary Simulation::run() {
        for (std::uint32_t frame = 0u; frame < frame_count; ++frame) step();
        return summarize();
    }

    void Simulation::step() {
        solver.forward(model, current_state, control, parameters, next_state, step_cache, workspace);
        std::swap(current_state, next_state);
    }

    Summary Simulation::summarize() {
        const std::size_t particle_count = model.particle_count;
        std::array<std::vector<float>, 6uz> state{
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
        };
        ::cuda::copy_bytes(stream, current_state.positions.x, ::cuda::std::span<float>{state[0].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.y, ::cuda::std::span<float>{state[1].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.z, ::cuda::std::span<float>{state[2].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.x, ::cuda::std::span<float>{state[3].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.y, ::cuda::std::span<float>{state[4].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.z, ::cuda::std::span<float>{state[5].data(), particle_count});
        stream.sync();

        std::array<float, 4uz> maximum_tension_conditions{};
        std::uint32_t active_tension_condition_count = 0u;
        constexpr float inverse_square_root_two = 0.70710678118654752440F;
        for (std::size_t triangle_index = 0uz; triangle_index < model.configuration.triangles.size(); ++triangle_index) {
            const deformables::cloth::Triangle triangle = model.configuration.triangles[triangle_index];
            const deformables::cloth::TriangleMaterialCoordinates<float> coordinates = model.configuration.material_coordinates[triangle_index];
            const float delta_u_first  = coordinates.second.u - coordinates.first.u;
            const float delta_v_first  = coordinates.second.v - coordinates.first.v;
            const float delta_u_second = coordinates.third.u - coordinates.first.u;
            const float delta_v_second = coordinates.third.v - coordinates.first.v;
            const float inverse_determinant = 1.0F / (delta_u_first * delta_v_second - delta_u_second * delta_v_first);
            const Vector3<float> first{.x = state[0][triangle.first], .y = state[1][triangle.first], .z = state[2][triangle.first]};
            const Vector3<float> second{.x = state[0][triangle.second], .y = state[1][triangle.second], .z = state[2][triangle.second]};
            const Vector3<float> third{.x = state[0][triangle.third], .y = state[1][triangle.third], .z = state[2][triangle.third]};
            const Vector3<float> first_edge = second - first;
            const Vector3<float> second_edge = third - first;
            const Vector3<float> u_derivative = delta_v_second * inverse_determinant * first_edge - delta_v_first * inverse_determinant * second_edge;
            const Vector3<float> v_derivative = -delta_u_second * inverse_determinant * first_edge + delta_u_first * inverse_determinant * second_edge;
            const std::array<Vector3<float>, 4uz> derivatives{u_derivative, v_derivative, inverse_square_root_two * (u_derivative - v_derivative), inverse_square_root_two * (u_derivative + v_derivative)};
            for (std::size_t direction = 0uz; direction < derivatives.size(); ++direction) {
                const float condition = std::max(length(derivatives[direction]) - 1.0F, 0.0F);
                maximum_tension_conditions[direction] = std::max(maximum_tension_conditions[direction], condition);
                if (condition > 0.0F) ++active_tension_condition_count;
            }
        }

        std::uint32_t buckled_hinge_count = 0u;
        float maximum_hinge_curvature = 0.0F;
        for (const deformables::cloth::Hinge hinge : model.topology.hinges) {
            const Vector3<float> first{.x = state[0][hinge.first_opposite], .y = state[1][hinge.first_opposite], .z = state[2][hinge.first_opposite]};
            const Vector3<float> second{.x = state[0][hinge.second_opposite], .y = state[1][hinge.second_opposite], .z = state[2][hinge.second_opposite]};
            const float span = length(second - first);
            const float rest_span = length(model.configuration.rest_positions[hinge.second_opposite] - model.configuration.rest_positions[hinge.first_opposite]);
            if (span >= rest_span) continue;
            ++buckled_hinge_count;
            const float normalized_span = span / rest_span;
            float lower = 0.0F;
            float upper = std::numbers::pi_v<float>;
            for (std::uint32_t iteration = 0u; iteration < 48u; ++iteration) {
                const float middle = 0.5F * (lower + upper);
                if (std::sin(middle) / middle > normalized_span) lower = middle;
                else upper = middle;
            }
            maximum_hinge_curvature = std::max(maximum_hinge_curvature, (lower + upper) / rest_span);
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
        Vector3<float> initial_probe_position = model.configuration.rest_positions[probe_particle];
        const std::uint32_t probe_row = probe_particle / columns;
        const std::uint32_t probe_column = probe_particle % columns;
        const float probe_row_fraction = static_cast<float>(probe_row) / static_cast<float>(rows - 1u);
        const float probe_row_phase = std::numbers::pi_v<float> * probe_row_fraction;
        const float probe_column_phase = 2.0F * std::numbers::pi_v<float> * static_cast<float>(probe_column) / static_cast<float>(columns - 1u);
        const float probe_in_plane_scale = 1.0F - (1.0F - initial_in_plane_scale) * probe_row_fraction;
        initial_probe_position.x = 0.5F * width + probe_in_plane_scale * (initial_probe_position.x - 0.5F * width);
        initial_probe_position.y = probe_in_plane_scale * initial_probe_position.y;
        initial_probe_position.z = initial_perturbation * std::sin(probe_row_phase) * std::sin(probe_column_phase);

        return {
            .frames                         = frame_count,
            .physical_time                  = static_cast<double>(frame_count) * time_step,
            .maximum_tension_conditions     = maximum_tension_conditions,
            .active_tension_condition_count = active_tension_condition_count,
            .buckled_hinge_count            = buckled_hinge_count,
            .maximum_hinge_curvature        = maximum_hinge_curvature,
            .maximum_fixed_position_error   = maximum_fixed_position_error,
            .maximum_position_magnitude     = maximum_position_magnitude,
            .maximum_velocity_magnitude     = maximum_velocity_magnitude,
            .probe_displacement             = length(probe_position - initial_probe_position),
            .probe_position                 = probe_position,
            .probe_velocity                 = probe_velocity,
        };
    }
} // namespace physica::examples::cloth::choi_ko
