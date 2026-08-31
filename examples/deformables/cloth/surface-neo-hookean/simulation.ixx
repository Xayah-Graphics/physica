module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.surface_neo_hookean;

import std;
import physica.deformables.cloth.model;
import physica.deformables.cloth.solvers.surface_neo_hookean;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::surface_neo_hookean {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        float minimum_surface_jacobian;
        float maximum_surface_jacobian;
        float maximum_absolute_log_surface_jacobian;
        double total_elastic_energy;
        float regularization_shift;
        float accepted_step_size;
        float maximum_domain_step;
        std::uint32_t accepted_line_search_candidate;
        float maximum_fixed_position_error;
        float maximum_position_magnitude;
        float maximum_velocity_magnitude;
        float probe_displacement;
        Vector3<float> probe_position;
        Vector3<float> probe_velocity;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows                        = 8u;
        inline static constexpr std::uint32_t columns                     = 12u;
        inline static constexpr float width                               = 1.4F;
        inline static constexpr float height                              = 0.9F;
        inline static constexpr float time_step                           = 1.0F / 240.0F;
        inline static constexpr std::uint32_t frame_count                 = 180u;
        inline static constexpr std::uint32_t newton_iteration_count      = 8u;
        inline static constexpr std::uint32_t pcg_iteration_count         = 128u;
        inline static constexpr std::uint32_t line_search_candidate_count = 14u;
        inline static constexpr float gravity_y                           = -9.81F;
        inline static constexpr float mass                                = 0.005F;
        inline static constexpr float young_modulus                       = 3.0e5F;
        inline static constexpr float poisson_ratio                       = 0.3F;
        inline static constexpr float thickness                           = 0.0015F;
        inline static constexpr float domain_safety                       = 0.9F;
        inline static constexpr float initial_compression                 = 0.9F;
        inline static constexpr float initial_perturbation                = 0.025F;
        inline static constexpr float initial_velocity                    = 0.03F;
        inline static constexpr std::uint32_t probe_particle              = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2uz> fixed_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::surface_neo_hookean::Solver solver;
        deformables::cloth::State<float> current_state;
        deformables::cloth::State<float> next_state;
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
                  .time_step = time_step,
                  .newton_iteration_count = newton_iteration_count,
                  .pcg_iteration_count = pcg_iteration_count,
                  .line_search_candidate_count = line_search_candidate_count,
                  .gravity = {.x = 0.0F, .y = gravity_y, .z = 0.0F},
                  .young_modulus = young_modulus,
                  .poisson_ratio = poisson_ratio,
                  .thickness = thickness,
                  .hessian_positive_margin = 1.0e-3F,
                  .armijo_coefficient = 1.0e-4F,
                  .line_search_contraction = 0.5F,
                  .domain_safety = domain_safety,
                  .masses = std::vector<float>(model.particle_count, mass),
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
        support::initialize(model, current_state, next_state, control);
        std::vector<Vector3<float>> positions = model.configuration.rest_positions;
        std::vector<Vector3<float>> velocities(model.particle_count);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t column = 0u; column < columns; ++column) {
                const std::uint32_t particle = row * columns + column;
                if (particle == fixed_particles[0] || particle == fixed_particles[1]) continue;
                const float row_phase    = std::numbers::pi_v<float> * static_cast<float>(row) / static_cast<float>(rows - 1u);
                const float column_phase = 2.0F * std::numbers::pi_v<float> * static_cast<float>(column) / static_cast<float>(columns - 1u);
                positions[particle].y   *= initial_compression;
                positions[particle].z    = initial_perturbation * std::sin(row_phase) * std::sin(column_phase);
                velocities[particle].z   = initial_velocity * std::sin(row_phase) * std::cos(column_phase);
            }
        }
        simulation::upload(stream, positions, current_state.positions);
        simulation::upload(stream, velocities, current_state.velocities);
        simulation::upload(stream, positions, next_state.positions);
        simulation::upload(stream, velocities, next_state.velocities);
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
        const std::size_t triangle_count = model.configuration.triangles.size();
        std::array<std::vector<float>, 6uz> state{
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
        };
        std::vector<float> surface_jacobians(triangle_count);
        std::vector<float> log_surface_jacobians(triangle_count);
        std::vector<float> triangle_energies(triangle_count);
        float regularization_shift{};
        float accepted_step_size{};
        float maximum_domain_step{};
        std::uint32_t accepted_candidate{};
        ::cuda::copy_bytes(stream, current_state.positions.x, ::cuda::std::span<float>{state[0].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.y, ::cuda::std::span<float>{state[1].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.z, ::cuda::std::span<float>{state[2].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.x, ::cuda::std::span<float>{state[3].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.y, ::cuda::std::span<float>{state[4].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.z, ::cuda::std::span<float>{state[5].data(), particle_count});
        ::cuda::copy_bytes(stream, step_cache.surface_jacobians.values, ::cuda::std::span<float>{surface_jacobians.data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.log_surface_jacobians.values, ::cuda::std::span<float>{log_surface_jacobians.data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.triangle_energies.values, ::cuda::std::span<float>{triangle_energies.data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.regularization_shift.values, ::cuda::std::span<float>{&regularization_shift, 1uz});
        ::cuda::copy_bytes(stream, step_cache.accepted_step_size.values, ::cuda::std::span<float>{&accepted_step_size, 1uz});
        ::cuda::copy_bytes(stream, step_cache.maximum_domain_step.values, ::cuda::std::span<float>{&maximum_domain_step, 1uz});
        ::cuda::copy_bytes(stream, step_cache.accepted_candidate.values, ::cuda::std::span<std::uint32_t>{&accepted_candidate, 1uz});
        stream.sync();

        float minimum_surface_jacobian = std::numeric_limits<float>::max();
        float maximum_surface_jacobian = 0.0F;
        float maximum_absolute_log_surface_jacobian = 0.0F;
        double total_elastic_energy = 0.0;
        for (std::size_t triangle = 0uz; triangle < triangle_count; ++triangle) {
            minimum_surface_jacobian = std::min(minimum_surface_jacobian, surface_jacobians[triangle]);
            maximum_surface_jacobian = std::max(maximum_surface_jacobian, surface_jacobians[triangle]);
            maximum_absolute_log_surface_jacobian = std::max(maximum_absolute_log_surface_jacobian, std::abs(log_surface_jacobians[triangle]));
            total_elastic_energy += triangle_energies[triangle];
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
        const float probe_row_phase = std::numbers::pi_v<float> * static_cast<float>(probe_row) / static_cast<float>(rows - 1u);
        const float probe_column_phase = 2.0F * std::numbers::pi_v<float> * static_cast<float>(probe_column) / static_cast<float>(columns - 1u);
        initial_probe_position.y *= initial_compression;
        initial_probe_position.z = initial_perturbation * std::sin(probe_row_phase) * std::sin(probe_column_phase);

        return {
            .frames = frame_count,
            .physical_time = static_cast<double>(frame_count) * time_step,
            .minimum_surface_jacobian = minimum_surface_jacobian,
            .maximum_surface_jacobian = maximum_surface_jacobian,
            .maximum_absolute_log_surface_jacobian = maximum_absolute_log_surface_jacobian,
            .total_elastic_energy = total_elastic_energy,
            .regularization_shift = regularization_shift,
            .accepted_step_size = accepted_step_size,
            .maximum_domain_step = maximum_domain_step,
            .accepted_line_search_candidate = accepted_candidate,
            .maximum_fixed_position_error = maximum_fixed_position_error,
            .maximum_position_magnitude = maximum_position_magnitude,
            .maximum_velocity_magnitude = maximum_velocity_magnitude,
            .probe_displacement = length(probe_position - initial_probe_position),
            .probe_position = probe_position,
            .probe_velocity = probe_velocity,
        };
    }
} // namespace physica::examples::cloth::surface_neo_hookean
