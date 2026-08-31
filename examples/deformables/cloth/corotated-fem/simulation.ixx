module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.corotated_fem;

import std;
import physica.deformables.cloth.model;
import physica.deformables.cloth.solvers.corotated_fem;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::corotated_fem {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        float maximum_absolute_principal_biot_strain;
        float minimum_surface_jacobian;
        double total_elastic_energy;
        float regularization_shift;
        float maximum_rank_safe_step;
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
        inline static constexpr std::uint32_t rows                        = 8u;
        inline static constexpr std::uint32_t columns                     = 14u;
        inline static constexpr float width                               = 1.5F;
        inline static constexpr float height                              = 0.7F;
        inline static constexpr float time_step                           = 1.0F / 240.0F;
        inline static constexpr std::uint32_t frame_count                 = 120u;
        inline static constexpr std::uint32_t newton_iteration_count      = 8u;
        inline static constexpr std::uint32_t pcg_iteration_count         = 128u;
        inline static constexpr std::uint32_t line_search_candidate_count = 14u;
        inline static constexpr float gravity_y                           = -9.81F;
        inline static constexpr float mass                                = 0.004F;
        inline static constexpr float young_modulus                       = 2.0e5F;
        inline static constexpr float poisson_ratio                       = 0.3F;
        inline static constexpr float thickness                           = 0.0015F;
        inline static constexpr float initial_rotation                    = 110.0F * std::numbers::pi_v<float> / 180.0F;
        inline static constexpr float initial_angular_velocity            = -0.7F;
        inline static constexpr float rank_safety_fraction                = 0.9F;
        inline static constexpr std::uint32_t probe_particle              = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2uz> fixed_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::corotated_fem::Solver solver;
        deformables::cloth::State<float> current_state;
        deformables::cloth::State<float> next_state;
        deformables::cloth::Control<float> control;
        decltype(solver)::Parameters parameters;
        decltype(solver)::StepCache step_cache;
        decltype(solver)::Workspace workspace;
        std::vector<Vector3<float>> initial_positions;

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
                  .rank_safety_fraction = rank_safety_fraction,
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
          workspace(solver.allocate_workspace(model)),
          initial_positions(model.configuration.rest_positions) {
        support::initialize(model, current_state, next_state, control);
        std::vector<Vector3<float>> velocities(model.particle_count);
        const float cosine = std::cos(initial_rotation);
        const float sine = std::sin(initial_rotation);
        for (std::uint32_t particle = 0u; particle < model.particle_count; ++particle) {
            const Vector3<float> rest = model.configuration.rest_positions[particle];
            initial_positions[particle] = {.x = rest.x, .y = cosine * rest.y, .z = sine * rest.y};
            velocities[particle] = {.x = 0.0F, .y = -initial_angular_velocity * initial_positions[particle].z, .z = initial_angular_velocity * initial_positions[particle].y};
        }
        for (const std::uint32_t particle : fixed_particles) velocities[particle] = {};
        simulation::upload(stream, initial_positions, current_state.positions);
        simulation::upload(stream, velocities, current_state.velocities);
        simulation::upload(stream, initial_positions, next_state.positions);
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
        std::array<std::vector<float>, 3uz> biot_strain{
            std::vector<float>(triangle_count),
            std::vector<float>(triangle_count),
            std::vector<float>(triangle_count),
        };
        std::vector<float> triangle_energies(triangle_count);
        float regularization_shift{};
        float maximum_rank_safe_step{};
        float accepted_step_size{};
        std::uint32_t accepted_candidate{};
        ::cuda::copy_bytes(stream, current_state.positions.x, ::cuda::std::span<float>{state[0].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.y, ::cuda::std::span<float>{state[1].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.z, ::cuda::std::span<float>{state[2].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.x, ::cuda::std::span<float>{state[3].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.y, ::cuda::std::span<float>{state[4].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.velocities.z, ::cuda::std::span<float>{state[5].data(), particle_count});
        ::cuda::copy_bytes(stream, step_cache.biot_strains.x, ::cuda::std::span<float>{biot_strain[0].data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.biot_strains.y, ::cuda::std::span<float>{biot_strain[1].data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.biot_strains.z, ::cuda::std::span<float>{biot_strain[2].data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.triangle_energies.values, ::cuda::std::span<float>{triangle_energies.data(), triangle_count});
        ::cuda::copy_bytes(stream, step_cache.regularization_shift.values, ::cuda::std::span<float>{&regularization_shift, 1uz});
        ::cuda::copy_bytes(stream, step_cache.maximum_rank_safe_step.values, ::cuda::std::span<float>{&maximum_rank_safe_step, 1uz});
        ::cuda::copy_bytes(stream, step_cache.accepted_step_size.values, ::cuda::std::span<float>{&accepted_step_size, 1uz});
        ::cuda::copy_bytes(stream, step_cache.accepted_candidate.values, ::cuda::std::span<std::uint32_t>{&accepted_candidate, 1uz});
        stream.sync();

        float maximum_absolute_principal_biot_strain = 0.0F;
        float minimum_surface_jacobian = std::numeric_limits<float>::max();
        double total_elastic_energy = 0.0;
        for (std::size_t triangle = 0uz; triangle < triangle_count; ++triangle) {
            const float difference = biot_strain[0][triangle] - biot_strain[2][triangle];
            const float radius = std::sqrt(0.25F * difference * difference + biot_strain[1][triangle] * biot_strain[1][triangle]);
            const float center = 0.5F * (biot_strain[0][triangle] + biot_strain[2][triangle]);
            maximum_absolute_principal_biot_strain = std::max(maximum_absolute_principal_biot_strain, std::max(std::abs(center + radius), std::abs(center - radius)));
            minimum_surface_jacobian = std::min(minimum_surface_jacobian, (1.0F + biot_strain[0][triangle]) * (1.0F + biot_strain[2][triangle]) - biot_strain[1][triangle] * biot_strain[1][triangle]);
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
        return {
            .frames = frame_count,
            .physical_time = static_cast<double>(frame_count) * time_step,
            .maximum_absolute_principal_biot_strain = maximum_absolute_principal_biot_strain,
            .minimum_surface_jacobian = minimum_surface_jacobian,
            .total_elastic_energy = total_elastic_energy,
            .regularization_shift = regularization_shift,
            .maximum_rank_safe_step = maximum_rank_safe_step,
            .accepted_step_size = accepted_step_size,
            .accepted_line_search_candidate = accepted_candidate,
            .maximum_fixed_position_error = maximum_fixed_position_error,
            .maximum_position_magnitude = maximum_position_magnitude,
            .maximum_velocity_magnitude = maximum_velocity_magnitude,
            .probe_displacement = length(probe_position - initial_positions[probe_particle]),
            .probe_position = probe_position,
            .probe_velocity = probe_velocity,
        };
    }
} // namespace physica::examples::cloth::corotated_fem
