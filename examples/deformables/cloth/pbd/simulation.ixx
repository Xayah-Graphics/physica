module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.pbd;

import std;
import physica.deformables.cloth.model;
import physica.deformables.cloth.solvers.pbd;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::pbd {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        float maximum_stretch_ratio;
        float maximum_fixed_position_error;
        Vector3<float> probe_position;
        Vector3<float> probe_velocity;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows                  = 8u;
        inline static constexpr std::uint32_t columns               = 12u;
        inline static constexpr float width                         = 1.4F;
        inline static constexpr float height                        = 0.9F;
        inline static constexpr float time_step                     = 1.0F / 120.0F;
        inline static constexpr std::uint32_t frame_count           = 120u;
        inline static constexpr std::uint32_t projection_iterations = 256u;
        inline static constexpr float gravity_y                     = -9.81F;
        inline static constexpr float mass                          = 0.04F;
        inline static constexpr std::uint32_t probe_particle        = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2u> fixed_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::pbd::Solver solver;
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
                  .time_step       = time_step,
                  .iteration_count = projection_iterations,
                  .gravity         = {.x = 0.0F, .y = gravity_y, .z = 0.0F},
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
        const std::vector<float> masses(model.particle_count, mass);
        ::cuda::copy_bytes(stream, masses, parameters.masses.values);
        support::initialize(model, current_state, next_state, control);
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
        std::array<std::vector<float>, 3u> positions{
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
        };
        Vector3<float> probe_velocity{};
        ::cuda::copy_bytes(stream, current_state.positions.x, ::cuda::std::span<float>{positions[0].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.y, ::cuda::std::span<float>{positions[1].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.z, ::cuda::std::span<float>{positions[2].data(), particle_count});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.x.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.x, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.y.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.y, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.z.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.z, 1uz});
        stream.sync();

        float maximum_stretch_ratio = 0.0F;
        for (const deformables::cloth::Edge edge : model.topology.edges) {
            const Vector3<float> first{.x = positions[0][edge.first], .y = positions[1][edge.first], .z = positions[2][edge.first]};
            const Vector3<float> second{.x = positions[0][edge.second], .y = positions[1][edge.second], .z = positions[2][edge.second]};
            const float rest_length = length(model.configuration.rest_positions[edge.second] - model.configuration.rest_positions[edge.first]);
            maximum_stretch_ratio   = std::max(maximum_stretch_ratio, length(second - first) / rest_length);
        }

        float maximum_fixed_position_error = 0.0F;
        for (const std::uint32_t particle : fixed_particles) {
            const Vector3<float> position{.x = positions[0][particle], .y = positions[1][particle], .z = positions[2][particle]};
            maximum_fixed_position_error = std::max(maximum_fixed_position_error, length(position - model.configuration.rest_positions[particle]));
        }

        return {
            .frames                       = frame_count,
            .physical_time                = static_cast<double>(frame_count) * time_step,
            .maximum_stretch_ratio        = maximum_stretch_ratio,
            .maximum_fixed_position_error = maximum_fixed_position_error,
            .probe_position               = {.x = positions[0][probe_particle], .y = positions[1][probe_particle], .z = positions[2][probe_particle]},
            .probe_velocity               = probe_velocity,
        };
    }
} // namespace physica::examples::cloth::pbd
