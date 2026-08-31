module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.provot_strain_limiting;

import std;
import physica.deformables.cloth.constraints.provot_strain_limit;
import physica.deformables.cloth.integrators.semi_implicit_euler;
import physica.deformables.cloth.model;
import physica.deformables.cloth.operators.mass_spring;
import physica.deformables.cloth.solvers.explicit_dynamics;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::provot_strain_limiting {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        float initial_probe_y;
        Vector3<float> final_probe_position;
        Vector3<float> final_probe_velocity;
        float integrated_maximum_stretch_ratio;
        float projected_maximum_stretch_ratio;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows                  = 8u;
        inline static constexpr std::uint32_t columns               = 12u;
        inline static constexpr float width                         = 1.4F;
        inline static constexpr float height                        = 0.9F;
        inline static constexpr float time_step                     = 1.0F / 240.0F;
        inline static constexpr std::uint32_t frame_count           = 240u;
        inline static constexpr float gravity_y                     = -9.81F;
        inline static constexpr float mass                          = 0.04F;
        inline static constexpr float stretch_stiffness             = 6.0F;
        inline static constexpr float stretch_damping               = 0.15F;
        inline static constexpr float bending_stiffness             = 0.1F;
        inline static constexpr float bending_damping               = 0.04F;
        inline static constexpr float maximum_stretch_ratio         = 1.08F;
        inline static constexpr std::uint32_t projection_iterations = 512u;
        inline static constexpr std::uint32_t probe_particle        = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2u> fixed_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::explicit_dynamics::Solver<float, deformables::cloth::operators::MassSpringForce, deformables::cloth::integrators::SemiImplicitEuler, deformables::cloth::constraints::ProvotStrainLimitConstraint> solver;
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
        [[nodiscard]] Summary summarize(float initial_probe_y);
    };

    Simulation::Simulation()
        : stream{::cuda::devices[0]},
          model(support::create_grid({.rows = rows, .columns = columns, .width = width, .height = height}), stream),
          solver(
              model,
              {
                  .force      = {.gravity = {.x = 0.0F, .y = gravity_y, .z = 0.0F}},
                  .integrator = {.time_step = time_step},
                  .constraint =
                      {
                          .maximum_stretch_ratio = maximum_stretch_ratio,
                          .iteration_count       = projection_iterations,
                          .fixed_vertices =
                              {
                                  {.particle = fixed_particles[0], .position = model.configuration.rest_positions[fixed_particles[0]]},
                                  {.particle = fixed_particles[1], .position = model.configuration.rest_positions[fixed_particles[1]]},
                              },
                      },
              }),
          current_state(solver.allocate_state(model)),
          next_state(solver.allocate_state(model)),
          control(solver.allocate_control(model)),
          parameters(solver.allocate_parameters(model)),
          step_cache(solver.allocate_step_cache(model)),
          workspace(solver.allocate_workspace(model)) {
        support::set_mass_spring_parameters(stream, parameters, {.mass = mass, .stretch_stiffness = stretch_stiffness, .stretch_damping = stretch_damping, .bending_stiffness = bending_stiffness, .bending_damping = bending_damping});
        support::initialize(model, current_state, next_state, control);
    }

    Summary Simulation::run() {
        const float initial_probe_y = model.configuration.rest_positions[probe_particle].y;
        for (std::uint32_t frame = 0u; frame < frame_count; ++frame) step();
        return summarize(initial_probe_y);
    }

    void Simulation::step() {
        solver.forward(model, current_state, control, parameters, next_state, step_cache, workspace);
        std::swap(current_state, next_state);
    }

    Summary Simulation::summarize(const float initial_probe_y) {
        const std::size_t particle_count = current_state.positions.x.size();
        std::array<std::vector<float>, 3u> projected{
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
        };
        std::array<std::vector<float>, 3u> integrated{
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
        };
        Vector3<float> probe_velocity{};
        ::cuda::copy_bytes(stream, current_state.positions.x, ::cuda::std::span<float>{projected[0].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.y, ::cuda::std::span<float>{projected[1].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.z, ::cuda::std::span<float>{projected[2].data(), particle_count});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.x.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.x, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.y.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.y, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.z.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.z, 1uz});
        ::cuda::copy_bytes(stream, step_cache.integrated_positions.x, ::cuda::std::span<float>{integrated[0].data(), particle_count});
        ::cuda::copy_bytes(stream, step_cache.integrated_positions.y, ::cuda::std::span<float>{integrated[1].data(), particle_count});
        ::cuda::copy_bytes(stream, step_cache.integrated_positions.z, ::cuda::std::span<float>{integrated[2].data(), particle_count});
        stream.sync();

        float integrated_maximum_stretch_ratio = 0.0F;
        float projected_maximum_stretch_ratio  = 0.0F;
        for (const deformables::cloth::Edge edge : model.topology.edges) {
            const float rest_length = length(model.configuration.rest_positions[edge.second] - model.configuration.rest_positions[edge.first]);
            const Vector3<float> integrated_first{.x = integrated[0][edge.first], .y = integrated[1][edge.first], .z = integrated[2][edge.first]};
            const Vector3<float> integrated_second{.x = integrated[0][edge.second], .y = integrated[1][edge.second], .z = integrated[2][edge.second]};
            const Vector3<float> projected_first{.x = projected[0][edge.first], .y = projected[1][edge.first], .z = projected[2][edge.first]};
            const Vector3<float> projected_second{.x = projected[0][edge.second], .y = projected[1][edge.second], .z = projected[2][edge.second]};
            integrated_maximum_stretch_ratio = std::max(integrated_maximum_stretch_ratio, length(integrated_second - integrated_first) / rest_length);
            projected_maximum_stretch_ratio  = std::max(projected_maximum_stretch_ratio, length(projected_second - projected_first) / rest_length);
        }

        return {
            .frames                           = frame_count,
            .physical_time                    = static_cast<double>(frame_count) * time_step,
            .initial_probe_y                  = initial_probe_y,
            .final_probe_position             = {.x = projected[0][probe_particle], .y = projected[1][probe_particle], .z = projected[2][probe_particle]},
            .final_probe_velocity             = probe_velocity,
            .integrated_maximum_stretch_ratio = integrated_maximum_stretch_ratio,
            .projected_maximum_stretch_ratio  = projected_maximum_stretch_ratio,
        };
    }
} // namespace physica::examples::cloth::provot_strain_limiting
