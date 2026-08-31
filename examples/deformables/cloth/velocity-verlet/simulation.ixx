module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.velocity_verlet;

import std;
import physica.deformables.cloth.constraints.fixed_position;
import physica.deformables.cloth.model;
import physica.deformables.cloth.operators.mass_spring;
import physica.deformables.cloth.solvers.velocity_verlet;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::velocity_verlet {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        float initial_probe_y;
        float first_frame_probe_y;
        Vector3<float> final_probe_position;
        Vector3<float> final_probe_velocity;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows           = 4u;
        inline static constexpr std::uint32_t columns        = 6u;
        inline static constexpr float width                  = 1.0F;
        inline static constexpr float height                 = 0.6F;
        inline static constexpr float time_step              = 1.0F / 600.0F;
        inline static constexpr std::uint32_t frame_count     = 120u;
        inline static constexpr float gravity_y              = -9.81F;
        inline static constexpr float mass                   = 0.05F;
        inline static constexpr float stretch_stiffness      = 25.0F;
        inline static constexpr float stretch_damping        = 0.4F;
        inline static constexpr float bending_stiffness      = 0.5F;
        inline static constexpr float bending_damping        = 0.1F;
        inline static constexpr std::uint32_t probe_particle = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2u> anchor_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::velocity_verlet::Solver<float, deformables::cloth::operators::MassSpringForce, deformables::cloth::constraints::FixedPositionConstraint> solver;
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
        [[nodiscard]] Summary summarize(float initial_probe_y, float first_frame_probe_y);
    };

    Simulation::Simulation()
        : stream{::cuda::devices[0]},
          model(support::create_grid({.rows = rows, .columns = columns, .width = width, .height = height}), stream),
          solver(model, {.time_step = time_step, .force = {.gravity = {.x = 0.0F, .y = gravity_y, .z = 0.0F}}, .constraint = support::create_anchors(model.configuration, anchor_particles)}),
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
        step();
        float first_frame_probe_y = 0.0F;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.positions.y.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&first_frame_probe_y, 1uz});
        stream.sync();
        for (std::uint32_t frame = 1u; frame < frame_count; ++frame) step();
        return summarize(initial_probe_y, first_frame_probe_y);
    }

    void Simulation::step() {
        solver.forward(model, current_state, control, parameters, next_state, step_cache, workspace);
        std::swap(current_state, next_state);
    }

    Summary Simulation::summarize(const float initial_probe_y, const float first_frame_probe_y) {
        Summary result{
            .frames               = frame_count,
            .physical_time        = static_cast<double>(frame_count) * time_step,
            .initial_probe_y      = initial_probe_y,
            .first_frame_probe_y  = first_frame_probe_y,
            .final_probe_position = {},
            .final_probe_velocity = {},
        };
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.positions.x.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&result.final_probe_position.x, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.positions.y.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&result.final_probe_position.y, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.positions.z.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&result.final_probe_position.z, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.x.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&result.final_probe_velocity.x, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.y.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&result.final_probe_velocity.y, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.z.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&result.final_probe_velocity.z, 1uz});
        stream.sync();
        return result;
    }
} // namespace physica::examples::cloth::velocity_verlet
