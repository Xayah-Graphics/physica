module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.analytic_collision;

import std;
import physica.deformables.cloth.constraints.analytic_collision;
import physica.deformables.cloth.integrators.semi_implicit_euler;
import physica.deformables.cloth.model;
import physica.deformables.cloth.operators.mass_spring;
import physica.deformables.cloth.solvers.explicit_dynamics;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::analytic_collision {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        Vector3<float> final_center_position;
        Vector3<float> final_center_velocity;
        Vector3<float> final_corner_position;
        float minimum_plane_clearance;
        float minimum_sphere_clearance;
        std::uint32_t plane_contact_vertices;
        std::uint32_t sphere_contact_vertices;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows            = 10u;
        inline static constexpr std::uint32_t columns         = 10u;
        inline static constexpr float width                   = 1.6F;
        inline static constexpr float height                  = 1.6F;
        inline static constexpr float initial_height          = 1.4F;
        inline static constexpr float time_step               = 1.0F / 480.0F;
        inline static constexpr std::uint32_t frame_count      = 384u;
        inline static constexpr float gravity_y               = -9.81F;
        inline static constexpr float mass                    = 0.04F;
        inline static constexpr float stretch_stiffness       = 30.0F;
        inline static constexpr float stretch_damping         = 0.8F;
        inline static constexpr float bending_stiffness       = 0.6F;
        inline static constexpr float bending_damping         = 0.2F;
        inline static constexpr float thickness               = 0.02F;
        inline static constexpr float plane_offset            = 0.0F;
        inline static constexpr Vector3<float> sphere_center{.x = 0.0F, .y = 0.46F, .z = 0.0F};
        inline static constexpr float sphere_radius           = 0.4F;
        inline static constexpr std::uint32_t center_particle = (rows / 2u) * columns + columns / 2u;
        inline static constexpr std::uint32_t corner_particle = 0u;

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::explicit_dynamics::Solver<float, deformables::cloth::operators::MassSpringForce, deformables::cloth::integrators::SemiImplicitEuler, deformables::cloth::constraints::AnalyticCollisionConstraint> solver;
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
                  .force      = {.gravity = {.x = 0.0F, .y = gravity_y, .z = 0.0F}},
                  .integrator = {.time_step = time_step},
                  .constraint =
                      {
                          .thickness = thickness,
                          .planes =
                              {
                                  {.normal = {.x = 0.0F, .y = 1.0F, .z = 0.0F}, .offset = plane_offset, .restitution = 0.0F, .friction = 0.0F},
                              },
                          .spheres =
                              {
                                  {.center = sphere_center, .radius = sphere_radius, .restitution = 0.0F, .friction = 0.0F},
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
        for (std::uint32_t frame = 0u; frame < frame_count; ++frame) step();
        return summarize();
    }

    deformables::cloth::ModelConfiguration<float> Simulation::create_configuration() {
        deformables::cloth::ModelConfiguration<float> result = support::create_grid({.rows = rows, .columns = columns, .width = width, .height = height});
        for (Vector3<float>& position : result.rest_positions) position = {.x = position.x - 0.5F * width, .y = initial_height, .z = position.y + 0.5F * height};
        return result;
    }

    void Simulation::step() {
        solver.forward(model, current_state, control, parameters, next_state, step_cache, workspace);
        std::swap(current_state, next_state);
    }

    Summary Simulation::summarize() {
        const std::size_t particle_count = current_state.positions.x.size();
        std::array<std::vector<float>, 3u> positions{
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
        };
        Vector3<float> center_velocity{};
        ::cuda::copy_bytes(stream, current_state.positions.x, ::cuda::std::span<float>{positions[0].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.y, ::cuda::std::span<float>{positions[1].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.z, ::cuda::std::span<float>{positions[2].data(), particle_count});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.x.data() + center_particle, 1uz}, ::cuda::std::span<float>{&center_velocity.x, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.y.data() + center_particle, 1uz}, ::cuda::std::span<float>{&center_velocity.y, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.z.data() + center_particle, 1uz}, ::cuda::std::span<float>{&center_velocity.z, 1uz});
        stream.sync();

        float minimum_plane_clearance          = std::numeric_limits<float>::max();
        float minimum_sphere_clearance         = std::numeric_limits<float>::max();
        std::uint32_t plane_contact_vertices  = 0u;
        std::uint32_t sphere_contact_vertices = 0u;
        for (std::size_t particle = 0uz; particle < particle_count; ++particle) {
            const Vector3<float> position{.x = positions[0][particle], .y = positions[1][particle], .z = positions[2][particle]};
            const float plane_clearance  = position.y - plane_offset;
            const float sphere_clearance = length(position - sphere_center) - sphere_radius;
            minimum_plane_clearance      = std::min(minimum_plane_clearance, plane_clearance);
            minimum_sphere_clearance     = std::min(minimum_sphere_clearance, sphere_clearance);
            if (plane_clearance <= thickness + 1.0e-5F) ++plane_contact_vertices;
            if (sphere_clearance <= thickness + 1.0e-5F) ++sphere_contact_vertices;
        }

        return {
            .frames                   = frame_count,
            .physical_time            = static_cast<double>(frame_count) * time_step,
            .final_center_position    = {.x = positions[0][center_particle], .y = positions[1][center_particle], .z = positions[2][center_particle]},
            .final_center_velocity    = center_velocity,
            .final_corner_position    = {.x = positions[0][corner_particle], .y = positions[1][corner_particle], .z = positions[2][corner_particle]},
            .minimum_plane_clearance  = minimum_plane_clearance,
            .minimum_sphere_clearance = minimum_sphere_clearance,
            .plane_contact_vertices   = plane_contact_vertices,
            .sphere_contact_vertices  = sphere_contact_vertices,
        };
    }
} // namespace physica::examples::cloth::analytic_collision
