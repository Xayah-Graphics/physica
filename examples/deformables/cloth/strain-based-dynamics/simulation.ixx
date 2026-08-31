module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.strain_based_dynamics;

import std;
import physica.deformables.cloth.model;
import physica.deformables.cloth.solvers.strain_based_dynamics;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::strain_based_dynamics {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        float maximum_uu_error;
        float maximum_vv_error;
        float maximum_uv_error;
        float maximum_fixed_error;
        Vector3<float> probe_position;
        Vector3<float> probe_velocity;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows            = 8u;
        inline static constexpr std::uint32_t columns         = 12u;
        inline static constexpr float width                   = 1.1F;
        inline static constexpr float height                  = 0.7F;
        inline static constexpr float time_step               = 1.0F / 120.0F;
        inline static constexpr std::uint32_t frame_count      = 120u;
        inline static constexpr std::uint32_t iteration_count  = 48u;
        inline static constexpr float gravity_y               = -9.81F;
        inline static constexpr float mass                    = 0.02F;
        inline static constexpr float stretch_stiffness_u     = 0.9F;
        inline static constexpr float stretch_stiffness_v     = 0.9F;
        inline static constexpr float shear_stiffness         = 0.8F;
        inline static constexpr std::uint32_t probe_particle  = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2u> fixed_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::strain_based_dynamics::Solver solver;
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
          solver(model,
                 {
                     .time_step           = time_step,
                     .iteration_count     = iteration_count,
                     .gravity             = {.x = 0.0F, .y = gravity_y, .z = 0.0F},
                     .stretch_stiffness_u = stretch_stiffness_u,
                     .stretch_stiffness_v = stretch_stiffness_v,
                     .shear_stiffness     = shear_stiffness,
                     .fixed_vertices      = {
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
        const std::size_t particle_count = current_state.positions.x.size();
        std::array<std::vector<float>, 3u> host{
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
        };
        Vector3<float> probe_velocity{};
        ::cuda::copy_bytes(stream, current_state.positions.x, ::cuda::std::span<float>{host[0].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.y, ::cuda::std::span<float>{host[1].data(), particle_count});
        ::cuda::copy_bytes(stream, current_state.positions.z, ::cuda::std::span<float>{host[2].data(), particle_count});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.x.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.x, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.y.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.y, 1uz});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{current_state.velocities.z.data() + probe_particle, 1uz}, ::cuda::std::span<float>{&probe_velocity.z, 1uz});
        stream.sync();

        std::vector<Vector3<float>> positions(particle_count);
        for (std::size_t particle = 0uz; particle < particle_count; ++particle) positions[particle] = {.x = host[0][particle], .y = host[1][particle], .z = host[2][particle]};

        float maximum_uu_error = 0.0F;
        float maximum_vv_error = 0.0F;
        float maximum_uv_error = 0.0F;
        for (std::size_t triangle_index = 0uz; triangle_index < model.configuration.triangles.size(); ++triangle_index) {
            const deformables::cloth::Triangle triangle = model.configuration.triangles[triangle_index];
            const deformables::cloth::TriangleMaterialCoordinates<float> coordinates = model.configuration.material_coordinates[triangle_index];
            const float rest_00 = coordinates.second.u - coordinates.first.u;
            const float rest_01 = coordinates.third.u - coordinates.first.u;
            const float rest_10 = coordinates.second.v - coordinates.first.v;
            const float rest_11 = coordinates.third.v - coordinates.first.v;
            const float inverse_determinant = 1.0F / (rest_00 * rest_11 - rest_01 * rest_10);
            const float inverse_rest_00 = rest_11 * inverse_determinant;
            const float inverse_rest_01 = -rest_01 * inverse_determinant;
            const float inverse_rest_10 = -rest_10 * inverse_determinant;
            const float inverse_rest_11 = rest_00 * inverse_determinant;
            const Vector3<float> first_displacement  = positions[triangle.second] - positions[triangle.first];
            const Vector3<float> second_displacement = positions[triangle.third] - positions[triangle.first];
            const Vector3<float> deformation_u = inverse_rest_00 * first_displacement + inverse_rest_10 * second_displacement;
            const Vector3<float> deformation_v = inverse_rest_01 * first_displacement + inverse_rest_11 * second_displacement;
            maximum_uu_error = std::max(maximum_uu_error, std::abs(dot(deformation_u, deformation_u) - 1.0F));
            maximum_vv_error = std::max(maximum_vv_error, std::abs(dot(deformation_v, deformation_v) - 1.0F));
            maximum_uv_error = std::max(maximum_uv_error, std::abs(dot(deformation_u, deformation_v)));
        }

        float maximum_fixed_error = 0.0F;
        for (const std::uint32_t particle : fixed_particles) maximum_fixed_error = std::max(maximum_fixed_error, length(positions[particle] - model.configuration.rest_positions[particle]));

        return {
            .frames              = frame_count,
            .physical_time       = static_cast<double>(frame_count) * time_step,
            .maximum_uu_error    = maximum_uu_error,
            .maximum_vv_error    = maximum_vv_error,
            .maximum_uv_error    = maximum_uv_error,
            .maximum_fixed_error = maximum_fixed_error,
            .probe_position      = positions[probe_particle],
            .probe_velocity      = probe_velocity,
        };
    }
} // namespace physica::examples::cloth::strain_based_dynamics
