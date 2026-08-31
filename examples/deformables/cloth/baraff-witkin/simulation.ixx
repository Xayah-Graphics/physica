module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.baraff_witkin;

import std;
import physica.deformables.cloth.model;
import physica.deformables.cloth.solvers.baraff_witkin;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::baraff_witkin {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        float maximum_material_stretch;
        float maximum_absolute_material_shear;
        float maximum_fixed_position_error;
        Vector3<float> probe_position;
        Vector3<float> probe_velocity;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows                  = 8u;
        inline static constexpr std::uint32_t columns               = 12u;
        inline static constexpr float width                         = 1.4F;
        inline static constexpr float height                        = 0.9F;
        inline static constexpr float time_step                     = 1.0F / 240.0F;
        inline static constexpr std::uint32_t frame_count           = 240u;
        inline static constexpr std::uint32_t pcg_iteration_count   = 128u;
        inline static constexpr float gravity_y                     = -9.81F;
        inline static constexpr float mass                          = 0.04F;
        inline static constexpr float stretch_u_stiffness           = 1.2e6F;
        inline static constexpr float stretch_v_stiffness           = 1.0e6F;
        inline static constexpr float shear_stiffness               = 4.0e5F;
        inline static constexpr float bend_u_stiffness              = 0.025F;
        inline static constexpr float bend_v_stiffness              = 0.05F;
        inline static constexpr float stretch_u_damping             = 1.2e4F;
        inline static constexpr float stretch_v_damping             = 1.0e4F;
        inline static constexpr float shear_damping                 = 4.0e3F;
        inline static constexpr float bend_u_damping                = 0.0025F;
        inline static constexpr float bend_v_damping                = 0.005F;
        inline static constexpr float initial_perturbation          = 0.035F;
        inline static constexpr std::uint32_t probe_particle        = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2u> fixed_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::baraff_witkin::Solver solver;
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
                  .time_step           = time_step,
                  .pcg_iteration_count = pcg_iteration_count,
                  .gravity             = {.x = 0.0F, .y = gravity_y, .z = 0.0F},
                  .stretch_u_target    = 1.0F,
                  .stretch_v_target    = 1.0F,
                  .stretch_u_stiffness = stretch_u_stiffness,
                  .stretch_v_stiffness = stretch_v_stiffness,
                  .shear_stiffness     = shear_stiffness,
                  .bend_u_stiffness    = bend_u_stiffness,
                  .bend_v_stiffness    = bend_v_stiffness,
                  .stretch_u_damping   = stretch_u_damping,
                  .stretch_v_damping   = stretch_v_damping,
                  .shear_damping       = shear_damping,
                  .bend_u_damping      = bend_u_damping,
                  .bend_v_damping      = bend_v_damping,
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
        const std::vector<float> masses(model.particle_count, mass);
        ::cuda::copy_bytes(stream, masses, parameters.masses.values);
        std::vector<Vector3<float>> initial_positions = model.configuration.rest_positions;
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t column = 0u; column < columns; ++column) {
                const std::uint32_t particle = row * columns + column;
                const float row_phase        = std::numbers::pi_v<float> * static_cast<float>(row) / static_cast<float>(rows - 1u);
                const float column_phase     = 2.0F * std::numbers::pi_v<float> * static_cast<float>(column) / static_cast<float>(columns - 1u);
                initial_positions[particle].z = initial_perturbation * std::sin(row_phase) * std::sin(column_phase);
            }
        }
        simulation::upload(stream, initial_positions, current_state.positions);
        simulation::upload(stream, initial_positions, next_state.positions);
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
        std::array<std::vector<float>, 6u> state{
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

        float maximum_material_stretch       = 0.0F;
        float maximum_absolute_material_shear = 0.0F;
        for (std::size_t triangle_index = 0uz; triangle_index < model.configuration.triangles.size(); ++triangle_index) {
            const deformables::cloth::Triangle triangle                               = model.configuration.triangles[triangle_index];
            const deformables::cloth::TriangleMaterialCoordinates<float> coordinates = model.configuration.material_coordinates[triangle_index];
            const float delta_u_first  = coordinates.second.u - coordinates.first.u;
            const float delta_v_first  = coordinates.second.v - coordinates.first.v;
            const float delta_u_second = coordinates.third.u - coordinates.first.u;
            const float delta_v_second = coordinates.third.v - coordinates.first.v;
            const float inverse_determinant = 1.0F / (delta_u_first * delta_v_second - delta_u_second * delta_v_first);
            const float inverse_00 = delta_v_second * inverse_determinant;
            const float inverse_01 = -delta_u_second * inverse_determinant;
            const float inverse_10 = -delta_v_first * inverse_determinant;
            const float inverse_11 = delta_u_first * inverse_determinant;
            const Vector3<float> first{.x = state[0][triangle.first], .y = state[1][triangle.first], .z = state[2][triangle.first]};
            const Vector3<float> second{.x = state[0][triangle.second], .y = state[1][triangle.second], .z = state[2][triangle.second]};
            const Vector3<float> third{.x = state[0][triangle.third], .y = state[1][triangle.third], .z = state[2][triangle.third]};
            const Vector3<float> first_edge  = second - first;
            const Vector3<float> second_edge = third - first;
            const Vector3<float> u_derivative = inverse_00 * first_edge + inverse_10 * second_edge;
            const Vector3<float> v_derivative = inverse_01 * first_edge + inverse_11 * second_edge;
            maximum_material_stretch        = std::max(maximum_material_stretch, std::max(length(u_derivative), length(v_derivative)));
            maximum_absolute_material_shear = std::max(maximum_absolute_material_shear, std::abs(dot(u_derivative, v_derivative)));
        }

        float maximum_fixed_position_error = 0.0F;
        for (const std::uint32_t particle : fixed_particles) {
            const Vector3<float> position{.x = state[0][particle], .y = state[1][particle], .z = state[2][particle]};
            maximum_fixed_position_error = std::max(maximum_fixed_position_error, length(position - model.configuration.rest_positions[particle]));
        }

        return {
            .frames                          = frame_count,
            .physical_time                   = static_cast<double>(frame_count) * time_step,
            .maximum_material_stretch        = maximum_material_stretch,
            .maximum_absolute_material_shear = maximum_absolute_material_shear,
            .maximum_fixed_position_error    = maximum_fixed_position_error,
            .probe_position                  = {.x = state[0][probe_particle], .y = state[1][probe_particle], .z = state[2][probe_particle]},
            .probe_velocity                  = {.x = state[3][probe_particle], .y = state[4][probe_particle], .z = state[5][probe_particle]},
        };
    }
} // namespace physica::examples::cloth::baraff_witkin
