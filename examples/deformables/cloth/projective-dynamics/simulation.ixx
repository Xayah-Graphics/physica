module;

#include <physica/cuda.h>

export module physica.example.deformables.cloth.projective_dynamics;

import std;
import physica.deformables.cloth.model;
import physica.deformables.cloth.solvers.projective_dynamics;
import physica.example.deformables.cloth.support.scene;

export namespace physica::examples::cloth::projective_dynamics {
    struct Summary final {
        std::uint32_t frames;
        double physical_time;
        float maximum_material_stretch_ratio;
        float maximum_deformation_metric;
        float maximum_bending_distance_error;
        float maximum_displacement;
        float maximum_fixed_position_error;
        Vector3<float> probe_position;
        Vector3<float> probe_velocity;
    };

    struct Simulation final {
        inline static constexpr std::uint32_t rows                   = 8u;
        inline static constexpr std::uint32_t columns                = 12u;
        inline static constexpr float width                          = 1.4F;
        inline static constexpr float height                         = 0.9F;
        inline static constexpr float time_step                      = 1.0F / 120.0F;
        inline static constexpr std::uint32_t frame_count            = 180u;
        inline static constexpr std::uint32_t global_iteration_count = 12u;
        inline static constexpr float gravity_y                      = -9.81F;
        inline static constexpr float mass                           = 0.04F;
        inline static constexpr float membrane_stiffness             = 800.0F;
        inline static constexpr float bending_stiffness              = 8.0F;
        inline static constexpr float initial_perturbation            = 0.06F;
        inline static constexpr std::uint32_t probe_particle         = (rows - 1u) * columns + columns / 2u;
        inline static constexpr std::array<std::uint32_t, 2u> fixed_particles{0u, columns - 1u};

        ::cuda::stream stream;

    private:
        deformables::cloth::Model<float> model;
        deformables::cloth::solvers::projective_dynamics::Solver solver;
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
                  .time_step              = time_step,
                  .global_iteration_count = global_iteration_count,
                  .gravity                = {.x = 0.0F, .y = gravity_y, .z = 0.0F},
                  .membrane_stiffness     = membrane_stiffness,
                  .bending_stiffness      = bending_stiffness,
                  .masses                 = std::vector<float>(model.particle_count, mass),
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

        std::vector<Vector3<float>> positions(particle_count);
        for (std::size_t particle = 0uz; particle < particle_count; ++particle) positions[particle] = {.x = state[0][particle], .y = state[1][particle], .z = state[2][particle]};

        float maximum_material_stretch_ratio = 0.0F;
        float maximum_deformation_metric      = 0.0F;
        for (std::size_t triangle_index = 0uz; triangle_index < model.configuration.triangles.size(); ++triangle_index) {
            const deformables::cloth::Triangle triangle = model.configuration.triangles[triangle_index];
            const deformables::cloth::TriangleMaterialCoordinates<float> material = model.configuration.material_coordinates[triangle_index];
            const float material_10_u = material.second.u - material.first.u;
            const float material_10_v = material.second.v - material.first.v;
            const float material_20_u = material.third.u - material.first.u;
            const float material_20_v = material.third.v - material.first.v;
            const float determinant   = material_10_u * material_20_v - material_20_u * material_10_v;
            const float inverse_00    = material_20_v / determinant;
            const float inverse_01    = -material_20_u / determinant;
            const float inverse_10    = -material_10_v / determinant;
            const float inverse_11    = material_10_u / determinant;
            const Vector3<float> first_edge  = positions[triangle.second] - positions[triangle.first];
            const Vector3<float> second_edge = positions[triangle.third] - positions[triangle.first];
            const Vector3<float> deformation_gradient_first_column  = inverse_00 * first_edge + inverse_10 * second_edge;
            const Vector3<float> deformation_gradient_second_column = inverse_01 * first_edge + inverse_11 * second_edge;
            const float metric_00 = dot(deformation_gradient_first_column, deformation_gradient_first_column);
            const float metric_01 = dot(deformation_gradient_first_column, deformation_gradient_second_column);
            const float metric_11 = dot(deformation_gradient_second_column, deformation_gradient_second_column);
            const float largest_eigenvalue = 0.5F * (metric_00 + metric_11 + std::sqrt((metric_00 - metric_11) * (metric_00 - metric_11) + 4.0F * metric_01 * metric_01));
            const float deformation_metric = std::sqrt((metric_00 - 1.0F) * (metric_00 - 1.0F) + 2.0F * metric_01 * metric_01 + (metric_11 - 1.0F) * (metric_11 - 1.0F));
            maximum_material_stretch_ratio = std::max(maximum_material_stretch_ratio, std::sqrt(largest_eigenvalue));
            maximum_deformation_metric      = std::max(maximum_deformation_metric, deformation_metric);
        }

        float maximum_bending_distance_error = 0.0F;
        for (const deformables::cloth::Hinge hinge : model.topology.hinges) {
            const float rest_length    = length(model.configuration.rest_positions[hinge.second_opposite] - model.configuration.rest_positions[hinge.first_opposite]);
            const float current_length = length(positions[hinge.second_opposite] - positions[hinge.first_opposite]);
            maximum_bending_distance_error = std::max(maximum_bending_distance_error, std::abs(current_length - rest_length));
        }

        float maximum_displacement         = 0.0F;
        float maximum_fixed_position_error = 0.0F;
        for (std::size_t particle = 0uz; particle < particle_count; ++particle) maximum_displacement = std::max(maximum_displacement, length(positions[particle] - model.configuration.rest_positions[particle]));
        for (const std::uint32_t particle : fixed_particles) maximum_fixed_position_error = std::max(maximum_fixed_position_error, length(positions[particle] - model.configuration.rest_positions[particle]));

        return {
            .frames                         = frame_count,
            .physical_time                  = static_cast<double>(frame_count) * time_step,
            .maximum_material_stretch_ratio = maximum_material_stretch_ratio,
            .maximum_deformation_metric     = maximum_deformation_metric,
            .maximum_bending_distance_error = maximum_bending_distance_error,
            .maximum_displacement           = maximum_displacement,
            .maximum_fixed_position_error   = maximum_fixed_position_error,
            .probe_position                 = positions[probe_particle],
            .probe_velocity                 = {.x = state[3][probe_particle], .y = state[4][probe_particle], .z = state[5][probe_particle]},
        };
    }
} // namespace physica::examples::cloth::projective_dynamics
