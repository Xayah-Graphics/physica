module;

#include "simulation_kernels.h"
#include <physica/cuda.h>

export module physica.example.deformables.cloth;

import std;
import physica.deformables.cloth;

export namespace physica::examples::cloth {
    struct Simulation final {
        inline static constexpr std::uint32_t rows = 64u;
        inline static constexpr std::uint32_t columns = 96u;
        inline static constexpr float width = 3.0F;
        inline static constexpr float height = 2.0F;
        inline static constexpr float time_step = 1.0F / 240.0F;
        inline static constexpr std::uint32_t integration_substeps = 8u;
        inline static constexpr float mass = 0.0125F;
        inline static constexpr float stretch_stiffness = 600.0F;
        inline static constexpr float stretch_damping = 1.2F;
        inline static constexpr float bending_stiffness = 8.0F;
        inline static constexpr float bending_damping = 0.4F;
        inline static constexpr float gravity_y = -0.1F;
        inline static constexpr float wind_speed = 6.0F;
        inline static constexpr float gust_strength = 0.35F;
        inline static constexpr float gust_frequency = 0.9F;
        inline static constexpr float air_density = 1.225F;
        inline static constexpr float drag_coefficient = 1.0F;
        inline static constexpr float skin_drag_coefficient = 0.10F;
        inline static constexpr float wind_ramp_duration = 0.5F;

        ::cuda::stream stream;
        const deformables::cloth::DomainConfiguration configuration;

    private:
        deformables::cloth::Solver<
            deformables::cloth::MassSpringForces,
            deformables::cloth::SemiImplicitEuler,
            deformables::cloth::FixedPositionConstraints
        > solver;

    public:
        deformables::cloth::State current_state;
        std::uint64_t step_index = 0u;
        double physical_time = 0.0;

        Simulation();

        Simulation(const Simulation&) = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&) = delete;
        Simulation& operator=(Simulation&&) = delete;

        void reset();
        void step();

    private:
        deformables::cloth::State next_state;
        deformables::cloth::Control control;
        deformables::cloth::Solver<
            deformables::cloth::MassSpringForces,
            deformables::cloth::SemiImplicitEuler,
            deformables::cloth::FixedPositionConstraints
        >::Parameters parameters;
        deformables::cloth::Solver<
            deformables::cloth::MassSpringForces,
            deformables::cloth::SemiImplicitEuler,
            deformables::cloth::FixedPositionConstraints
        >::StepCache step_cache;

        [[nodiscard]] static deformables::cloth::DomainConfiguration create_domain_configuration();
    };

    Simulation::Simulation()
        : stream{::cuda::devices[0]},
          configuration(create_domain_configuration()),
          solver(configuration, {.gravity = {.x = 0.0F, .y = gravity_y, .z = 0.0F}}, {.time_step = time_step / static_cast<float>(integration_substeps)}, deformables::cloth::ExecutionMode::forward, stream),
          current_state(solver.allocate_state()),
          next_state(solver.allocate_state()),
          control(solver.allocate_control()),
          parameters(solver.allocate_parameters()),
          step_cache(solver.allocate_step_cache()) {
        const std::vector<float> masses(configuration.rest_positions.size(), mass);
        const std::vector<float> stretch_stiffnesses(parameters.forces.stretch.stiffnesses.values.size(), stretch_stiffness);
        const std::vector<float> stretch_dampings(parameters.forces.stretch.dampings.values.size(), stretch_damping);
        const std::vector<float> bending_stiffnesses(parameters.forces.bending.stiffnesses.values.size(), bending_stiffness);
        const std::vector<float> bending_dampings(parameters.forces.bending.dampings.values.size(), bending_damping);
        ::cuda::copy_bytes(stream, masses, parameters.masses.values);
        ::cuda::copy_bytes(stream, stretch_stiffnesses, parameters.forces.stretch.stiffnesses.values);
        ::cuda::copy_bytes(stream, stretch_dampings, parameters.forces.stretch.dampings.values);
        ::cuda::copy_bytes(stream, bending_stiffnesses, parameters.forces.bending.stiffnesses.values);
        ::cuda::copy_bytes(stream, bending_dampings, parameters.forces.bending.dampings.values);
        stream.sync();
        reset();
    }

    void Simulation::reset() {
        solver.reset_state(current_state);
        solver.reset_state(next_state);
        ::cuda::fill_bytes(stream, control.external_forces.x.values, 0u);
        ::cuda::fill_bytes(stream, control.external_forces.y.values, 0u);
        ::cuda::fill_bytes(stream, control.external_forces.z.values, 0u);
        step_index = 0u;
        physical_time = 0.0;
    }

    void Simulation::step() {
        constexpr float substep_time_step = time_step / static_cast<float>(integration_substeps);
        for (std::uint32_t substep = 0u; substep < integration_substeps; ++substep) {
            simulation_cuda::write_control(
                stream,
                {.rows = rows, .columns = columns, .width = width, .height = height},
                step_index * integration_substeps + substep,
                substep_time_step,
                {.speed = wind_speed, .gust_strength = gust_strength, .gust_frequency = gust_frequency, .air_density = air_density, .drag_coefficient = drag_coefficient, .skin_drag_coefficient = skin_drag_coefficient, .ramp_duration = wind_ramp_duration},
                current_state.positions.x.values.data(),
                current_state.positions.y.values.data(),
                current_state.positions.z.values.data(),
                current_state.velocities.x.values.data(),
                current_state.velocities.y.values.data(),
                current_state.velocities.z.values.data(),
                control.external_forces.x.values.data(),
                control.external_forces.y.values.data(),
                control.external_forces.z.values.data()
            );
            solver.forward_step(current_state, control, parameters, next_state, step_cache);
            std::swap(current_state, next_state);
        }
        ++step_index;
        physical_time = static_cast<double>(step_index) * time_step;
    }

    deformables::cloth::DomainConfiguration Simulation::create_domain_configuration() {
        deformables::cloth::DomainConfiguration result{
            .rest_positions = std::vector<deformables::cloth::Vector3>(static_cast<std::size_t>(rows) * columns),
            .triangles = {},
            .anchors = std::vector<std::optional<deformables::cloth::Vector3>>(static_cast<std::size_t>(rows) * columns),
        };
        const float spacing_x = width / static_cast<float>(columns - 1u);
        const float spacing_y = height / static_cast<float>(rows - 1u);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t column = 0u; column < columns; ++column) {
                const std::uint32_t particle = row * columns + column;
                result.rest_positions[particle] = {.x = static_cast<float>(column) * spacing_x, .y = -static_cast<float>(row) * spacing_y, .z = 0.0F};
                if (column == 0u) result.anchors[particle] = result.rest_positions[particle];
            }
        }
        result.triangles.reserve(static_cast<std::size_t>(rows - 1u) * (columns - 1u) * 2uz);
        for (std::uint32_t row = 0u; row + 1u < rows; ++row) {
            for (std::uint32_t column = 0u; column + 1u < columns; ++column) {
                const std::uint32_t top_left = row * columns + column;
                const std::uint32_t top_right = top_left + 1u;
                const std::uint32_t bottom_left = top_left + columns;
                const std::uint32_t bottom_right = bottom_left + 1u;
                if ((row + column) % 2u == 0u) {
                    result.triangles.push_back({.first = top_left, .second = top_right, .third = bottom_right});
                    result.triangles.push_back({.first = top_left, .second = bottom_right, .third = bottom_left});
                } else {
                    result.triangles.push_back({.first = top_left, .second = top_right, .third = bottom_left});
                    result.triangles.push_back({.first = top_right, .second = bottom_right, .third = bottom_left});
                }
            }
        }
        return result;
    }
} // namespace physica::examples::cloth
