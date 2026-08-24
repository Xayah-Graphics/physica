module;

#include "simulation_kernels.h"
#include <physica/cuda.h>

export module physica.example.fluids.gas.smoke;

import std;
import physica.fluids.gas.smoke;

export namespace physica::examples::smoke {
    struct Simulation final {
        inline static constexpr std::array<std::uint32_t, 3> resolution{128u, 192u, 128u};
        inline static constexpr float cell_size = 1.0F / 128.0F;
        inline static constexpr float time_step = 1.0F / 120.0F;
        inline static constexpr std::uint32_t pressure_iterations = 160u;
        inline static constexpr float density_source_rate = 4.0F;
        inline static constexpr float temperature_source_rate = 8.0F;
        inline static constexpr float ambient_temperature = 0.0F;
        inline static constexpr float density_buoyancy = -0.1F;
        inline static constexpr float temperature_buoyancy = 1.0F;
        inline static constexpr float vorticity_confinement = 2.0F;
        inline static constexpr fluids::gas::smoke::Vector3 left_source_center{0.25F, 0.10F, 0.36F};
        inline static constexpr fluids::gas::smoke::Vector3 right_source_center{0.75F, 0.10F, 0.64F};
        inline static constexpr float source_radius = 0.055F;
        inline static constexpr fluids::gas::smoke::Vector3 left_acceleration{3.5F, 5.0F, 1.8F};
        inline static constexpr fluids::gas::smoke::Vector3 right_acceleration{-3.5F, 5.0F, -1.8F};
        inline static constexpr float pulse_period = 0.9F;

        ::cuda::stream stream;
        const fluids::gas::smoke::DomainConfiguration configuration;

    private:
        fluids::gas::smoke::Solver<
            fluids::gas::smoke::SemiLagrangianRK2,
            fluids::gas::smoke::BuoyancyVorticityForces,
            fluids::gas::smoke::MacProjection<fluids::gas::smoke::RedBlackGaussSeidel>
        > solver;

    public:
        fluids::gas::smoke::State current_state;
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
        fluids::gas::smoke::State next_state;
        fluids::gas::smoke::Control control;
        fluids::gas::smoke::BuoyancyVorticityForces::Parameters parameters;
        fluids::gas::smoke::Solver<
            fluids::gas::smoke::SemiLagrangianRK2,
            fluids::gas::smoke::BuoyancyVorticityForces,
            fluids::gas::smoke::MacProjection<fluids::gas::smoke::RedBlackGaussSeidel>
        >::StepCache step_cache;

        [[nodiscard]] static fluids::gas::smoke::DomainConfiguration create_domain_configuration();
    };

    Simulation::Simulation()
        : stream{::cuda::devices[0]},
          configuration{create_domain_configuration()},
          solver{
              configuration,
              fluids::gas::smoke::SemiLagrangianRK2::Configuration{},
              fluids::gas::smoke::BuoyancyVorticityForces::Configuration{.vorticity_confinement_enabled = true},
              fluids::gas::smoke::MacProjection<fluids::gas::smoke::RedBlackGaussSeidel>::Configuration{.pressure = {.iterations = pressure_iterations}},
              fluids::gas::smoke::ExecutionMode::forward,
              stream
          },
          current_state{solver.allocate_state()},
          next_state{solver.allocate_state()},
          control{solver.allocate_control()},
          parameters{solver.allocate_parameters()},
          step_cache{solver.allocate_step_cache()} {
        ::cuda::copy_bytes(stream, ::cuda::std::span{&ambient_temperature, 1uz}, parameters.ambient_temperature);
        ::cuda::copy_bytes(stream, ::cuda::std::span{&density_buoyancy, 1uz}, parameters.density_buoyancy);
        ::cuda::copy_bytes(stream, ::cuda::std::span{&temperature_buoyancy, 1uz}, parameters.temperature_buoyancy);
        ::cuda::copy_bytes(stream, ::cuda::std::span{&vorticity_confinement, 1uz}, parameters.vorticity_confinement);
    }

    void Simulation::reset() {
        const auto clear = [this](fluids::gas::smoke::State& state) {
            ::cuda::fill_bytes(stream, state.density.values, 0u);
            ::cuda::fill_bytes(stream, state.temperature.values, 0u);
            ::cuda::fill_bytes(stream, state.velocity.x, 0u);
            ::cuda::fill_bytes(stream, state.velocity.y, 0u);
            ::cuda::fill_bytes(stream, state.velocity.z, 0u);
        };
        clear(current_state);
        clear(next_state);
        step_index = 0u;
        physical_time = 0.0;
    }

    void Simulation::step() {
        simulation_cuda::write_control(
            stream,
            {.nx = resolution[0], .ny = resolution[1], .nz = resolution[2], .cell_size = cell_size, .time_step = time_step},
            step_index,
            pulse_period,
            {.x = left_source_center.x, .y = left_source_center.y, .z = left_source_center.z},
            {.x = right_source_center.x, .y = right_source_center.y, .z = right_source_center.z},
            source_radius,
            density_source_rate,
            temperature_source_rate,
            {.x = left_acceleration.x, .y = left_acceleration.y, .z = left_acceleration.z},
            {.x = right_acceleration.x, .y = right_acceleration.y, .z = right_acceleration.z},
            control.density_source.values.data(),
            control.temperature_source.values.data(),
            control.external_acceleration.x.values.data(),
            control.external_acceleration.y.values.data(),
            control.external_acceleration.z.values.data()
        );
        solver.forward_step(current_state, control, parameters, next_state, step_cache);
        std::swap(current_state, next_state);
        ++step_index;
        physical_time = static_cast<double>(step_index) * time_step;
    }

    fluids::gas::smoke::DomainConfiguration Simulation::create_domain_configuration() {
        fluids::gas::smoke::DomainConfiguration result{
            .resolution = resolution,
            .cell_size = cell_size,
            .time_step = time_step,
        };
        result.velocity_boundary.x_min.mode = fluids::gas::smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.x_max.mode = fluids::gas::smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.y_min.mode = fluids::gas::smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.y_max.mode = fluids::gas::smoke::VelocityBoundaryMode::zero_gradient;
        result.velocity_boundary.z_min.mode = fluids::gas::smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.z_max.mode = fluids::gas::smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.pressure_boundary.y_max.mode = fluids::gas::smoke::ScalarBoundaryMode::fixed_value;
        result.density_boundary.y_max.mode = fluids::gas::smoke::ScalarBoundaryMode::fixed_value;
        result.temperature_boundary.y_max.mode = fluids::gas::smoke::ScalarBoundaryMode::fixed_value;
        return result;
    }
} // namespace physica::examples::smoke
