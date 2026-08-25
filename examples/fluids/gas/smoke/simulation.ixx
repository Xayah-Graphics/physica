module;

#include "simulation_kernels.h"
#include <physica/cuda.h>

export module physica.example.fluids.gas.smoke;

import std;
import physica.fluids.gas.domain;
import physica.fluids.gas.operators.force;
import physica.fluids.gas.operators.advection;
import physica.fluids.gas.operators.diffusion;
import physica.fluids.gas.operators.projection;
import physica.fluids.gas.smoke;

export namespace physica::examples::smoke {
    struct Simulation final {
        inline static constexpr std::array<std::uint32_t, 3> resolution{128u, 192u, 128u};
        inline static constexpr float cell_size                   = 1.0F / 128.0F;
        inline static constexpr float time_step                   = 1.0F / 120.0F;
        inline static constexpr std::uint32_t pressure_iterations = 160u;
        inline static constexpr float density_source_rate         = 4.0F;
        inline static constexpr float temperature_source_rate     = 8.0F;
        inline static constexpr float ambient_temperature         = 0.0F;
        inline static constexpr float density_buoyancy            = -0.1F;
        inline static constexpr float temperature_buoyancy        = 1.0F;
        inline static constexpr float vorticity_confinement       = 2.0F;
        inline static constexpr fluids::gas::Vector3 left_source_center{0.25F, 0.10F, 0.36F};
        inline static constexpr fluids::gas::Vector3 right_source_center{0.75F, 0.10F, 0.64F};
        inline static constexpr float source_radius = 0.055F;
        inline static constexpr fluids::gas::Vector3 left_acceleration{3.5F, 5.0F, 1.8F};
        inline static constexpr fluids::gas::Vector3 right_acceleration{-3.5F, 5.0F, -1.8F};
        inline static constexpr float pulse_period = 0.9F;

        ::cuda::stream stream;
        const fluids::gas::DomainConfiguration configuration;
        fluids::gas::Domain domain;

    private:
        fluids::gas::smoke::Solver<fluids::gas::operators::SemiLagrangianRK2, fluids::gas::operators::IdentityVelocityDiffusion, fluids::gas::operators::ThermalBuoyancyVorticity, fluids::gas::operators::MacProjection<fluids::gas::operators::RedBlackGaussSeidel>> solver;

    public:
        fluids::gas::smoke::State current_state;
        std::uint64_t step_index = 0u;
        double physical_time     = 0.0;

        Simulation();

        Simulation(const Simulation&)            = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&)                 = delete;
        Simulation& operator=(Simulation&&)      = delete;

        void reset();
        void step();

    private:
        fluids::gas::smoke::State next_state;
        fluids::gas::smoke::Control control;
        fluids::gas::operators::ThermalBuoyancyVorticity::Parameters parameters;
        decltype(solver)::StepCache step_cache;
        decltype(solver)::Workspace workspace;

        [[nodiscard]] static fluids::gas::DomainConfiguration create_domain_configuration();
    };

    Simulation::Simulation()
        : stream{::cuda::devices[0]}, configuration{create_domain_configuration()}, domain{configuration, stream}, solver{domain,
                                                                                                                       decltype(solver)::Configuration{
                                                                                                                           .force = {.vorticity_confinement_enabled = true},
                                                                                                                           .projection =
                                                                                                                               {
                                                                                                                                   .boundary = {.y_max = {.mode = fluids::gas::ScalarBoundaryMode::fixed_value}},
                                                                                                                                   .pressure = {.iterations = pressure_iterations},
                                                                                                                               },
                                                                                                                           .density_boundary     = {.y_max = {.mode = fluids::gas::ScalarBoundaryMode::fixed_value}},
                                                                                                                           .temperature_boundary = {.y_max = {.mode = fluids::gas::ScalarBoundaryMode::fixed_value}},
                                                                                                                       }},
          current_state{solver.allocate_state(domain)}, next_state{solver.allocate_state(domain)}, control{solver.allocate_control(domain)}, parameters{solver.allocate_parameters(domain)}, step_cache{solver.allocate_step_cache(domain)}, workspace{solver.allocate_workspace(domain)} {
        const std::array force_parameters{ambient_temperature, density_buoyancy, temperature_buoyancy, vorticity_confinement};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{force_parameters.data(), force_parameters.size()}, parameters.values);
    }

    void Simulation::reset() {
        domain.clear(current_state.density);
        domain.clear(current_state.temperature);
        domain.clear(current_state.velocity);
        domain.clear(next_state.density);
        domain.clear(next_state.temperature);
        domain.clear(next_state.velocity);
        step_index    = 0u;
        physical_time = 0.0;
    }

    void Simulation::step() {
        simulation_cuda::write_control(stream, {.nx = resolution[0], .ny = resolution[1], .nz = resolution[2], .cell_size = cell_size, .time_step = time_step}, step_index, pulse_period, {.x = left_source_center.x, .y = left_source_center.y, .z = left_source_center.z}, {.x = right_source_center.x, .y = right_source_center.y, .z = right_source_center.z}, source_radius, density_source_rate, temperature_source_rate, {.x = left_acceleration.x, .y = left_acceleration.y, .z = left_acceleration.z}, {.x = right_acceleration.x, .y = right_acceleration.y, .z = right_acceleration.z}, control.density_source.values.data(), control.temperature_source.values.data(), control.external_acceleration.x.values.data(), control.external_acceleration.y.values.data(), control.external_acceleration.z.values.data());
        solver.forward(domain, current_state, control, parameters, next_state, step_cache, workspace);
        std::swap(current_state, next_state);
        ++step_index;
        physical_time = static_cast<double>(step_index) * time_step;
    }

    fluids::gas::DomainConfiguration Simulation::create_domain_configuration() {
        fluids::gas::DomainConfiguration result{
            .resolution = resolution,
            .cell_size  = cell_size,
            .time_step  = time_step,
        };
        result.velocity_boundary.x_min.mode = fluids::gas::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.x_max.mode = fluids::gas::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.y_min.mode = fluids::gas::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.y_max.mode = fluids::gas::VelocityBoundaryMode::zero_gradient;
        result.velocity_boundary.z_min.mode = fluids::gas::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.z_max.mode = fluids::gas::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        return result;
    }
} // namespace physica::examples::smoke
