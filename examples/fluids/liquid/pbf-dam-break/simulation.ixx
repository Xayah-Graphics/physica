module;

#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/stream>

export module physica.example.fluids.liquid.pbf_dam_break;

import std;
import physica.fluids.liquid.particle;

export namespace physica::examples::pbf_dam_break {
    struct Simulation final {
        inline static constexpr std::array<std::uint32_t, 3u> lattice{16u, 28u, 12u};
        inline static constexpr std::uint32_t particle_count = lattice[0] * lattice[1] * lattice[2];
        inline static constexpr float spacing = 0.025F;
        inline static constexpr float time_step = 1.0F / 240.0F;
        inline static constexpr float particle_radius = 0.010F;
        inline static constexpr float support_radius = 0.050F;

        ::cuda::stream stream;
        const fluids::liquid::particle::DomainConfiguration configuration;

    private:
        fluids::liquid::particle::PBF solver;

    public:
        fluids::liquid::particle::PBF::State current_state;
        fluids::liquid::particle::PBF::StepCache step_cache;
        std::uint64_t step_index{};
        double physical_time{};

        Simulation();

        Simulation(const Simulation&) = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&) = delete;
        Simulation& operator=(Simulation&&) = delete;

        void reset();
        void step();

    private:
        fluids::liquid::particle::PBF::State next_state;
        fluids::liquid::particle::Control control;
        fluids::liquid::particle::PBF::Parameters parameters;

        [[nodiscard]] static fluids::liquid::particle::DomainConfiguration create_configuration();
    };

    Simulation::Simulation()
        : stream{::cuda::devices[0]},
          configuration(create_configuration()),
          solver(configuration, {.pressure_iterations = 5u, .checkpoint_interval = 2u}, fluids::liquid::particle::ExecutionMode::forward, stream),
          current_state(solver.allocate_state()),
          step_cache(solver.allocate_step_cache()),
          next_state(solver.allocate_state()),
          control(solver.allocate_control()),
          parameters(solver.allocate_parameters()) {
        std::vector<float> values(particle_count, 1000.0F * spacing * spacing * spacing);
        ::cuda::copy_bytes(stream, values, parameters.particles.masses);
        std::ranges::fill(values, 1000.0F);
        ::cuda::copy_bytes(stream, values, parameters.particles.rest_densities);
        std::ranges::fill(values, 0.0F);
        ::cuda::copy_bytes(stream, values, parameters.particles.viscosities);
        ::cuda::copy_bytes(stream, values, parameters.particles.surface_tensions);
        std::ranges::fill(values, 1000.0F);
        ::cuda::copy_bytes(stream, values, parameters.relaxation);
        std::ranges::fill(values, 0.0F);
        ::cuda::copy_bytes(stream, values, parameters.artificial_pressure_strength);
        std::ranges::fill(values, 4.0F);
        ::cuda::copy_bytes(stream, values, parameters.artificial_pressure_exponent);
        std::ranges::fill(values, 0.3F * support_radius);
        ::cuda::copy_bytes(stream, values, parameters.artificial_pressure_radius);
        std::ranges::fill(values, 0.0F);
        ::cuda::copy_bytes(stream, values, parameters.xsph_viscosity);
        ::cuda::copy_bytes(stream, values, parameters.vorticity_confinement);
        reset();
    }

    void Simulation::reset() {
        std::array<std::vector<float>, 3u> positions{
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
            std::vector<float>(particle_count),
        };
        for (std::uint32_t z = 0u; z < lattice[2]; ++z)
            for (std::uint32_t y = 0u; y < lattice[1]; ++y)
                for (std::uint32_t x = 0u; x < lattice[0]; ++x) {
                    const std::uint32_t particle = (z * lattice[1] + y) * lattice[0] + x;
                    positions[0][particle] = 0.08F + static_cast<float>(x) * spacing;
                    positions[1][particle] = 0.08F + static_cast<float>(y) * spacing;
                    positions[2][particle] = 0.08F + static_cast<float>(z) * spacing;
                }
        ::cuda::copy_bytes(stream, positions[0], current_state.particles.positions.x);
        ::cuda::copy_bytes(stream, positions[1], current_state.particles.positions.y);
        ::cuda::copy_bytes(stream, positions[2], current_state.particles.positions.z);
        ::cuda::fill_bytes(stream, current_state.particles.velocities.x, 0u);
        ::cuda::fill_bytes(stream, current_state.particles.velocities.y, 0u);
        ::cuda::fill_bytes(stream, current_state.particles.velocities.z, 0u);
        ::cuda::fill_bytes(stream, control.external_accelerations.x, 0u);
        ::cuda::fill_bytes(stream, control.external_accelerations.y, 0u);
        ::cuda::fill_bytes(stream, control.external_accelerations.z, 0u);
        ::cuda::fill_bytes(stream, step_cache.vorticity_magnitudes.values, 0u);
        current_state.particles.step_index = 0u;
        solver.copy_state(current_state, next_state);
        stream.sync();
        step_index = 0u;
        physical_time = 0.0;
    }

    void Simulation::step() {
        solver.forward_step(current_state, control, parameters, next_state, step_cache);
        std::swap(current_state, next_state);
        step_index = current_state.particles.step_index;
        physical_time = static_cast<double>(step_index) * time_step;
    }

    fluids::liquid::particle::DomainConfiguration Simulation::create_configuration() {
        return {
            .particle_count = particle_count,
            .time_step = time_step,
            .support_radius = support_radius,
            .particle_radius = particle_radius,
            .gravity = {.x = 0.0F, .y = -9.81F, .z = 0.0F},
            .boundary = {
                .minimum = {.x = 0.0F, .y = 0.0F, .z = 0.0F},
                .maximum = {.x = 0.8F, .y = 1.1F, .z = 0.5F},
                .velocity = {},
                .no_slip = true,
            },
            .boundary_particles = {},
        };
    }
} // namespace physica::examples::pbf_dam_break
