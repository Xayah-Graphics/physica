module;

#include <physica/cuda.h>

export module physica.example.fluids.liquid.pbf_dam_break;

import std;
import physica.fluids.liquid.domain;
import physica.fluids.liquid.pbf;

export namespace physica::examples::pbf_dam_break {
    struct Simulation final {
        inline static constexpr std::array<std::uint32_t, 3u> lattice{16u, 28u, 12u};
        inline static constexpr std::uint32_t particle_count = lattice[0] * lattice[1] * lattice[2];
        inline static constexpr float spacing                = 0.025F;
        inline static constexpr float time_step              = 1.0F / 240.0F;
        inline static constexpr float particle_radius        = 0.010F;
        inline static constexpr float support_radius         = 0.050F;

        ::cuda::stream stream;
        const fluids::liquid::DomainConfiguration configuration;

    private:
        fluids::liquid::Domain domain;
        fluids::liquid::pbf::Solver solver;

    public:
        fluids::liquid::pbf::Solver::State current_state;
        fluids::liquid::pbf::Solver::StepCache step_cache;
        std::uint64_t step_index{};
        double physical_time{};

        Simulation();

        Simulation(const Simulation&)            = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&)                 = delete;
        Simulation& operator=(Simulation&&)      = delete;

        void reset();
        void step();

    private:
        fluids::liquid::pbf::Solver::State next_state;
        fluids::liquid::pbf::Solver::Control control;
        fluids::liquid::pbf::Solver::Parameters parameters;
        fluids::liquid::pbf::Solver::Workspace workspace;

        [[nodiscard]] static fluids::liquid::DomainConfiguration create_configuration();
    };

    Simulation::Simulation() : stream{::cuda::devices[0]}, configuration(create_configuration()), domain(configuration, stream), solver({.pressure_iterations = 5u, .checkpoint_interval = 2u}), current_state(solver.allocate_state(domain)), step_cache(solver.allocate_step_cache(domain)), next_state(solver.allocate_state(domain)), control(solver.allocate_control(domain)), parameters(solver.allocate_parameters(domain)), workspace(solver.allocate_workspace(domain)) {
        std::vector<float> values(particle_count, 1000.0F * spacing * spacing * spacing);
        ::cuda::copy_bytes(stream, values, parameters.masses);
        std::ranges::fill(values, 1000.0F);
        ::cuda::copy_bytes(stream, values, parameters.rest_densities);
        std::ranges::fill(values, 0.0F);
        ::cuda::copy_bytes(stream, values, parameters.viscosities);
        ::cuda::copy_bytes(stream, values, parameters.surface_tensions);
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
                    positions[0][particle]       = 0.08F + static_cast<float>(x) * spacing;
                    positions[1][particle]       = 0.08F + static_cast<float>(y) * spacing;
                    positions[2][particle]       = 0.08F + static_cast<float>(z) * spacing;
                }
        ::cuda::copy_bytes(stream, positions[0], current_state.positions.x);
        ::cuda::copy_bytes(stream, positions[1], current_state.positions.y);
        ::cuda::copy_bytes(stream, positions[2], current_state.positions.z);
        ::cuda::fill_bytes(stream, current_state.velocities.x, 0u);
        ::cuda::fill_bytes(stream, current_state.velocities.y, 0u);
        ::cuda::fill_bytes(stream, current_state.velocities.z, 0u);
        ::cuda::fill_bytes(stream, control.external_accelerations.x, 0u);
        ::cuda::fill_bytes(stream, control.external_accelerations.y, 0u);
        ::cuda::fill_bytes(stream, control.external_accelerations.z, 0u);
        ::cuda::fill_bytes(stream, step_cache.vorticity_magnitudes.values, 0u);
        current_state.step_index = 0u;
        solver.copy_state(domain, current_state, next_state);
        stream.sync();
        step_index    = 0u;
        physical_time = 0.0;
    }

    void Simulation::step() {
        solver.forward(domain, current_state, control, parameters, next_state, step_cache, workspace);
        std::swap(current_state, next_state);
        step_index    = current_state.step_index;
        physical_time = static_cast<double>(step_index) * time_step;
    }

    fluids::liquid::DomainConfiguration Simulation::create_configuration() {
        return {
            .particle_count  = particle_count,
            .time_step       = time_step,
            .support_radius  = support_radius,
            .particle_radius = particle_radius,
            .boundary =
                {
                    .minimum  = {.x = 0.0F, .y = 0.0F, .z = 0.0F},
                    .maximum  = {.x = 0.8F, .y = 1.1F, .z = 0.5F},
                    .velocity = {},
                    .no_slip  = true,
                },
            .boundary_particles = {},
        };
    }
} // namespace physica::examples::pbf_dam_break
