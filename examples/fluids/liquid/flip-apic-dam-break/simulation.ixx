module;

#include <physica/cuda.h>

export module physica.example.fluids.liquid.flip_apic_dam_break;

import std;
import physica.fluids.liquid.solvers.pic;

export namespace physica::examples::flip_apic_dam_break {
    struct Simulation final {
        inline static constexpr std::array<std::uint32_t, 3u> resolution{32u, 48u, 24u};
        inline static constexpr float cell_size                      = 0.025F;
        inline static constexpr float particle_radius                = 0.0045F;
        inline static constexpr std::uint32_t maximum_particle_count = 100000u;
        inline static constexpr std::array<std::uint32_t, 3u> initial_cells{12u, 30u, 20u};
        inline static constexpr std::uint32_t particles_per_cell     = 8u;
        inline static constexpr std::uint32_t initial_particle_count = initial_cells[0] * initial_cells[1] * initial_cells[2] * particles_per_cell;

        ::cuda::stream stream;
        const fluids::liquid::solvers::pic::ModelConfiguration configuration;
        fluids::liquid::solvers::pic::Model model;

    private:
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::FlipTransfer> flip_solver;
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::ApicTransfer> apic_solver;

    public:
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::FlipTransfer>::State flip_state;
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::FlipTransfer>::StepCache flip_cache;
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::ApicTransfer>::State apic_state;
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::ApicTransfer>::StepCache apic_cache;
        std::uint64_t step_index{};
        std::uint64_t flip_substep_count{};
        std::uint64_t apic_substep_count{};
        double physical_time{};
        fluids::liquid::solvers::pic::ParticleStep::Diagnostics flip_particle_diagnostics;
        fluids::liquid::solvers::pic::ParticleStep::Diagnostics apic_particle_diagnostics;

        Simulation();

        Simulation(const Simulation&)            = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&)                 = delete;
        Simulation& operator=(Simulation&&)      = delete;

        void reset();
        void step(double seconds);

    private:
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::FlipTransfer>::State flip_next_state;
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::FlipTransfer>::Workspace flip_workspace;
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::ApicTransfer>::State apic_next_state;
        fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::ApicTransfer>::Workspace apic_workspace;
        double flip_time{};
        double apic_time{};

        [[nodiscard]] static fluids::liquid::solvers::pic::ModelConfiguration create_configuration();
        [[nodiscard]] static fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::FlipTransfer>::Configuration create_flip_configuration();
        [[nodiscard]] static fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::ApicTransfer>::Configuration create_apic_configuration();

        template <class SolverType>
        void advance(SolverType& solver, typename SolverType::State& current_state, typename SolverType::State& next_state, typename SolverType::StepCache& cache, typename SolverType::Workspace& workspace, double target_time, double& algorithm_time, std::uint64_t& substep_count) {
            while (algorithm_time < target_time) {
                const float stable_time_step = solver.stable_time_step(model, current_state, workspace);
                const bool final_substep     = target_time - algorithm_time <= static_cast<double>(stable_time_step);
                const float time_step        = final_substep ? static_cast<float>(target_time - algorithm_time) : stable_time_step;
                solver.forward(model, static_cast<float>(algorithm_time), time_step, step_index * 65537u + substep_count, current_state, next_state, cache, workspace);
                std::swap(current_state, next_state);
                algorithm_time = final_substep ? target_time : algorithm_time + static_cast<double>(time_step);
                ++substep_count;
            }
        }
    };

    Simulation::Simulation() : stream{::cuda::devices[0]}, configuration(create_configuration()), model(configuration, stream), flip_solver(create_flip_configuration()), apic_solver(create_apic_configuration()), flip_state(flip_solver.allocate_state(model)), flip_cache(flip_solver.allocate_step_cache(model)), apic_state(apic_solver.allocate_state(model)), apic_cache(apic_solver.allocate_step_cache(model)), flip_next_state(flip_solver.allocate_state(model)), flip_workspace(flip_solver.allocate_workspace(model)), apic_next_state(apic_solver.allocate_state(model)), apic_workspace(apic_solver.allocate_workspace(model)) {
        reset();
    }

    void Simulation::reset() {
        std::array<std::vector<float>, 3u> positions{
            std::vector<float>(initial_particle_count),
            std::vector<float>(initial_particle_count),
            std::vector<float>(initial_particle_count),
        };
        std::uint32_t particle{};
        for (std::uint32_t z = 0u; z < initial_cells[2]; ++z)
            for (std::uint32_t y = 0u; y < initial_cells[1]; ++y)
                for (std::uint32_t x = 0u; x < initial_cells[0]; ++x)
                    for (std::uint32_t local = 0u; local < particles_per_cell; ++local) {
                        positions[0][particle] = (2.0F + static_cast<float>(x) + ((local & 1u) == 0u ? 0.25F : 0.75F)) * cell_size;
                        positions[1][particle] = (2.0F + static_cast<float>(y) + ((local & 2u) == 0u ? 0.25F : 0.75F)) * cell_size;
                        positions[2][particle] = (2.0F + static_cast<float>(z) + ((local & 4u) == 0u ? 0.25F : 0.75F)) * cell_size;
                        ++particle;
                    }
        ::cuda::copy_bytes(stream, positions[0], ::cuda::std::span{flip_state.positions.x.data(), initial_particle_count});
        ::cuda::copy_bytes(stream, positions[1], ::cuda::std::span{flip_state.positions.y.data(), initial_particle_count});
        ::cuda::copy_bytes(stream, positions[2], ::cuda::std::span{flip_state.positions.z.data(), initial_particle_count});
        ::cuda::copy_bytes(stream, positions[0], ::cuda::std::span{apic_state.positions.x.data(), initial_particle_count});
        ::cuda::copy_bytes(stream, positions[1], ::cuda::std::span{apic_state.positions.y.data(), initial_particle_count});
        ::cuda::copy_bytes(stream, positions[2], ::cuda::std::span{apic_state.positions.z.data(), initial_particle_count});
        simulation::clear(model.grid.stream, flip_state.velocities);
        simulation::clear(model.grid.stream, apic_state.velocities);
        simulation::clear(model.grid.stream, apic_state.transfer.affine);
        flip_state.particle_count = initial_particle_count;
        apic_state.particle_count = initial_particle_count;
        flip_solver.copy_state(model, flip_state, flip_next_state);
        apic_solver.copy_state(model, apic_state, apic_next_state);
        stream.sync();
        step_index                = 0u;
        flip_substep_count        = 0u;
        apic_substep_count        = 0u;
        physical_time             = 0.0;
        flip_time                 = 0.0;
        apic_time                 = 0.0;
        flip_particle_diagnostics = {};
        apic_particle_diagnostics = {};
    }

    void Simulation::step(const double seconds) {
        const double target_time = physical_time + seconds;
        advance(flip_solver, flip_state, flip_next_state, flip_cache, flip_workspace, target_time, flip_time, flip_substep_count);
        advance(apic_solver, apic_state, apic_next_state, apic_cache, apic_workspace, target_time, apic_time, apic_substep_count);
        physical_time             = target_time;
        flip_particle_diagnostics = flip_solver.particle_diagnostics(model, static_cast<float>(physical_time), flip_state, flip_workspace);
        apic_particle_diagnostics = apic_solver.particle_diagnostics(model, static_cast<float>(physical_time), apic_state, apic_workspace);
        ++step_index;
    }

    fluids::liquid::solvers::pic::ModelConfiguration Simulation::create_configuration() {
        return {
            .grid =
                {
                    .resolution = resolution,
                    .cell_size  = cell_size,
                    .origin     = {},
                    .velocity   = {},
                },
            .maximum_particle_count = maximum_particle_count,
            .particle_radius        = particle_radius,
            .no_slip                = false,
        };
    }

    fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::FlipTransfer>::Configuration Simulation::create_flip_configuration() {
        return {
            .transfer = {.ratio = 0.95F},
            .grid_step =
                {
                    .acceleration           = {.x = 0.0F, .y = -9.81F, .z = 0.0F},
                    .level_set_radius_cells = 0.75F,
                    .extrapolation_layers   = 4u,
                },
            .particle_step     = {.minimum_per_cell = 3u, .target_per_cell = 8u, .maximum_per_cell = 12u},
            .projection        = {.density = 1000.0F, .maximum_iterations = 160u, .tolerance = 1.0e-5F},
            .maximum_time_step = 1.0F / 240.0F,
            .cfl_number        = 1.0F,
        };
    }

    fluids::liquid::solvers::pic::Solver<fluids::liquid::solvers::pic::ApicTransfer>::Configuration Simulation::create_apic_configuration() {
        return {
            .transfer = {.affine_ratio = 1.0F},
            .grid_step =
                {
                    .acceleration           = {.x = 0.0F, .y = -9.81F, .z = 0.0F},
                    .level_set_radius_cells = 0.75F,
                    .extrapolation_layers   = 4u,
                },
            .particle_step     = {.minimum_per_cell = 3u, .target_per_cell = 8u, .maximum_per_cell = 12u},
            .projection        = {.density = 1000.0F, .maximum_iterations = 160u, .tolerance = 1.0e-5F},
            .maximum_time_step = 1.0F / 240.0F,
            .cfl_number        = 1.0F,
        };
    }
} // namespace physica::examples::flip_apic_dam_break
