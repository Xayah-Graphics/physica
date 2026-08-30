module;

#include <physica/cuda.h>

export module physica.fluids.liquid.solvers.pic;

import std;
export import physica.fluids.liquid.solvers.pic.model;
export import physica.fluids.liquid.solvers.pic.grid_step;
export import physica.fluids.liquid.solvers.pic.particle_step;
export import physica.fluids.liquid.solvers.pic.projection;
export import physica.fluids.liquid.solvers.pic.transfer;

export namespace physica::fluids::liquid::solvers::pic {
    struct StepDiagnostics final {
        GridStep::Diagnostics divergence_before_projection;
        GridStep::Diagnostics divergence_after_projection;
        Projection::Diagnostics projection;
        std::uint32_t particle_count_before{};
        std::uint32_t particle_count_after{};
        float time_step{};
    };

    template <class Algorithm>
    concept TransferAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Model& model, typename Algorithm::State& state, const typename Algorithm::State& constant_state, typename Algorithm::Workspace& workspace, const simulation::VectorField<float>& positions, const simulation::VectorField<float>& velocities, simulation::VectorField<float>& output_velocities, simulation::VectorField<float>& grid_velocity, simulation::VectorField<float>& grid_mass, const simulation::ScalarField<std::uint32_t>& particle_flags, const ParticleStep::Maintenance& maintenance) {
        { algorithm.allocate_state(model) } -> std::same_as<typename Algorithm::State>;
        { algorithm.allocate_workspace(model) } -> std::same_as<typename Algorithm::Workspace>;
        algorithm.clear_state(model, state);
        algorithm.copy_state(model, constant_state, state);
        algorithm.particle_to_grid(model, float{}, std::uint32_t{}, positions, velocities, constant_state, grid_velocity, grid_mass);
        algorithm.grid_to_particle(model, float{}, std::uint32_t{}, positions, velocities, constant_state, grid_velocity, grid_velocity, output_velocities, state);
        algorithm.compact_and_seed(model, float{}, std::uint32_t{}, maintenance, positions, grid_velocity, constant_state, state, particle_flags, particle_flags, workspace);
        algorithm.commit_compaction(state, workspace);
    };

    template <TransferAlgorithm Transfer>
    struct Solver final {
        struct Configuration final {
            typename Transfer::Configuration transfer{};
            GridStep::Configuration grid_step{};
            ParticleStep::Configuration particle_step{};
            Projection::Configuration projection{};
            float maximum_time_step{1.0F / 120.0F};
            float cfl_number{1.0F};
        };

        struct State final {
            simulation::VectorField<float> positions;
            simulation::VectorField<float> velocities;
            typename Transfer::State transfer;
            std::uint32_t particle_count{};
        };

        struct StepCache final {
            GridStep::State grid;
            Projection::Workspace projection;
            StepDiagnostics diagnostics;
        };

        struct Workspace final {
            GridStep::Workspace grid_step;
            ParticleStep::Workspace particle_step;
            typename Transfer::Workspace transfer;
        };

        explicit Solver(Configuration configuration) : transfer(std::move(configuration.transfer)), grid_step(std::move(configuration.grid_step)), particle_step(std::move(configuration.particle_step)), projection(std::move(configuration.projection)), maximum_time_step(configuration.maximum_time_step), cfl_number(configuration.cfl_number) {}

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State allocate_state(const Model& model) const {
            State state{
                .positions      = simulation::VectorField<float>(model.grid.stream, model.maximum_particle_count),
                .velocities     = simulation::VectorField<float>(model.grid.stream, model.maximum_particle_count),
                .transfer       = transfer.allocate_state(model),
                .particle_count = 0u,
            };
            simulation::clear(model.grid.stream, state.positions);
            simulation::clear(model.grid.stream, state.velocities);
            transfer.clear_state(model, state.transfer);
            return state;
        }

        [[nodiscard]] StepCache allocate_step_cache(const Model& model) const {
            return {.grid = grid_step.allocate_state(model), .projection = projection.allocate_workspace(model)};
        }

        [[nodiscard]] Workspace allocate_workspace(const Model& model) const {
            return {.grid_step = grid_step.allocate_workspace(model), .particle_step = particle_step.allocate_workspace(model), .transfer = transfer.allocate_workspace(model)};
        }

        void copy_state(const Model& model, const State& source, State& destination) const {
            simulation::copy(model.grid.stream, source.positions, destination.positions);
            simulation::copy(model.grid.stream, source.velocities, destination.velocities);
            transfer.copy_state(model, source.transfer, destination.transfer);
            destination.particle_count = source.particle_count;
        }

        [[nodiscard]] float stable_time_step(const Model& model, const State& state, Workspace& workspace) const {
            const float particle_speed         = particle_step.maximum_speed(model, state.particle_count, state.velocities, workspace.particle_step);
            const Vector3<float> grid_velocity = model.grid.configuration.velocity;
            const float boundary_speed         = std::sqrt(grid_velocity.x * grid_velocity.x + grid_velocity.y * grid_velocity.y + grid_velocity.z * grid_velocity.z);
            const float maximum_speed          = std::max(particle_speed, boundary_speed);
            if (maximum_speed == 0.0F) return maximum_time_step;
            return std::min(maximum_time_step, cfl_number * model.grid.configuration.cell_size / maximum_speed);
        }

        [[nodiscard]] ParticleStep::Diagnostics particle_diagnostics(const Model& model, const float time, const State& state, Workspace& workspace) const {
            const float cell_size     = model.grid.configuration.cell_size;
            const float particle_mass = projection.configuration.density * cell_size * cell_size * cell_size / static_cast<float>(particle_step.configuration.target_per_cell);
            return particle_step.diagnostics(model, time, state.particle_count, particle_mass, state.positions, state.velocities, workspace.particle_step);
        }

        void forward(const Model& model, const float time, const float time_step, const std::uint64_t seed, const State& state, State& next_state, StepCache& cache, Workspace& workspace) const {
            grid_step.begin_transfer(model, cache.grid, workspace.grid_step);
            transfer.particle_to_grid(model, time, state.particle_count, state.positions, state.velocities, state.transfer, cache.grid.velocity_before_projection, workspace.grid_step.face_mass);
            grid_step.classify_and_normalize(model, time, state.particle_count, state.positions, cache.grid, workspace.grid_step);
            grid_step.extrapolate_before_projection(model, time, cache.grid, workspace.grid_step);
            model.grid.copy(cache.grid.velocity_before_projection, workspace.grid_step.projection_input_velocity);
            grid_step.apply_force_and_constrain(model, time, time_step, cache.grid.cell_types, workspace.grid_step.projection_input_velocity);
            cache.diagnostics.divergence_before_projection = grid_step.divergence(model, time, cache.grid.cell_types, workspace.grid_step.projection_input_velocity, cache.grid.divergence, workspace.grid_step);
            cache.diagnostics.projection                   = projection.forward(model, time, time_step, cache.grid.cell_types, cache.grid.level_set, workspace.grid_step.projection_input_velocity, cache.grid.velocity, cache.grid.pressure, cache.projection);
            grid_step.apply_force_and_constrain(model, time, 0.0F, cache.grid.cell_types, cache.grid.velocity);
            grid_step.prepare_after_projection(model, time, cache.grid, workspace.grid_step);
            cache.diagnostics.divergence_after_projection = grid_step.divergence(model, time, cache.grid.cell_types, cache.grid.velocity, cache.grid.divergence, workspace.grid_step);
            simulation::copy(model.grid.stream, state.positions, next_state.positions);
            transfer.grid_to_particle(model, time, state.particle_count, state.positions, state.velocities, state.transfer, cache.grid.velocity_before_projection, cache.grid.velocity, next_state.velocities, next_state.transfer);
            particle_step.advect(model, time, time_step, state.particle_count, cache.grid.velocity, next_state.positions, next_state.velocities);
            const ParticleStep::Maintenance maintenance = particle_step.plan_maintenance(model, time, state.particle_count, next_state.positions, cache.grid.cell_types, cache.grid.level_set, workspace.particle_step);
            particle_step.compact_and_seed(model, time, seed, state.particle_count, maintenance, next_state.positions, next_state.velocities, cache.grid.velocity, workspace.particle_step);
            transfer.compact_and_seed(model, time, state.particle_count, maintenance, workspace.particle_step.compacted_positions, cache.grid.velocity, next_state.transfer, next_state.transfer, workspace.particle_step.keep_flags, workspace.particle_step.destinations, workspace.transfer);
            std::swap(next_state.positions, workspace.particle_step.compacted_positions);
            std::swap(next_state.velocities, workspace.particle_step.compacted_velocities);
            transfer.commit_compaction(next_state.transfer, workspace.transfer);
            next_state.particle_count               = maintenance.particle_count();
            cache.diagnostics.particle_count_before = state.particle_count;
            cache.diagnostics.particle_count_after  = next_state.particle_count;
            cache.diagnostics.time_step             = time_step;
        }

    private:
        Transfer transfer;
        GridStep grid_step;
        ParticleStep particle_step;
        Projection projection;
        const float maximum_time_step;
        const float cfl_number;
    };
} // namespace physica::fluids::liquid::solvers::pic
