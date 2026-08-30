module;

#include "kernels.h"
#include <physica/cuda.h>
#include <spectra/sdk/cuda_types.h>

export module physica.example.fluids.liquid.flip_apic_dam_break.spectra;

import std;
import physica.example.fluids.liquid.flip_apic_dam_break;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::flip_apic_dam_break {
    struct Settings final {};

    struct Provider final {
    private:
        struct SimulationDeleter final {
            void operator()(Simulation* simulation) const noexcept;
        };

    public:
        [[no_unique_address]] Settings settings;

        static constexpr auto description = spectra::sdk::describe("physica.example.fluids.liquid.flip_apic_dam_break", spectra::sdk::particles<"flip-particles">(spectra::sdk::field<"velocity", spectra::sdk::Float3>("Velocity", "m/s"), spectra::sdk::field<"speed", float>("Speed", "m/s")), spectra::sdk::particles<"apic-particles">(spectra::sdk::field<"velocity", spectra::sdk::Float3>("Velocity", "m/s"), spectra::sdk::field<"speed", float>("Speed", "m/s"), spectra::sdk::field<"affine-magnitude", float>("Affine Magnitude", "1/s")),
            spectra::sdk::volume<"flip-grid">(spectra::sdk::volume_field<"velocity", spectra::sdk::MacFloat3>("Velocity", "m/s", {.vector_space = spectra::sdk::VolumeVectorSpace::World}), spectra::sdk::volume_field<"pressure", float>("Pressure", "Pa", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"divergence", float>("Divergence", "1/s", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"level-set", float>("Particle Level Set", "m", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"cell-type", std::uint32_t>("Cell Type", {}, {.sampling = spectra::sdk::VolumeFieldSampling::Cell})),
            spectra::sdk::volume<"apic-grid">(spectra::sdk::volume_field<"velocity", spectra::sdk::MacFloat3>("Velocity", "m/s", {.vector_space = spectra::sdk::VolumeVectorSpace::World}), spectra::sdk::volume_field<"pressure", float>("Pressure", "Pa", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"divergence", float>("Divergence", "1/s", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"level-set", float>("Particle Level Set", "m", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"cell-type", std::uint32_t>("Cell Type", {}, {.sampling = spectra::sdk::VolumeFieldSampling::Cell})), spectra::sdk::metric<"step", std::uint64_t>("Frame", {}, "Simulation"), spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true), spectra::sdk::metric<"flip-particle-count", std::uint64_t>("FLIP Particles", {}, "FLIP"), spectra::sdk::metric<"apic-particle-count", std::uint64_t>("APIC Particles", {}, "APIC"),
            spectra::sdk::metric<"flip-substeps", std::uint64_t>("FLIP Substeps", {}, "FLIP"), spectra::sdk::metric<"apic-substeps", std::uint64_t>("APIC Substeps", {}, "APIC"), spectra::sdk::metric<"flip-pressure-iterations", std::uint64_t>("FLIP Pressure Iterations", {}, "FLIP"), spectra::sdk::metric<"apic-pressure-iterations", std::uint64_t>("APIC Pressure Iterations", {}, "APIC"), spectra::sdk::metric<"flip-pressure-residual", float>("FLIP Pressure Residual", {}, "FLIP", true), spectra::sdk::metric<"apic-pressure-residual", float>("APIC Pressure Residual", {}, "APIC", true), spectra::sdk::metric<"flip-divergence-before", float>("FLIP Divergence Before", "1/s", "FLIP", true), spectra::sdk::metric<"flip-divergence-after", float>("FLIP Divergence After", "1/s", "FLIP", true), spectra::sdk::metric<"apic-divergence-before", float>("APIC Divergence Before", "1/s", "APIC", true), spectra::sdk::metric<"apic-divergence-after", float>("APIC Divergence After", "1/s", "APIC", true),
            spectra::sdk::metric<"flip-kinetic-energy", float>("FLIP Kinetic Energy", "J", "FLIP", true), spectra::sdk::metric<"apic-kinetic-energy", float>("APIC Kinetic Energy", "J", "APIC", true), spectra::sdk::metric<"flip-linear-momentum", spectra::sdk::Float3>("FLIP Linear Momentum", "kg m/s", "FLIP"), spectra::sdk::metric<"apic-linear-momentum", spectra::sdk::Float3>("APIC Linear Momentum", "kg m/s", "APIC"), spectra::sdk::metric<"flip-angular-momentum", spectra::sdk::Float3>("FLIP Angular Momentum", "kg m2/s", "FLIP"), spectra::sdk::metric<"apic-angular-momentum", spectra::sdk::Float3>("APIC Angular Momentum", "kg m2/s", "APIC"));

        Provider(Settings settings, const std::filesystem::path& assets);
        ~Provider() noexcept;

        Provider(const Provider&)            = delete;
        Provider& operator=(const Provider&) = delete;

        void setup(spectra::sdk::cuda::Setup& setup);
        void reset(std::uint64_t seed);
        void step(double seconds);
        void publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame);

    private:
        std::unique_ptr<Simulation, SimulationDeleter> simulation;
    };

    void Provider::SimulationDeleter::operator()(Simulation* const source) const noexcept {
        delete source;
    }

    Provider::Provider(const Settings source, const std::filesystem::path&) : settings(source), simulation{new Simulation{}} {}

    Provider::~Provider() noexcept {}

    void Provider::setup(spectra::sdk::cuda::Setup& setup) {
        static_cast<void>(setup.particles<"flip-particles">(Simulation::maximum_particle_count, Simulation::particle_radius));
        static_cast<void>(setup.particles<"apic-particles">(Simulation::maximum_particle_count, Simulation::particle_radius));
        setup.volume<"flip-grid">({Simulation::resolution[0], Simulation::resolution[1], Simulation::resolution[2]});
        setup.volume<"apic-grid">({Simulation::resolution[0], Simulation::resolution[1], Simulation::resolution[2]});
    }

    void Provider::reset(const std::uint64_t) {
        simulation->reset();
    }

    void Provider::step(const double seconds) {
        simulation->step(seconds);
    }

    void Provider::publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame) {
        spectra::sdk::cuda::Frame frame                    = output.begin(simulation->stream.get());
        const spectra::sdk::cuda::Particles flip_particles = frame.particles<"flip-particles">(simulation->flip_state.particle_count);
        spectra_cuda::write_flip_particles(simulation->stream, simulation->flip_state.particle_count, 0.0F, simulation->flip_state.positions.x.data(), simulation->flip_state.positions.y.data(), simulation->flip_state.positions.z.data(), simulation->flip_state.velocities.x.data(), simulation->flip_state.velocities.y.data(), simulation->flip_state.velocities.z.data(), flip_particles.positions.data(), flip_particles.field<"velocity", spectra::sdk::Float3>().data(), flip_particles.field<"speed", float>().data());
        const spectra::sdk::cuda::Particles apic_particles = frame.particles<"apic-particles">(simulation->apic_state.particle_count);
        const auto& affine                                 = simulation->apic_state.transfer.affine;
        spectra_cuda::write_apic_particles(simulation->stream, simulation->apic_state.particle_count, 0.95F, simulation->apic_state.positions.x.data(), simulation->apic_state.positions.y.data(), simulation->apic_state.positions.z.data(), simulation->apic_state.velocities.x.data(), simulation->apic_state.velocities.y.data(), simulation->apic_state.velocities.z.data(), affine.c00.data(), affine.c01.data(), affine.c02.data(), affine.c10.data(), affine.c11.data(), affine.c12.data(), affine.c20.data(), affine.c21.data(), affine.c22.data(), apic_particles.positions.data(), apic_particles.field<"velocity", spectra::sdk::Float3>().data(), apic_particles.field<"speed", float>().data(), apic_particles.field<"affine-magnitude", float>().data());

        const spectra::sdk::cuda::Volume flip_grid       = frame.volume<"flip-grid">();
        const spectra::sdk::cuda::MacField flip_velocity = flip_grid.field<"velocity">();
        ::cuda::copy_bytes(simulation->stream, simulation->flip_cache.grid.velocity.x, ::cuda::std::span{flip_velocity.x.data(), flip_velocity.x.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->flip_cache.grid.velocity.y, ::cuda::std::span{flip_velocity.y.data(), flip_velocity.y.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->flip_cache.grid.velocity.z, ::cuda::std::span{flip_velocity.z.data(), flip_velocity.z.size()});
        const std::span<float> flip_pressure          = flip_grid.field<"pressure", float>();
        const std::span<float> flip_divergence        = flip_grid.field<"divergence", float>();
        const std::span<float> flip_level_set         = flip_grid.field<"level-set", float>();
        const std::span<std::uint32_t> flip_cell_type = flip_grid.field<"cell-type", std::uint32_t>();
        ::cuda::copy_bytes(simulation->stream, simulation->flip_cache.grid.pressure.values, ::cuda::std::span{flip_pressure.data(), flip_pressure.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->flip_cache.grid.divergence.values, ::cuda::std::span{flip_divergence.data(), flip_divergence.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->flip_cache.grid.level_set.values, ::cuda::std::span{flip_level_set.data(), flip_level_set.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->flip_cache.grid.cell_types.values, ::cuda::std::span{flip_cell_type.data(), flip_cell_type.size()});

        const spectra::sdk::cuda::Volume apic_grid       = frame.volume<"apic-grid">();
        const spectra::sdk::cuda::MacField apic_velocity = apic_grid.field<"velocity">();
        ::cuda::copy_bytes(simulation->stream, simulation->apic_cache.grid.velocity.x, ::cuda::std::span{apic_velocity.x.data(), apic_velocity.x.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->apic_cache.grid.velocity.y, ::cuda::std::span{apic_velocity.y.data(), apic_velocity.y.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->apic_cache.grid.velocity.z, ::cuda::std::span{apic_velocity.z.data(), apic_velocity.z.size()});
        const std::span<float> apic_pressure          = apic_grid.field<"pressure", float>();
        const std::span<float> apic_divergence        = apic_grid.field<"divergence", float>();
        const std::span<float> apic_level_set         = apic_grid.field<"level-set", float>();
        const std::span<std::uint32_t> apic_cell_type = apic_grid.field<"cell-type", std::uint32_t>();
        ::cuda::copy_bytes(simulation->stream, simulation->apic_cache.grid.pressure.values, ::cuda::std::span{apic_pressure.data(), apic_pressure.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->apic_cache.grid.divergence.values, ::cuda::std::span{apic_divergence.data(), apic_divergence.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->apic_cache.grid.level_set.values, ::cuda::std::span{apic_level_set.data(), apic_level_set.size()});
        ::cuda::copy_bytes(simulation->stream, simulation->apic_cache.grid.cell_types.values, ::cuda::std::span{apic_cell_type.data(), apic_cell_type.size()});

        frame.metric<"step">().upload(simulation->step_index);
        frame.metric<"time">().upload(simulation->physical_time);
        frame.metric<"flip-particle-count">().upload(static_cast<std::uint64_t>(simulation->flip_state.particle_count));
        frame.metric<"apic-particle-count">().upload(static_cast<std::uint64_t>(simulation->apic_state.particle_count));
        frame.metric<"flip-substeps">().upload(simulation->flip_substep_count);
        frame.metric<"apic-substeps">().upload(simulation->apic_substep_count);
        frame.metric<"flip-pressure-iterations">().upload(static_cast<std::uint64_t>(simulation->flip_cache.diagnostics.projection.iterations));
        frame.metric<"apic-pressure-iterations">().upload(static_cast<std::uint64_t>(simulation->apic_cache.diagnostics.projection.iterations));
        frame.metric<"flip-pressure-residual">().upload(simulation->flip_cache.diagnostics.projection.relative_residual);
        frame.metric<"apic-pressure-residual">().upload(simulation->apic_cache.diagnostics.projection.relative_residual);
        frame.metric<"flip-divergence-before">().upload(simulation->flip_cache.diagnostics.divergence_before_projection.l2);
        frame.metric<"flip-divergence-after">().upload(simulation->flip_cache.diagnostics.divergence_after_projection.l2);
        frame.metric<"apic-divergence-before">().upload(simulation->apic_cache.diagnostics.divergence_before_projection.l2);
        frame.metric<"apic-divergence-after">().upload(simulation->apic_cache.diagnostics.divergence_after_projection.l2);
        frame.metric<"flip-kinetic-energy">().upload(simulation->flip_particle_diagnostics.kinetic_energy);
        frame.metric<"apic-kinetic-energy">().upload(simulation->apic_particle_diagnostics.kinetic_energy);
        frame.metric<"flip-linear-momentum">().upload(spectra::sdk::Float3{simulation->flip_particle_diagnostics.linear_momentum.x, simulation->flip_particle_diagnostics.linear_momentum.y, simulation->flip_particle_diagnostics.linear_momentum.z});
        frame.metric<"apic-linear-momentum">().upload(spectra::sdk::Float3{simulation->apic_particle_diagnostics.linear_momentum.x, simulation->apic_particle_diagnostics.linear_momentum.y, simulation->apic_particle_diagnostics.linear_momentum.z});
        frame.metric<"flip-angular-momentum">().upload(spectra::sdk::Float3{simulation->flip_particle_diagnostics.angular_momentum.x, simulation->flip_particle_diagnostics.angular_momentum.y, simulation->flip_particle_diagnostics.angular_momentum.z});
        frame.metric<"apic-angular-momentum">().upload(spectra::sdk::Float3{simulation->apic_particle_diagnostics.angular_momentum.x, simulation->apic_particle_diagnostics.angular_momentum.y, simulation->apic_particle_diagnostics.angular_momentum.z});
        frame.commit();
    }
} // namespace physica::examples::flip_apic_dam_break
