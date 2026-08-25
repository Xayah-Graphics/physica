module;

#include "kernels.h"
#include <physica/cuda.h>
#include <spectra/sdk/cuda_types.h>

export module physica.example.fluids.liquid.pbf_dam_break.spectra;

import std;
import physica.example.fluids.liquid.pbf_dam_break;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::pbf_dam_break {
    struct Settings final {};

    struct Provider final {
    private:
        struct SimulationDeleter final {
            void operator()(Simulation* simulation) const noexcept;
        };

    public:
        [[no_unique_address]] Settings settings;

        static constexpr auto description = spectra::sdk::describe("physica.example.fluids.liquid.pbf_dam_break", spectra::sdk::particles<"particles">(spectra::sdk::field<"velocity", spectra::sdk::Float3>("Velocity", "m/s"), spectra::sdk::field<"vorticity", float>("Vorticity", "1/s")), spectra::sdk::metric<"step", std::uint64_t>("Physical Step", {}, "Simulation"), spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true));

        Provider(Settings settings, const std::filesystem::path& assets);
        ~Provider() noexcept;

        Provider(const Provider&)            = delete;
        Provider& operator=(const Provider&) = delete;

        void setup(spectra::sdk::cuda::Setup& setup);
        void reset(std::uint64_t seed);
        void step(double seconds);
        void publish(spectra::sdk::cuda::Output& output);

    private:
        std::unique_ptr<Simulation, SimulationDeleter> simulation;
    };

    void Provider::SimulationDeleter::operator()(Simulation* const source) const noexcept {
        delete source;
    }

    Provider::Provider(const Settings source, const std::filesystem::path&) : settings(source), simulation{new Simulation{}} {}

    Provider::~Provider() noexcept {}

    void Provider::setup(spectra::sdk::cuda::Setup& setup) {
        static_cast<void>(setup.particles<"particles">(Simulation::particle_count, Simulation::particle_radius));
    }

    void Provider::reset(const std::uint64_t) {
        simulation->reset();
    }

    void Provider::step(const double) {
        simulation->step();
    }

    void Provider::publish(spectra::sdk::cuda::Output& output) {
        spectra::sdk::cuda::Frame frame                  = output.begin(simulation->stream.get());
        const spectra::sdk::cuda::Particles particles    = frame.particles<"particles">(Simulation::particle_count);
        const std::span<spectra::sdk::Float3> positions  = particles.positions;
        const std::span<spectra::sdk::Float3> velocities = particles.field<"velocity", spectra::sdk::Float3>();
        spectra_cuda::write_vectors(simulation->stream, Simulation::particle_count, simulation->current_state.particles.positions.x.data(), simulation->current_state.particles.positions.y.data(), simulation->current_state.particles.positions.z.data(), simulation->current_state.particles.velocities.x.data(), simulation->current_state.particles.velocities.y.data(), simulation->current_state.particles.velocities.z.data(), positions.data(), velocities.data());
        const std::span<float> vorticity = particles.field<"vorticity", float>();
        ::cuda::copy_bytes(simulation->stream, simulation->step_cache.vorticity_magnitudes.values, ::cuda::std::span{vorticity.data(), vorticity.size()});
        frame.metric<"step">().upload(simulation->step_index);
        frame.metric<"time">().upload(simulation->physical_time);
        frame.commit();
    }
} // namespace physica::examples::pbf_dam_break
