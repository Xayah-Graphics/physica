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
        void publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame);

    private:
        static void destroy_simulation(void* simulation) noexcept;
        [[nodiscard]] Simulation& simulation_state() noexcept;
        std::unique_ptr<void, decltype(&destroy_simulation)> simulation{nullptr, destroy_simulation};
    };

    void Provider::destroy_simulation(void* const source) noexcept {
        delete static_cast<Simulation*>(source);
    }

    Simulation& Provider::simulation_state() noexcept {
        return *static_cast<Simulation*>(simulation.get());
    }

    Provider::Provider(const Settings source, const std::filesystem::path&) : settings(source), simulation{std::make_unique<Simulation>().release(), destroy_simulation} {}

    Provider::~Provider() noexcept {}

    void Provider::setup(spectra::sdk::cuda::Setup& setup) {
        static_cast<void>(setup.particles<"particles">(Simulation::particle_count, Simulation::particle_radius));
    }

    void Provider::reset(const std::uint64_t) {
        Simulation& state = simulation_state();
        state.reset();
    }

    void Provider::step(const double) {
        Simulation& state = simulation_state();
        state.step();
    }

    void Provider::publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame) {
        Simulation& state                                = simulation_state();
        spectra::sdk::cuda::Frame frame                  = output.begin(state.stream.get());
        const spectra::sdk::cuda::Particles particles    = frame.particles<"particles">(Simulation::particle_count);
        const std::span<spectra::sdk::Float3> positions  = particles.positions;
        const std::span<spectra::sdk::Float3> velocities = particles.field<"velocity", spectra::sdk::Float3>();
        spectra_cuda::write_vectors(state.stream, Simulation::particle_count, state.current_state.positions.x.data(), state.current_state.positions.y.data(), state.current_state.positions.z.data(), state.current_state.velocities.x.data(), state.current_state.velocities.y.data(), state.current_state.velocities.z.data(), positions.data(), velocities.data());
        const std::span<float> vorticity = particles.field<"vorticity", float>();
        ::cuda::copy_bytes(state.stream, state.step_cache.vorticity_magnitudes.values, ::cuda::std::span{vorticity.data(), vorticity.size()});
        frame.metric<"step">().upload(state.step_index);
        frame.metric<"time">().upload(state.physical_time);
        frame.commit();
    }
} // namespace physica::examples::pbf_dam_break
