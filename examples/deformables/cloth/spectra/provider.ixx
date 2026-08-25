module;

#include "kernels.h"
#include <physica/cuda.h>
#include <spectra/sdk/cuda_types.h>

export module physica.example.deformables.cloth.spectra;

import std;
import physica.example.deformables.cloth;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::cloth {
    struct Settings final {};

    struct Provider final {
    private:
        struct SimulationDeleter final {
            void operator()(Simulation* simulation) const noexcept;
        };

    public:
        [[no_unique_address]] Settings settings;

        static constexpr auto description = spectra::sdk::describe("physica.example.deformables.cloth", spectra::sdk::mesh<"surface">({.attributes = spectra::sdk::MeshAttribute::Normal}), spectra::sdk::metric<"step", std::uint64_t>("Step", {}, "Simulation"), spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true));

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
        static_cast<void>(setup.mesh<"surface">(Simulation::rows * Simulation::columns, 0u));
    }

    void Provider::reset(const std::uint64_t) {
        simulation->reset();
    }

    void Provider::step(const double) {
        simulation->step();
    }

    void Provider::publish(spectra::sdk::cuda::Output& output) {
        spectra::sdk::cuda::Frame frame        = output.begin(simulation->stream.get());
        const spectra::sdk::cuda::Mesh surface = frame.mesh<"surface">();
        spectra_cuda::write_surface(simulation->stream, Simulation::rows, Simulation::columns, simulation->current_state.positions.x.data(), simulation->current_state.positions.y.data(), simulation->current_state.positions.z.data(), surface.positions.data(), surface.normals.data());
        frame.metric<"step">().upload(simulation->step_index);
        frame.metric<"time">().upload(simulation->physical_time);
        frame.commit();
    }
} // namespace physica::examples::cloth
