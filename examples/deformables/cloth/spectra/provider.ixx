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
        static_cast<void>(setup.mesh<"surface">(Simulation::rows * Simulation::columns, 0u));
    }

    void Provider::reset(const std::uint64_t) {
        simulation_state().reset();
    }

    void Provider::step(const double) {
        simulation_state().step();
    }

    void Provider::publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame) {
        Simulation& state                      = simulation_state();
        spectra::sdk::cuda::Frame frame        = output.begin(state.stream.get());
        const spectra::sdk::cuda::Mesh surface = frame.mesh<"surface">();
        spectra_cuda::write_surface(state.stream, Simulation::rows, Simulation::columns, state.current_state.positions.x.data(), state.current_state.positions.y.data(), state.current_state.positions.z.data(), surface.positions.data(), surface.normals.data());
        frame.metric<"step">().upload(state.step_index);
        frame.metric<"time">().upload(state.physical_time);
        frame.commit();
    }
} // namespace physica::examples::cloth
