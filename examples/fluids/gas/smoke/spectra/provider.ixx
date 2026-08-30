module;

#include <physica/cuda.h>
#include <spectra/sdk/cuda_types.h>

export module physica.example.fluids.gas.smoke.spectra;

import std;
import physica.example.fluids.gas.smoke;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::smoke {
    struct Settings final {};

    struct Provider final {
    public:
        [[no_unique_address]] Settings settings;

        static constexpr auto description = spectra::sdk::describe("physica.example.fluids.gas.smoke", spectra::sdk::volume<"smoke">(spectra::sdk::volume_field<"density", float>("Density", {}, {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"temperature", float>("Temperature", {}, {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"velocity", spectra::sdk::MacFloat3>("Velocity", "m/s", {.sampling = spectra::sdk::VolumeFieldSampling::Cell, .vector_space = spectra::sdk::VolumeVectorSpace::World})), spectra::sdk::metric<"step", std::uint64_t>("Step", {}, "Simulation"), spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true));

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
        setup.volume<"smoke">({Simulation::resolution[0], Simulation::resolution[1], Simulation::resolution[2]});
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
        Simulation& state                       = simulation_state();
        spectra::sdk::cuda::Frame frame         = output.begin(state.stream.get());
        const spectra::sdk::cuda::Volume volume = frame.volume<"smoke">();
        const std::span<float> density          = volume.field<"density", float>();
        const std::span<float> temperature      = volume.field<"temperature", float>();
        ::cuda::copy_bytes(state.stream, state.current_state.density.values, ::cuda::std::span{density.data(), density.size()});
        ::cuda::copy_bytes(state.stream, state.current_state.temperature.values, ::cuda::std::span{temperature.data(), temperature.size()});
        const spectra::sdk::cuda::MacField velocity = volume.field<"velocity">();
        ::cuda::copy_bytes(state.stream, state.current_state.velocity.x, ::cuda::std::span{velocity.x.data(), velocity.x.size()});
        ::cuda::copy_bytes(state.stream, state.current_state.velocity.y, ::cuda::std::span{velocity.y.data(), velocity.y.size()});
        ::cuda::copy_bytes(state.stream, state.current_state.velocity.z, ::cuda::std::span{velocity.z.data(), velocity.z.size()});
        frame.metric<"step">().upload(state.step_index);
        frame.metric<"time">().upload(state.physical_time);
        frame.commit();
    }
} // namespace physica::examples::smoke
