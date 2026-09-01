module;

#include <spectra/sdk/cuda_types.h>

export module physica.example.fluids.gas.smoke.module;

import std;
import physica.example.fluids.gas.smoke;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::smoke {
    struct Settings final {};

    struct Module final {
        [[no_unique_address]] Settings settings;

        static constexpr auto description = spectra::sdk::describe("physica.example.fluids.gas.smoke", spectra::sdk::volume<"smoke">(spectra::sdk::volume_field<"density", float>("Density", {}, {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"temperature", float>("Temperature", {}, {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"velocity", spectra::sdk::MacFloat3>("Velocity", "m/s", {.sampling = spectra::sdk::VolumeFieldSampling::Cell, .vector_space = spectra::sdk::VolumeVectorSpace::World})), spectra::sdk::metric<"step", std::uint64_t>("Step", {}, "Simulation"), spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true));

        Module(Settings settings, const std::filesystem::path& assets, const spectra::sdk::SceneInputs& inputs);
        ~Module();

        Module(const Module&)            = delete;
        Module& operator=(const Module&) = delete;

        void setup(spectra::sdk::cuda::Setup& setup);
        void reset(std::uint64_t seed);
        void step(double seconds);
        void publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame);

    private:
        std::unique_ptr<Simulation> simulation;
    };

} // namespace physica::examples::smoke
