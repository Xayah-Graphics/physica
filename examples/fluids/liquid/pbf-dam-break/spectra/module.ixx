module;

#include <spectra/sdk/cuda_types.h>

export module physica.example.fluids.liquid.pbf_dam_break.module;

import std;
import physica.example.fluids.liquid.pbf_dam_break;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::pbf_dam_break {
    struct Settings final {};

    struct Module final {
        [[no_unique_address]] Settings settings;

        static constexpr auto description = spectra::sdk::describe("physica.example.fluids.liquid.pbf_dam_break", spectra::sdk::particles<"particles">(spectra::sdk::field<"velocity", spectra::sdk::Float3>("Velocity", "m/s"), spectra::sdk::field<"vorticity", float>("Vorticity", "1/s")), spectra::sdk::metric<"step", std::uint64_t>("Physical Step", {}, "Simulation"), spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true));

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

} // namespace physica::examples::pbf_dam_break
