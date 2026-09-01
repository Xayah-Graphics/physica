module;

#include <spectra/sdk/cuda_types.h>

export module physica.example.fluids.liquid.flip_apic_dam_break.module;

import std;
import physica.example.fluids.liquid.flip_apic_dam_break;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::flip_apic_dam_break {
    struct Settings final {};

    struct Module final {
        [[no_unique_address]] Settings settings;

        static constexpr auto description = spectra::sdk::describe("physica.example.fluids.liquid.flip_apic_dam_break", spectra::sdk::particles<"flip-particles">(spectra::sdk::field<"velocity", spectra::sdk::Float3>("Velocity", "m/s"), spectra::sdk::field<"speed", float>("Speed", "m/s")), spectra::sdk::particles<"apic-particles">(spectra::sdk::field<"velocity", spectra::sdk::Float3>("Velocity", "m/s"), spectra::sdk::field<"speed", float>("Speed", "m/s"), spectra::sdk::field<"affine-magnitude", float>("Affine Magnitude", "1/s")),
            spectra::sdk::volume<"flip-grid">(spectra::sdk::volume_field<"velocity", spectra::sdk::MacFloat3>("Velocity", "m/s", {.vector_space = spectra::sdk::VolumeVectorSpace::World}), spectra::sdk::volume_field<"pressure", float>("Pressure", "Pa", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"divergence", float>("Divergence", "1/s", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"level-set", float>("Particle Level Set", "m", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"cell-type", std::uint32_t>("Cell Type", {}, {.sampling = spectra::sdk::VolumeFieldSampling::Cell})),
            spectra::sdk::volume<"apic-grid">(spectra::sdk::volume_field<"velocity", spectra::sdk::MacFloat3>("Velocity", "m/s", {.vector_space = spectra::sdk::VolumeVectorSpace::World}), spectra::sdk::volume_field<"pressure", float>("Pressure", "Pa", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"divergence", float>("Divergence", "1/s", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"level-set", float>("Particle Level Set", "m", {.sampling = spectra::sdk::VolumeFieldSampling::Cell}), spectra::sdk::volume_field<"cell-type", std::uint32_t>("Cell Type", {}, {.sampling = spectra::sdk::VolumeFieldSampling::Cell})), spectra::sdk::metric<"step", std::uint64_t>("Frame", {}, "Simulation"), spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true), spectra::sdk::metric<"flip-particle-count", std::uint64_t>("FLIP Particles", {}, "FLIP"), spectra::sdk::metric<"apic-particle-count", std::uint64_t>("APIC Particles", {}, "APIC"),
            spectra::sdk::metric<"flip-substeps", std::uint64_t>("FLIP Substeps", {}, "FLIP"), spectra::sdk::metric<"apic-substeps", std::uint64_t>("APIC Substeps", {}, "APIC"), spectra::sdk::metric<"flip-pressure-iterations", std::uint64_t>("FLIP Pressure Iterations", {}, "FLIP"), spectra::sdk::metric<"apic-pressure-iterations", std::uint64_t>("APIC Pressure Iterations", {}, "APIC"), spectra::sdk::metric<"flip-pressure-residual", float>("FLIP Pressure Residual", {}, "FLIP", true), spectra::sdk::metric<"apic-pressure-residual", float>("APIC Pressure Residual", {}, "APIC", true), spectra::sdk::metric<"flip-divergence-before", float>("FLIP Divergence Before", "1/s", "FLIP", true), spectra::sdk::metric<"flip-divergence-after", float>("FLIP Divergence After", "1/s", "FLIP", true), spectra::sdk::metric<"apic-divergence-before", float>("APIC Divergence Before", "1/s", "APIC", true), spectra::sdk::metric<"apic-divergence-after", float>("APIC Divergence After", "1/s", "APIC", true),
            spectra::sdk::metric<"flip-kinetic-energy", float>("FLIP Kinetic Energy", "J", "FLIP", true), spectra::sdk::metric<"apic-kinetic-energy", float>("APIC Kinetic Energy", "J", "APIC", true), spectra::sdk::metric<"flip-linear-momentum", spectra::sdk::Float3>("FLIP Linear Momentum", "kg m/s", "FLIP"), spectra::sdk::metric<"apic-linear-momentum", spectra::sdk::Float3>("APIC Linear Momentum", "kg m/s", "APIC"), spectra::sdk::metric<"flip-angular-momentum", spectra::sdk::Float3>("FLIP Angular Momentum", "kg m2/s", "FLIP"), spectra::sdk::metric<"apic-angular-momentum", spectra::sdk::Float3>("APIC Angular Momentum", "kg m2/s", "APIC"));

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

} // namespace physica::examples::flip_apic_dam_break
