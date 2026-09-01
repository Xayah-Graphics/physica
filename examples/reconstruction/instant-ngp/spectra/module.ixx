module;

export module physica.example.reconstruction.instant_ngp.module;

import std;
import physica.reconstruction.dataset.nerf_synthetic;
import physica.reconstruction.instant_ngp;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::instant_ngp {
    struct Settings final {};

    struct Module final {
        [[no_unique_address]] Settings settings;

        static constexpr auto description = spectra::sdk::describe("physica.example.reconstruction.instant-ngp", spectra::sdk::cameras<"training">(), spectra::sdk::hash_grid_radiance_field<"field">(), spectra::sdk::metric<"step", std::uint32_t>("Step", {}, "Training"), spectra::sdk::metric<"loss", float>("Loss", {}, "Training", true), spectra::sdk::metric<"psnr", float>("PSNR", "dB", "Training", true), spectra::sdk::metric<"sample-efficiency", float>("Sample efficiency", {}, "Training", true), spectra::sdk::metric<"occupancy", float>("Occupancy", {}, "Training", true));

        Module(Settings settings, const std::filesystem::path& assets, const spectra::sdk::SceneInputs& inputs);
        ~Module();

        Module(const Module&)            = delete;
        Module& operator=(const Module&) = delete;

        void setup(spectra::sdk::cuda::Setup& setup);
        void reset(std::uint64_t seed);
        void step(double seconds);
        void publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame);

    private:
        inline static constexpr float scene_scale = 0.33F;

        reconstruction::dataset::multiview::Dataset dataset;
        std::unique_ptr<reconstruction::instant_ngp::InstantNGP<reconstruction::instant_ngp::nerf_synthetic_network_shape, reconstruction::instant_ngp::nerf_synthetic_sampling_shape, reconstruction::instant_ngp::nerf_synthetic_rendering_shape>> instant_ngp;
        reconstruction::instant_ngp::OptimizationStats training{};
        float psnr = std::numeric_limits<float>::quiet_NaN();
    };

} // namespace physica::examples::instant_ngp
