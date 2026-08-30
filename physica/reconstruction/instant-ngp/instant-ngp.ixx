module;

#include <physica/cuda.h>

export module physica.reconstruction.instant_ngp;

import std;
import physica.reconstruction.dataset.multiview;
export import physica.reconstruction.instant_ngp.scene;
export import physica.reconstruction.instant_ngp.network;
export import physica.reconstruction.instant_ngp.sampling;
export import physica.reconstruction.instant_ngp.rendering;

export namespace physica::reconstruction::instant_ngp {
    struct TrainingState final {
        std::uint32_t seed               = 1337u;
        std::uint32_t step               = 0u;
        std::uint32_t training_frame_set = 0u;
    };

    struct OptimizationStats final {
        std::uint32_t begin_step        = 0u;
        std::uint32_t end_step          = 0u;
        std::uint32_t rays              = 0u;
        std::uint32_t generated_samples = 0u;
        std::uint32_t compacted_samples = 0u;
        std::uint32_t occupied_cells    = 0u;
        float loss                      = 0.0F;
        float sample_efficiency         = 0.0F;
        float occupancy_ratio           = 0.0F;
        float elapsed_ms                = 0.0F;
    };

    struct EvaluationStats final {
        std::string frame_set;
        std::uint32_t step        = 0u;
        std::uint32_t frame_count = 0u;
        std::uint64_t pixel_count = 0u;
        float mse                 = 0.0F;
        float psnr                = 0.0F;
        float elapsed_ms          = 0.0F;
    };

    struct InstantNGPDeviceState final {
        void* stream = nullptr;
        NetworkDeviceState network;
        SamplingDeviceState sampling;
    };

    template <NetworkShape NetworkSpec, SamplingShape SamplingSpec, RenderingShape RenderingSpec>
    struct InstantNGP final {
        TrainingState state;

        InstantNGP(const dataset::multiview::Dataset& dataset, std::uint32_t device_ordinal, std::uint32_t training_frame_set, float scene_scale, std::uint32_t seed = 1337u);
        ~InstantNGP() noexcept;

        InstantNGP(const InstantNGP&)            = delete;
        InstantNGP& operator=(const InstantNGP&) = delete;
        InstantNGP(InstantNGP&&)                 = delete;
        InstantNGP& operator=(InstantNGP&&)      = delete;

        OptimizationStats optimize(std::uint32_t iterations);
        EvaluationStats evaluate(std::uint32_t frame_set);
        void save(const std::filesystem::path& path) const;
        void load(const std::filesystem::path& path);
        [[nodiscard]] InstantNGPDeviceState device_state() const noexcept;

    private:
        ::cuda::stream stream;
        Scene scene;
        Network<NetworkSpec> network;
        Sampling<SamplingSpec, NetworkSpec> sampling;
        Rendering<RenderingSpec, NetworkSpec> rendering;
    };

    extern template struct InstantNGP<nerf_synthetic_network_shape, nerf_synthetic_sampling_shape, nerf_synthetic_rendering_shape>;
} // namespace physica::reconstruction::instant_ngp
