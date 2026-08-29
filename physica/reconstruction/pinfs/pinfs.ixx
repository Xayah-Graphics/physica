module;

#include <physica/cuda.h>

export module physica.reconstruction.pinfs;

import std;
import physica.reconstruction.dataset.pinf;
export import physica.math;
export import physica.reconstruction.pinfs.scene;
export import physica.reconstruction.pinfs.network;
export import physica.reconstruction.pinfs.field;
export import physica.reconstruction.pinfs.sampling;
export import physica.reconstruction.pinfs.rendering;
export import physica.reconstruction.pinfs.physics;
export import physica.reconstruction.pinfs.perceptual;

export namespace physica::reconstruction::pinfs {
    enum class SceneRepresentation : std::uint8_t {
        dynamic_volume,
        hybrid_surface,
    };

    struct Configuration final {
        SceneRepresentation representation          = SceneRepresentation::dynamic_volume;
        std::uint32_t position_frequency_count      = 0u;
        float first_frequency                       = 30.0F;
        bool unique_first_frequency                 = false;
        std::uint32_t coarse_samples                = 64u;
        std::uint32_t importance_samples            = 0u;
        std::uint32_t rays_per_step                 = 1024u;
        std::uint32_t rays_per_batch                = 128u;
        std::uint32_t inference_batch_size          = 32u * 1024u;
        std::uint32_t central_crop_steps            = 500u;
        float central_crop_fraction                 = 0.5F;
        std::uint32_t learning_rate_decay_thousands = 500u;
        std::uint32_t layer_fading_steps            = 50'000u;
        std::uint32_t temporal_fading_steps         = 2'000u;
        std::uint32_t velocity_delay_steps          = 20'000u;
        std::uint32_t volume_width                  = 128u;
        std::uint32_t perceptual_stride             = 4u;
        float learning_rate                         = 5.0e-4F;
        float perceptual_weight                     = 0.0F;
        float ghost_weight                          = 0.0F;
        float ghost_scale                           = 4.0F;
        float overlay_weight                        = 0.0F;
        float physics_weight                        = 1.0e-3F;
        float eikonal_weight                        = 1.0e-2F;
        float deviation_weight                      = 0.0F;
        float neumann_weight                        = 0.0F;
        bool train_warp                             = true;
        std::optional<AxisAlignedBox3<float>> normalized_bounds;
        std::optional<float> near_distance;
        std::optional<float> far_distance;
    };

    inline constexpr Configuration sphere_neus_configuration{
        .representation     = SceneRepresentation::hybrid_surface,
        .coarse_samples     = 32u,
        .importance_samples = 64u,
        .perceptual_weight  = 0.003F,
        .ghost_weight       = 0.003F,
        .ghost_scale        = 9.0F,
        .overlay_weight     = 0.002F,
        .neumann_weight     = 1.0F,
        .normalized_bounds  = AxisAlignedBox3<float>{.minimum = {0.0F, 0.0F, 0.0F}, .maximum = {1.0F, 1.0F, 1.0F}},
    };

    inline constexpr Configuration game_neus_configuration{
        .representation           = SceneRepresentation::hybrid_surface,
        .position_frequency_count = 6u,
        .first_frequency          = 6.0F,
        .unique_first_frequency   = true,
        .coarse_samples           = 40u,
        .importance_samples       = 80u,
        .temporal_fading_steps    = 10'000u,
        .perceptual_stride        = 3u,
        .perceptual_weight        = 0.003F,
        .ghost_weight             = 0.003F,
        .ghost_scale              = 36.0F,
        .overlay_weight           = 0.002F,
        .neumann_weight           = 1.0F,
        .normalized_bounds        = AxisAlignedBox3<float>{.minimum = {0.0F, -0.1F, -0.1F}, .maximum = {1.2F, 0.9F, 1.35F}},
        .near_distance            = 0.5F,
        .far_distance             = 6.0F,
    };

    inline constexpr Configuration scalar_real_configuration{
        .coarse_samples       = 16u,
        .importance_samples   = 32u,
        .central_crop_steps   = 1'000u,
        .velocity_delay_steps = 10'000u,
        .perceptual_weight    = 0.01F,
        .ghost_weight         = 0.07F,
        .normalized_bounds    = AxisAlignedBox3<float>{.minimum = {0.05F, 0.05F, 0.05F}, .maximum = {0.9F, 0.9F, 0.9F}},
    };

    struct TrainingState final {
        std::uint32_t seed = 42u;
        std::uint32_t step = 0u;
    };

    struct OptimizationStats final {
        std::uint32_t begin_step = 0u;
        std::uint32_t end_step   = 0u;
        float loss               = 0.0F;
        float image_loss         = 0.0F;
        float coarse_image_loss  = 0.0F;
        float perceptual_loss    = 0.0F;
        float ghost_loss         = 0.0F;
        float overlay_loss       = 0.0F;
        float eikonal_loss       = 0.0F;
        float deviation_loss     = 0.0F;
        float physics_loss       = 0.0F;
        float neumann_loss       = 0.0F;
        float psnr               = 0.0F;
        float elapsed_ms         = 0.0F;
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

    struct RenderedFrame final {
        std::uint32_t width  = 0u;
        std::uint32_t height = 0u;
        std::vector<Vector3<float>> rgb;
    };

    struct VolumeSnapshot final {
        Vector3<std::uint32_t> resolution;
        std::vector<float> density;
        std::vector<Vector3<float>> velocity;
    };

    struct PINFS final {
        TrainingState state;

        PINFS(const dataset::pinf::Dataset& dataset, Configuration configuration, const std::filesystem::path& perceptual_weights, std::uint32_t device_ordinal, std::uint32_t seed = 42u);
        ~PINFS() noexcept;

        PINFS(const PINFS&)            = delete;
        PINFS& operator=(const PINFS&) = delete;
        PINFS(PINFS&&)                 = delete;
        PINFS& operator=(PINFS&&)      = delete;

        OptimizationStats optimize(std::uint32_t iterations);
        RenderedFrame render(const dataset::multiview::Frame& frame);
        EvaluationStats evaluate(std::string_view frame_set, std::uint32_t maximum_frames, std::uint32_t stride);
        VolumeSnapshot sample_volume(float time, Vector3<std::uint32_t> resolution);
        void save(const std::filesystem::path& path) const;
        void load(const std::filesystem::path& path);

    private:
        struct ForwardPass final {
            RaySamples coarse_samples;
            DeviceTensor coarse_dynamic;
            RenderingOutput coarse_output;
            RaySamples fine_samples;
            DeviceTensor dynamic;
            StaticFieldOutput static_output;
            RenderingOutput output;
        };

        [[nodiscard]] float learning_rate() const;
        ForwardPass forward(DeviceRays rays, float time, bool perturb, std::uint32_t ray_offset);

        Configuration configuration;
        ::cuda::stream stream;
        Scene scene;
        Sampling sampling;
        Rendering rendering;
        DynamicField coarse_field;
        DynamicField dynamic_field;
        VelocityField velocity_field;
        std::optional<StaticField> static_field;
        ::cuda::device_buffer<Vector3<float>> fine_rendered;
        ::cuda::device_buffer<Vector3<float>> coarse_rendered;
        ::cuda::device_buffer<Vector3<float>> fine_perceptual_adjoints;
        ::cuda::device_buffer<Vector3<float>> coarse_perceptual_adjoints;
        Physics physics;
        std::optional<PerceptualLoss> perceptual_loss;
    };
} // namespace physica::reconstruction::pinfs
