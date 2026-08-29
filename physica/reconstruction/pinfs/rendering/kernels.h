#ifndef PHYSICA_RECONSTRUCTION_PINFS_RENDERING_KERNELS_H
#define PHYSICA_RECONSTRUCTION_PINFS_RENDERING_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::reconstruction::pinfs::kernels {
    struct RenderViews final {
        float* rgb;
        float* accumulation;
        float* dynamic_rgb;
        float* dynamic_accumulation;
        float* static_rgb;
        float* static_accumulation;
        float* weights;
        float* dynamic_alpha;
        float* static_alpha;
        float* shared_transmittance;
        float* dynamic_transmittance;
        float* static_transmittance;
    };

    struct RenderAdjointViews final {
        float* rgb;
        float* accumulation;
        float* dynamic_rgb;
        float* dynamic_accumulation;
        float* static_rgb;
        float* static_accumulation;
    };

    void render_dynamic(::cuda::stream_ref stream, const float* z, const float* positions, const float* directions, const float* dynamic, const float* background, const float* aabb, RenderViews output, std::uint32_t ray_count, std::uint32_t samples_per_ray);
    void render_hybrid(::cuda::stream_ref stream, const float* z, const float* positions, const float* directions, const float* dynamic, const float* static_color, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* background, const float* aabb, RenderViews output, std::uint32_t ray_count, std::uint32_t samples_per_ray);
    void initialize_loss(::cuda::stream_ref stream, RenderViews coarse, RenderViews fine, RenderAdjointViews coarse_adjoints, RenderAdjointViews fine_adjoints, const float* target, const float* background, const float* fine_perceptual_adjoints, const float* coarse_perceptual_adjoints, double* losses, std::uint32_t ray_count, std::uint32_t normalization_ray_count, bool hybrid, float temporal_fading, float ghost_weight, float ghost_scale, float perceptual_loss);
    void regularization_loss(::cuda::stream_ref stream, const float* positions, const float* dynamic, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* aabb, float* dynamic_adjoints, float* sdf_adjoints, float* sdf_gradient_adjoints, float* inverse_deviation_adjoint, double* losses, std::uint32_t sample_count, std::uint32_t normalization_sample_count, float overlay_weight, float eikonal_weight, float deviation_weight, std::uint32_t step);
    void backward_dynamic(::cuda::stream_ref stream, const float* z, const float* positions, const float* directions, const float* dynamic, const float* background, const float* aabb, RenderViews output, RenderAdjointViews output_adjoints, float* dynamic_adjoints, std::uint32_t ray_count, std::uint32_t samples_per_ray);
    void backward_hybrid(::cuda::stream_ref stream, const float* z, const float* positions, const float* directions, const float* dynamic, const float* static_color, const float* sdf, const float* sdf_derivatives, const float* inverse_deviation, const float* background, const float* aabb, RenderViews output, RenderAdjointViews output_adjoints, float* dynamic_adjoints, float* static_color_adjoints, float* sdf_adjoints, float* sdf_gradient_adjoints, float* inverse_deviation_adjoint, std::uint32_t ray_count, std::uint32_t samples_per_ray);
} // namespace physica::reconstruction::pinfs::kernels

#endif
