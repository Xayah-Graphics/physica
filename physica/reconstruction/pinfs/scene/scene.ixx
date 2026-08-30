module;

#include <physica/cuda.h>

export module physica.reconstruction.pinfs.scene;

import std;
import physica.reconstruction.dataset.pinf;
export import physica.math;

export namespace physica::reconstruction::pinfs {
    struct SpacetimePoint final {
        Vector3<float> position;
        float time = 0.0F;
    };

    static_assert(sizeof(SpacetimePoint) == 4uz * sizeof(float));

    struct DeviceRays final {
        const Ray3<float>* data = nullptr;
        std::uint32_t count     = 0u;
    };

    struct TrainingRays final {
        DeviceRays rays;
        const Vector3<float>* target = nullptr;
        float time                   = 0.0F;
    };

    struct Scene final {
        const dataset::pinf::Dataset& dataset;
        const dataset::multiview::FrameSet& training;
        std::vector<Vector3<float>> voxel_positions;
        ::cuda::device_buffer<Vector3<float>> background;
        ::cuda::device_buffer<AxisAlignedBox3<float>> bounds;

        Scene(const dataset::pinf::Dataset& dataset, ::cuda::stream_ref stream, std::uint32_t rays_per_step, std::uint32_t inference_batch_size, std::uint32_t volume_width, std::uint32_t perceptual_stride, std::uint32_t central_crop_steps, float central_crop_fraction, std::optional<AxisAlignedBox3<float>> normalized_bounds);

        TrainingRays prepare_training_rays(const dataset::multiview::Frame& frame, bool perceptual, std::uint32_t seed, std::uint32_t step);
        DeviceRays prepare_rendering_rays(const dataset::multiview::Frame& frame, std::uint32_t offset, std::uint32_t count);
        const SpacetimePoint* prepare_volume_points(float time, Vector3<std::uint32_t> resolution, std::size_t offset, std::uint32_t count);

    private:
        ::cuda::stream_ref stream;
        std::uint32_t rays_per_step      = 0u;
        std::uint32_t perceptual_stride  = 0u;
        std::uint32_t central_crop_steps = 0u;
        float central_crop_fraction      = 0.0F;
        Matrix4<float> world_from_voxel;
        ::cuda::host_buffer<Ray3<float>> host_rays;
        ::cuda::host_buffer<Vector3<float>> host_targets;
        ::cuda::host_buffer<SpacetimePoint> host_volume_points;
        ::cuda::device_buffer<Ray3<float>> device_rays;
        ::cuda::device_buffer<Vector3<float>> device_targets;
        ::cuda::device_buffer<SpacetimePoint> device_volume_points;
        std::vector<std::uint32_t> pixels;
        std::vector<std::uint32_t> candidates;
    };
} // namespace physica::reconstruction::pinfs
