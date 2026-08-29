module;

#include <physica/cuda.h>

export module physica.reconstruction.pinfs.sampling;

import std;
import physica.reconstruction.pinfs.network;
import physica.reconstruction.pinfs.scene;

export namespace physica::reconstruction::pinfs {
    struct RaySamples final {
        float* z{};
        Vector3<float>* positions{};
        SpacetimePoint* dynamic_points{};
        Vector3<float>* directions{};
        std::uint32_t ray_count{};
        std::uint32_t samples_per_ray{};

        [[nodiscard]] std::uint32_t sample_count() const noexcept {
            return ray_count * samples_per_ray;
        }
    };

    struct Sampling final {
        Sampling(::cuda::stream_ref stream, std::uint32_t maximum_ray_count, std::uint32_t maximum_samples_per_ray, std::uint32_t maximum_importance_samples);

        RaySamples coarse(DeviceRays rays, std::uint32_t samples_per_ray, float near_distance, float far_distance, float time, std::uint32_t seed, std::uint32_t step, std::uint32_t ray_offset, bool perturb);
        void warp(RaySamples samples, ConstDeviceTensor velocity, float amount, std::uint32_t seed, std::uint32_t step, std::uint32_t ray_offset);
        void sample_pdf(const RaySamples& coarse_samples, const float* coarse_weights, std::uint32_t importance_count, std::uint32_t seed, std::uint32_t step, std::uint32_t ray_offset, bool deterministic);
        void begin_neus_upsampling(const RaySamples& coarse_samples, ConstDeviceTensor sdf);
        RaySamples neus_upsampling_positions(DeviceRays rays, std::uint32_t new_sample_count, float inverse_deviation);
        void commit_neus_upsampling(ConstDeviceTensor new_sdf, bool last);
        RaySamples fine(DeviceRays rays, const RaySamples& coarse_samples, std::uint32_t pdf_sample_count, std::uint32_t static_sample_count, float time);

    private:
        ::cuda::stream_ref stream;
        ::cuda::device_buffer<float> coarse_z;
        ::cuda::device_buffer<Vector3<float>> coarse_positions;
        ::cuda::device_buffer<SpacetimePoint> coarse_dynamic_points;
        ::cuda::device_buffer<Vector3<float>> coarse_directions;
        ::cuda::device_buffer<float> fine_z;
        ::cuda::device_buffer<Vector3<float>> fine_positions;
        ::cuda::device_buffer<SpacetimePoint> fine_dynamic_points;
        ::cuda::device_buffer<Vector3<float>> fine_directions;
        ::cuda::device_buffer<float> pdf_z;
        ::cuda::device_buffer<float> upsample_z_a;
        ::cuda::device_buffer<float> upsample_z_b;
        ::cuda::device_buffer<float> upsample_sdf_a;
        ::cuda::device_buffer<float> upsample_sdf_b;
        ::cuda::device_buffer<float> new_z;
        ::cuda::device_buffer<Vector3<float>> new_positions;
        ::cuda::device_buffer<float> new_sdf;
        float* current_upsample_z{};
        float* next_upsample_z{};
        float* current_upsample_sdf{};
        float* next_upsample_sdf{};
        std::uint32_t upsample_ray_count{};
        std::uint32_t upsample_sample_count{};
        std::uint32_t pending_new_sample_count{};
    };
} // namespace physica::reconstruction::pinfs
