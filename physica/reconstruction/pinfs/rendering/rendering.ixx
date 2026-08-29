module;

#include <physica/cuda.h>

export module physica.reconstruction.pinfs.rendering;

import std;
export import physica.math;
import physica.reconstruction.pinfs.field;
import physica.reconstruction.pinfs.network;
import physica.reconstruction.pinfs.sampling;

export namespace physica::reconstruction::pinfs {
    struct RenderingOutput final {
        Vector3<float>* rgb{};
        float* accumulation{};
        Vector3<float>* dynamic_rgb{};
        float* dynamic_accumulation{};
        Vector3<float>* static_rgb{};
        float* static_accumulation{};
        float* weights{};
        std::uint32_t ray_count{};
        std::uint32_t samples_per_ray{};
    };

    struct RenderingLossWeights final {
        float temporal_fading{};
        float ghost{};
        float ghost_scale{};
        float overlay{};
        float eikonal{};
        float deviation{};
        std::uint32_t step{};
    };

    struct FieldAdjoints final {
        ConstDeviceTensor coarse;
        ConstDeviceTensor dynamic;
        ConstDeviceTensor static_color;
        const float* sdf{};
        const float* sdf_gradient{};
        const float* inverse_deviation{};
    };

    struct RenderingLossStatistics final {
        float total{};
        float image{};
        float coarse_image{};
        float perceptual{};
        float ghost{};
        float overlay{};
        float eikonal{};
        float deviation{};
    };

    struct Rendering final {
        Rendering(::cuda::stream_ref stream, std::uint32_t maximum_ray_count, std::uint32_t maximum_samples_per_ray);

        void begin_step();
        RenderingOutput coarse(const RaySamples& samples, ConstDeviceTensor dynamic, const Vector3<float>* background, const AxisAlignedBox3<float>* bounds);
        RenderingOutput fine(const RaySamples& samples, ConstDeviceTensor dynamic, const StaticFieldOutput* static_output, const Vector3<float>* background, const AxisAlignedBox3<float>* bounds);
        FieldAdjoints backward(const RaySamples& coarse_samples, ConstDeviceTensor coarse_dynamic, const RaySamples& fine_samples, ConstDeviceTensor fine_dynamic, const StaticFieldOutput* static_output, const Vector3<float>* target, const Vector3<float>* background, const AxisAlignedBox3<float>* bounds, RenderingLossWeights weights, const Vector3<float>* fine_perceptual_adjoints, const Vector3<float>* coarse_perceptual_adjoints, float perceptual_loss, std::uint32_t normalization_ray_count, std::uint32_t normalization_sample_count);
        [[nodiscard]] RenderingLossStatistics end_step() const;

    private:
        struct Storage final {
            Storage(::cuda::stream_ref stream, std::uint32_t maximum_ray_count, std::uint32_t maximum_samples_per_ray);

            ::cuda::device_buffer<Vector3<float>> rgb;
            ::cuda::device_buffer<float> accumulation;
            ::cuda::device_buffer<Vector3<float>> dynamic_rgb;
            ::cuda::device_buffer<float> dynamic_accumulation;
            ::cuda::device_buffer<Vector3<float>> static_rgb;
            ::cuda::device_buffer<float> static_accumulation;
            ::cuda::device_buffer<float> weights;
            ::cuda::device_buffer<float> dynamic_alpha;
            ::cuda::device_buffer<float> static_alpha;
            ::cuda::device_buffer<float> shared_transmittance;
            ::cuda::device_buffer<float> dynamic_transmittance;
            ::cuda::device_buffer<float> static_transmittance;
            ::cuda::device_buffer<Vector3<float>> rgb_adjoints;
            ::cuda::device_buffer<float> accumulation_adjoints;
            ::cuda::device_buffer<Vector3<float>> dynamic_rgb_adjoints;
            ::cuda::device_buffer<float> dynamic_accumulation_adjoints;
            ::cuda::device_buffer<Vector3<float>> static_rgb_adjoints;
            ::cuda::device_buffer<float> static_accumulation_adjoints;
        };

        ::cuda::stream_ref stream;
        Storage coarse_storage;
        Storage fine_storage;
        ::cuda::device_buffer<float> coarse_dynamic_adjoints;
        ::cuda::device_buffer<float> fine_dynamic_adjoints;
        ::cuda::device_buffer<float> static_color_adjoints;
        ::cuda::device_buffer<float> sdf_adjoints;
        ::cuda::device_buffer<float> sdf_gradient_adjoints;
        ::cuda::device_buffer<float> inverse_deviation_adjoint;
        ::cuda::device_buffer<double> losses;
    };
} // namespace physica::reconstruction::pinfs
