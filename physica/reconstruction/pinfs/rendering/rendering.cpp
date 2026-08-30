module;

#include "kernels.h"
#include <physica/cuda.h>

module physica.reconstruction.pinfs.rendering;

import std;
import physica.reconstruction.pinfs.field;
import physica.reconstruction.pinfs.sampling;

namespace physica::reconstruction::pinfs {
    namespace {
        kernels::RenderViews render_views(auto& storage) {
            return {
                .rgb                   = reinterpret_cast<float*>(storage.rgb.data()),
                .accumulation          = storage.accumulation.data(),
                .dynamic_rgb           = reinterpret_cast<float*>(storage.dynamic_rgb.data()),
                .dynamic_accumulation  = storage.dynamic_accumulation.data(),
                .static_rgb            = reinterpret_cast<float*>(storage.static_rgb.data()),
                .static_accumulation   = storage.static_accumulation.data(),
                .weights               = storage.weights.data(),
                .dynamic_alpha         = storage.dynamic_alpha.data(),
                .static_alpha          = storage.static_alpha.data(),
                .shared_transmittance  = storage.shared_transmittance.data(),
                .dynamic_transmittance = storage.dynamic_transmittance.data(),
                .static_transmittance  = storage.static_transmittance.data(),
            };
        }

        kernels::RenderAdjointViews render_adjoint_views(auto& storage) {
            return {
                .rgb                  = reinterpret_cast<float*>(storage.rgb_adjoints.data()),
                .accumulation         = storage.accumulation_adjoints.data(),
                .dynamic_rgb          = reinterpret_cast<float*>(storage.dynamic_rgb_adjoints.data()),
                .dynamic_accumulation = storage.dynamic_accumulation_adjoints.data(),
                .static_rgb           = reinterpret_cast<float*>(storage.static_rgb_adjoints.data()),
                .static_accumulation  = storage.static_accumulation_adjoints.data(),
            };
        }
    } // namespace

    Rendering::Rendering(const ::cuda::stream_ref source_stream, const std::uint32_t source_maximum_ray_count, const std::uint32_t source_maximum_samples_per_ray)
        : stream{source_stream}, coarse_storage{stream, source_maximum_ray_count, source_maximum_samples_per_ray}, fine_storage{stream, source_maximum_ray_count, source_maximum_samples_per_ray}, coarse_dynamic_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray * 4uz, ::cuda::no_init}, fine_dynamic_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray * 4uz, ::cuda::no_init}, static_color_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray * 3uz, ::cuda::no_init}, sdf_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray, ::cuda::no_init},
          sdf_gradient_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(source_maximum_ray_count) * source_maximum_samples_per_ray * 3uz, ::cuda::no_init}, inverse_deviation_adjoint{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init}, losses{stream, ::cuda::device_default_memory_pool(stream.device()), 8uz, ::cuda::no_init} {}

    void Rendering::begin_step() {
        ::cuda::fill_bytes(stream, losses, 0u);
    }

    RenderingOutput Rendering::coarse(const RaySamples& samples, const ConstDeviceTensor dynamic, const Vector3<float>* background, const AxisAlignedBox3<float>* bounds) {
        kernels::render_dynamic(stream, samples.z, reinterpret_cast<const float*>(samples.positions), reinterpret_cast<const float*>(samples.directions), dynamic.values, reinterpret_cast<const float*>(background), reinterpret_cast<const float*>(bounds), render_views(coarse_storage), samples.ray_count, samples.samples_per_ray);
        return {.rgb = coarse_storage.rgb.data(), .accumulation = coarse_storage.accumulation.data(), .dynamic_rgb = coarse_storage.dynamic_rgb.data(), .dynamic_accumulation = coarse_storage.dynamic_accumulation.data(), .static_rgb = coarse_storage.static_rgb.data(), .static_accumulation = coarse_storage.static_accumulation.data(), .weights = coarse_storage.weights.data(), .ray_count = samples.ray_count, .samples_per_ray = samples.samples_per_ray};
    }

    RenderingOutput Rendering::fine(const RaySamples& samples, const ConstDeviceTensor dynamic, const StaticFieldOutput* static_output, const Vector3<float>* background, const AxisAlignedBox3<float>* bounds) {
        if (static_output == nullptr) kernels::render_dynamic(stream, samples.z, reinterpret_cast<const float*>(samples.positions), reinterpret_cast<const float*>(samples.directions), dynamic.values, reinterpret_cast<const float*>(background), reinterpret_cast<const float*>(bounds), render_views(fine_storage), samples.ray_count, samples.samples_per_ray);
        else kernels::render_hybrid(stream, samples.z, reinterpret_cast<const float*>(samples.positions), reinterpret_cast<const float*>(samples.directions), dynamic.values, static_output->color.values, static_output->sdf.values, static_output->sdf.derivatives, static_output->inverse_deviation, reinterpret_cast<const float*>(background), reinterpret_cast<const float*>(bounds), render_views(fine_storage), samples.ray_count, samples.samples_per_ray);
        return {.rgb = fine_storage.rgb.data(), .accumulation = fine_storage.accumulation.data(), .dynamic_rgb = fine_storage.dynamic_rgb.data(), .dynamic_accumulation = fine_storage.dynamic_accumulation.data(), .static_rgb = fine_storage.static_rgb.data(), .static_accumulation = fine_storage.static_accumulation.data(), .weights = fine_storage.weights.data(), .ray_count = samples.ray_count, .samples_per_ray = samples.samples_per_ray};
    }

    FieldAdjoints Rendering::backward(const RaySamples& coarse_samples, const ConstDeviceTensor coarse_dynamic, const RaySamples& fine_samples, const ConstDeviceTensor fine_dynamic, const StaticFieldOutput* static_output, const Vector3<float>* target, const Vector3<float>* background, const AxisAlignedBox3<float>* bounds, const RenderingLossWeights weights, const Vector3<float>* fine_perceptual_adjoints, const Vector3<float>* coarse_perceptual_adjoints, const float perceptual_loss, const std::uint32_t normalization_ray_count, const std::uint32_t normalization_sample_count) {
        const std::size_t coarse_sample_count = coarse_samples.sample_count();
        const std::size_t fine_sample_count   = fine_samples.sample_count();
        ::cuda::fill_bytes(stream, ::cuda::std::span<float>{coarse_dynamic_adjoints.data(), coarse_sample_count * 4uz}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span<float>{fine_dynamic_adjoints.data(), fine_sample_count * 4uz}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span<float>{static_color_adjoints.data(), fine_sample_count * 3uz}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span<float>{sdf_adjoints.data(), fine_sample_count}, 0u);
        ::cuda::fill_bytes(stream, ::cuda::std::span<float>{sdf_gradient_adjoints.data(), fine_sample_count * 3uz}, 0u);
        ::cuda::fill_bytes(stream, inverse_deviation_adjoint, 0u);
        kernels::initialize_loss(stream, render_views(coarse_storage), render_views(fine_storage), render_adjoint_views(coarse_storage), render_adjoint_views(fine_storage), reinterpret_cast<const float*>(target), reinterpret_cast<const float*>(background), reinterpret_cast<const float*>(fine_perceptual_adjoints), reinterpret_cast<const float*>(coarse_perceptual_adjoints), losses.data(), fine_samples.ray_count, normalization_ray_count, static_output != nullptr, weights.temporal_fading, weights.ghost, weights.ghost_scale, perceptual_loss);
        kernels::backward_dynamic(stream, coarse_samples.z, reinterpret_cast<const float*>(coarse_samples.positions), reinterpret_cast<const float*>(coarse_samples.directions), coarse_dynamic.values, reinterpret_cast<const float*>(background), reinterpret_cast<const float*>(bounds), render_views(coarse_storage), render_adjoint_views(coarse_storage), coarse_dynamic_adjoints.data(), coarse_samples.ray_count, coarse_samples.samples_per_ray);
        if (static_output == nullptr) kernels::backward_dynamic(stream, fine_samples.z, reinterpret_cast<const float*>(fine_samples.positions), reinterpret_cast<const float*>(fine_samples.directions), fine_dynamic.values, reinterpret_cast<const float*>(background), reinterpret_cast<const float*>(bounds), render_views(fine_storage), render_adjoint_views(fine_storage), fine_dynamic_adjoints.data(), fine_samples.ray_count, fine_samples.samples_per_ray);
        else {
            kernels::regularization_loss(stream, reinterpret_cast<const float*>(fine_samples.positions), fine_dynamic.values, static_output->sdf.values, static_output->sdf.derivatives, static_output->inverse_deviation, reinterpret_cast<const float*>(bounds), fine_dynamic_adjoints.data(), sdf_adjoints.data(), sdf_gradient_adjoints.data(), inverse_deviation_adjoint.data(), losses.data(), fine_samples.sample_count(), normalization_sample_count, weights.overlay, weights.eikonal, weights.deviation, weights.step);
            kernels::backward_hybrid(stream, fine_samples.z, reinterpret_cast<const float*>(fine_samples.positions), reinterpret_cast<const float*>(fine_samples.directions), fine_dynamic.values, static_output->color.values, static_output->sdf.values, static_output->sdf.derivatives, static_output->inverse_deviation, reinterpret_cast<const float*>(background), reinterpret_cast<const float*>(bounds), render_views(fine_storage), render_adjoint_views(fine_storage), fine_dynamic_adjoints.data(), static_color_adjoints.data(), sdf_adjoints.data(), sdf_gradient_adjoints.data(), inverse_deviation_adjoint.data(), fine_samples.ray_count, fine_samples.samples_per_ray);
        }
        return {
            .coarse            = {.values = coarse_dynamic_adjoints.data(), .width = 4u, .sample_count = coarse_samples.sample_count()},
            .dynamic           = {.values = fine_dynamic_adjoints.data(), .width = 4u, .sample_count = fine_samples.sample_count()},
            .static_color      = {.values = static_color_adjoints.data(), .width = 3u, .sample_count = fine_samples.sample_count()},
            .sdf               = sdf_adjoints.data(),
            .sdf_gradient      = sdf_gradient_adjoints.data(),
            .inverse_deviation = inverse_deviation_adjoint.data(),
        };
    }

    RenderingLossStatistics Rendering::end_step() const {
        std::array<double, 8> host{};
        ::cuda::copy_bytes(stream, losses, ::cuda::std::span<double>{host.data(), host.size()});
        stream.sync();
        return {.total = static_cast<float>(host[0]), .image = static_cast<float>(host[1]), .coarse_image = static_cast<float>(host[2]), .perceptual = static_cast<float>(host[3]), .ghost = static_cast<float>(host[4]), .overlay = static_cast<float>(host[5]), .eikonal = static_cast<float>(host[6]), .deviation = static_cast<float>(host[7])};
    }

    Rendering::Storage::Storage(const ::cuda::stream_ref stream, const std::uint32_t maximum_ray_count, const std::uint32_t maximum_samples_per_ray)
        : rgb{stream, ::cuda::device_default_memory_pool(stream.device()), maximum_ray_count, ::cuda::no_init}, accumulation{stream, ::cuda::device_default_memory_pool(stream.device()), maximum_ray_count, ::cuda::no_init}, dynamic_rgb{stream, ::cuda::device_default_memory_pool(stream.device()), maximum_ray_count, ::cuda::no_init}, dynamic_accumulation{stream, ::cuda::device_default_memory_pool(stream.device()), maximum_ray_count, ::cuda::no_init}, static_rgb{stream, ::cuda::device_default_memory_pool(stream.device()), maximum_ray_count, ::cuda::no_init}, static_accumulation{stream, ::cuda::device_default_memory_pool(stream.device()), maximum_ray_count, ::cuda::no_init}, weights{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(maximum_ray_count) * maximum_samples_per_ray, ::cuda::no_init}, dynamic_alpha{stream, ::cuda::device_default_memory_pool(stream.device()), weights.size(), ::cuda::no_init},
          static_alpha{stream, ::cuda::device_default_memory_pool(stream.device()), weights.size(), ::cuda::no_init}, shared_transmittance{stream, ::cuda::device_default_memory_pool(stream.device()), weights.size(), ::cuda::no_init}, dynamic_transmittance{stream, ::cuda::device_default_memory_pool(stream.device()), weights.size(), ::cuda::no_init}, static_transmittance{stream, ::cuda::device_default_memory_pool(stream.device()), weights.size(), ::cuda::no_init}, rgb_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), rgb.size(), ::cuda::no_init}, accumulation_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), accumulation.size(), ::cuda::no_init}, dynamic_rgb_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), dynamic_rgb.size(), ::cuda::no_init}, dynamic_accumulation_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), dynamic_accumulation.size(), ::cuda::no_init},
          static_rgb_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_rgb.size(), ::cuda::no_init}, static_accumulation_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), static_accumulation.size(), ::cuda::no_init} {}
} // namespace physica::reconstruction::pinfs
