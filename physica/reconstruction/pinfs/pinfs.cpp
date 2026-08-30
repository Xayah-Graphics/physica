module;

#include <cuda/std/random>
#include <physica/cuda.h>

module physica.reconstruction.pinfs;

import std;
import physica.reconstruction.dataset.pinf;
import physica.reconstruction.pinfs.scene;
import physica.reconstruction.pinfs.network;
import physica.reconstruction.pinfs.field;
import physica.reconstruction.pinfs.sampling;
import physica.reconstruction.pinfs.rendering;
import physica.reconstruction.pinfs.physics;
import physica.reconstruction.pinfs.perceptual;
import physica.serialization.safetensors;

namespace physica::reconstruction::pinfs {
    namespace {
        inline constexpr std::uint32_t pinfs_random_domain = 4u;

        enum class PINFSRandomSequence : std::uint32_t {
            training_frame,
        };

        float fade(const std::uint32_t step, const std::uint32_t begin, const std::uint32_t duration) {
            return std::clamp((static_cast<float>(step) - static_cast<float>(begin)) / static_cast<float>(duration), 0.0F, 1.0F);
        }

        const dataset::multiview::FrameSet& find_frame_set(const dataset::pinf::Dataset& dataset, const std::string_view name) {
            return *std::ranges::find_if(dataset.multiview.frame_sets, [&](const dataset::multiview::FrameSet& frame_set) { return frame_set.name == name; });
        }

        std::size_t training_frame_index(const std::uint32_t seed, const std::uint32_t step, const std::size_t frame_count) {
            ::cuda::std::philox4x32 random{seed};
            random.set_counter({pinfs_random_domain, static_cast<std::uint32_t>(PINFSRandomSequence::training_frame), step, 0u});
            return random() % frame_count;
        }
    } // namespace

    PINFS::PINFS(const dataset::pinf::Dataset& source_dataset, Configuration source_configuration, const std::filesystem::path& perceptual_weights, const std::uint32_t device_ordinal, const std::uint32_t seed)
        : state{.seed = seed}, configuration{std::move(source_configuration)}, stream{::cuda::devices[device_ordinal]}, scene{source_dataset, stream, configuration.rays_per_step, configuration.inference_batch_size, configuration.volume_width, configuration.perceptual_stride, configuration.central_crop_steps, configuration.central_crop_fraction, configuration.normalized_bounds}, sampling{stream, configuration.rays_per_batch, configuration.coarse_samples + configuration.importance_samples, configuration.importance_samples}, rendering{stream, configuration.rays_per_batch, configuration.coarse_samples + configuration.importance_samples}, coarse_field{stream, configuration.rays_per_batch * configuration.coarse_samples, 0u, 0u, seed, configuration.first_frequency, configuration.unique_first_frequency},
          dynamic_field{stream, std::ranges::max(std::array{configuration.rays_per_batch * (configuration.coarse_samples + configuration.importance_samples), physics_sample_count, configuration.inference_batch_size}), physics_sample_count, 4u, seed + 1u, configuration.first_frequency, configuration.unique_first_frequency}, velocity_field{stream, std::ranges::max(std::array{configuration.rays_per_batch * (configuration.coarse_samples + configuration.importance_samples), physics_sample_count, configuration.inference_batch_size}), physics_sample_count, seed + 2u}, fine_rendered{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.rays_per_step, ::cuda::no_init}, coarse_rendered{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.rays_per_step, ::cuda::no_init}, fine_perceptual_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.rays_per_step, ::cuda::no_init},
          coarse_perceptual_adjoints{stream, ::cuda::device_default_memory_pool(stream.device()), configuration.rays_per_step, ::cuda::no_init}, physics{stream, scene.voxel_positions} {
        if (configuration.representation == SceneRepresentation::hybrid_surface) static_field.emplace(stream, std::max(configuration.rays_per_batch * (configuration.coarse_samples + configuration.importance_samples), physics_sample_count), std::max(configuration.rays_per_batch * (configuration.coarse_samples + configuration.importance_samples), physics_sample_count), seed + 3u, configuration.position_frequency_count);
        if (configuration.perceptual_weight > 0.0F) perceptual_loss.emplace(stream, static_cast<std::uint32_t>(std::sqrt(static_cast<float>(configuration.rays_per_step))), perceptual_weights);
    }

    PINFS::~PINFS() noexcept = default;

    OptimizationStats PINFS::optimize(const std::uint32_t iterations) {
        const auto start = std::chrono::steady_clock::now();
        OptimizationStats statistics{.begin_step = state.step};
        for ([[maybe_unused]] const std::uint32_t iteration : std::views::iota(0u, iterations)) {
            coarse_field.set_fading_step(state.step, configuration.layer_fading_steps);
            dynamic_field.set_fading_step(state.step, configuration.layer_fading_steps);
            velocity_field.set_fading_step(state.step - std::min(state.step, configuration.velocity_delay_steps), configuration.layer_fading_steps);
            if (static_field) static_field->set_fading_step(std::max(1u, state.step), configuration.layer_fading_steps);
            const dataset::multiview::Frame& frame = scene.training.frames[training_frame_index(state.seed, state.step, scene.training.frames.size())];

            float physics_value{};
            float neumann_value{};
            if (state.step >= configuration.velocity_delay_steps && (state.step + 1u) % 10u == 0u) {
                const PhysicsSamples samples = physics.sample(frame.time, state.seed, state.step);
                const DeviceTensor velocity  = velocity_field.forward(samples.points);
                const DeviceTensor density   = dynamic_field.forward(samples.points);
                DeviceTensor static_sdf;
                const DeviceTensor* static_sdf_pointer{};
                if (static_field) {
                    static_sdf         = static_field->forward_sdf(samples.positions, samples.points.sample_count);
                    static_sdf_pointer = &static_sdf;
                }
                const PhysicsLoss result = physics.loss({.values = density.values, .derivatives = density.derivatives, .width = density.width, .sample_count = density.sample_count, .derivative_count = density.derivative_count}, {.values = velocity.values, .derivatives = velocity.derivatives, .width = velocity.width, .sample_count = velocity.sample_count, .derivative_count = velocity.derivative_count}, static_sdf_pointer, static_field ? static_field->evaluate_inverse_deviation() : nullptr, configuration.physics_weight, fade(state.step, configuration.velocity_delay_steps, 10'000u), configuration.neumann_weight);
                velocity_field.clear_gradients();
                velocity_field.backward(result.velocity_adjoints);
                velocity_field.step(learning_rate());
                physics_value = result.loss;
                neumann_value = result.neumann;
            }

            const bool perceptual   = configuration.perceptual_weight > 0.0F && (state.step + 1u) % 4u == 0u;
            const TrainingRays rays = scene.prepare_training_rays(frame, perceptual, state.seed, state.step);
            float perceptual_value{};
            if (perceptual) {
                for (std::uint32_t offset = 0u; offset < configuration.rays_per_step; offset += configuration.rays_per_batch) {
                    const std::uint32_t count = std::min(configuration.rays_per_batch, configuration.rays_per_step - offset);
                    const ForwardPass pass    = forward({.data = rays.rays.data + offset, .count = count}, rays.time, true, offset);
                    ::cuda::copy_bytes(stream, ::cuda::std::span<const Vector3<float>>{pass.output.rgb, count}, ::cuda::std::span<Vector3<float>>{fine_rendered.data() + offset, count});
                    ::cuda::copy_bytes(stream, ::cuda::std::span<const Vector3<float>>{pass.coarse_output.rgb, count}, ::cuda::std::span<Vector3<float>>{coarse_rendered.data() + offset, count});
                }
                const std::array<float, 4> layer_weights{
                    configuration.perceptual_weight * 0.25F * fade(state.step, 30'000u, 10'000u),
                    configuration.perceptual_weight * 0.25F * fade(state.step, 20'000u, 10'000u),
                    configuration.perceptual_weight * 0.25F * fade(state.step, 10'000u, 10'000u),
                    configuration.perceptual_weight * 0.25F * fade(state.step, 0u, 10'000u),
                };
                perceptual_value = perceptual_loss->loss_and_backward(fine_rendered.data(), coarse_rendered.data(), rays.target, layer_weights, fine_perceptual_adjoints.data(), coarse_perceptual_adjoints.data());
            } else {
                ::cuda::fill_bytes(stream, fine_perceptual_adjoints, 0u);
                ::cuda::fill_bytes(stream, coarse_perceptual_adjoints, 0u);
            }

            coarse_field.clear_gradients();
            dynamic_field.clear_gradients();
            if (static_field) static_field->clear_gradients();
            rendering.begin_step();
            const std::uint32_t normalization_sample_count = configuration.rays_per_step * (configuration.coarse_samples + configuration.importance_samples);
            for (std::uint32_t offset = 0u; offset < configuration.rays_per_step; offset += configuration.rays_per_batch) {
                const std::uint32_t count    = std::min(configuration.rays_per_batch, configuration.rays_per_step - offset);
                const ForwardPass pass       = forward({.data = rays.rays.data + offset, .count = count}, rays.time, true, offset);
                const FieldAdjoints adjoints = rendering.backward(pass.coarse_samples, {.values = pass.coarse_dynamic.values, .width = pass.coarse_dynamic.width, .sample_count = pass.coarse_dynamic.sample_count}, pass.fine_samples, {.values = pass.dynamic.values, .width = pass.dynamic.width, .sample_count = pass.dynamic.sample_count}, static_field ? &pass.static_output : nullptr, rays.target + offset, scene.background.data(), scene.bounds.data(),
                    {
                        .temporal_fading = fade(state.step, 0u, configuration.temporal_fading_steps),
                        .ghost           = configuration.ghost_weight * fade(state.step, 2'000u, 20'000u),
                        .ghost_scale     = configuration.ghost_scale,
                        .overlay         = configuration.overlay_weight * fade(state.step, 2'000u, 20'000u),
                        .eikonal         = configuration.eikonal_weight,
                        .deviation       = offset == 0u ? configuration.deviation_weight : 0.0F,
                        .step            = state.step,
                    },
                    fine_perceptual_adjoints.data() + offset, coarse_perceptual_adjoints.data() + offset, offset == 0u ? perceptual_value : 0.0F, configuration.rays_per_step, normalization_sample_count);
                if (static_field) static_field->backward(adjoints.static_color, adjoints.sdf, adjoints.sdf_gradient, adjoints.inverse_deviation);
                dynamic_field.backward(adjoints.dynamic);
                coarse_field.backward(adjoints.coarse);
            }
            const RenderingLossStatistics step_loss = rendering.end_step();
            const float rate                        = learning_rate();
            if (static_field) static_field->step(rate);
            dynamic_field.step(rate);
            coarse_field.step(rate);

            statistics.loss += step_loss.total + physics_value + neumann_value;
            statistics.image_loss += step_loss.image;
            statistics.coarse_image_loss += step_loss.coarse_image;
            statistics.perceptual_loss += step_loss.perceptual;
            statistics.ghost_loss += step_loss.ghost;
            statistics.overlay_loss += step_loss.overlay;
            statistics.eikonal_loss += step_loss.eikonal;
            statistics.deviation_loss += step_loss.deviation;
            statistics.physics_loss += physics_value;
            statistics.neumann_loss += neumann_value;
            statistics.psnr += -10.0F * std::log10(step_loss.image);
            ++state.step;
        }
        const float inverse_iterations = 1.0F / static_cast<float>(iterations);
        statistics.end_step            = state.step;
        statistics.loss *= inverse_iterations;
        statistics.image_loss *= inverse_iterations;
        statistics.coarse_image_loss *= inverse_iterations;
        statistics.perceptual_loss *= inverse_iterations;
        statistics.ghost_loss *= inverse_iterations;
        statistics.overlay_loss *= inverse_iterations;
        statistics.eikonal_loss *= inverse_iterations;
        statistics.deviation_loss *= inverse_iterations;
        statistics.physics_loss *= inverse_iterations;
        statistics.neumann_loss *= inverse_iterations;
        statistics.psnr *= inverse_iterations;
        statistics.elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
        return statistics;
    }

    RenderedFrame PINFS::render(const dataset::multiview::Frame& frame) {
        const std::uint32_t pixel_count = frame.extent.width * frame.extent.height;
        RenderedFrame result{.width = frame.extent.width, .height = frame.extent.height, .rgb = std::vector<Vector3<float>>(pixel_count)};
        for (std::uint32_t offset = 0u; offset < pixel_count; offset += configuration.rays_per_batch) {
            const std::uint32_t count = std::min(configuration.rays_per_batch, pixel_count - offset);
            const ForwardPass pass    = forward(scene.prepare_rendering_rays(frame, offset, count), frame.time, false, offset);
            ::cuda::copy_bytes(stream, ::cuda::std::span<const Vector3<float>>{pass.output.rgb, count}, ::cuda::std::span<Vector3<float>>{result.rgb.data() + offset, count});
        }
        stream.sync();
        return result;
    }

    EvaluationStats PINFS::evaluate(const std::string_view frame_set_name, const std::uint32_t maximum_frames, const std::uint32_t stride) {
        const auto start                              = std::chrono::steady_clock::now();
        const dataset::multiview::FrameSet& frame_set = find_frame_set(scene.dataset, frame_set_name);
        const std::uint32_t available_frames          = frame_set.time_count / stride;
        const std::uint32_t frame_count               = std::min(maximum_frames, available_frames);
        double normalized_squared_error{};
        double psnr{};
        std::uint64_t pixel_count{};
        for (const std::uint32_t frame_index : std::views::iota(0u, frame_count)) {
            const dataset::multiview::Frame& frame = frame_set.frames[frame_index * stride];
            const RenderedFrame rendered           = render(frame);
            pixel_count += static_cast<std::uint64_t>(frame.extent.width) * frame.extent.height;
            double frame_squared_error{};
            for (const std::size_t pixel : std::views::iota(0uz, rendered.rgb.size())) {
                for (const std::size_t component : std::views::iota(0uz, 3uz)) {
                    const std::uint8_t prediction = static_cast<std::uint8_t>(255.0F * std::clamp(rendered.rgb[pixel][component], 0.0F, 1.0F));
                    const std::uint8_t target     = frame.rgba[pixel * 4uz + component];
                    const double difference       = static_cast<double>(prediction) - static_cast<double>(target);
                    frame_squared_error += difference * difference;
                }
            }
            const double frame_mse = frame_squared_error / static_cast<double>(rendered.rgb.size() * 3uz);
            normalized_squared_error += frame_squared_error / (255.0 * 255.0);
            psnr += 10.0 * std::log10(255.0 * 255.0 / frame_mse);
        }
        const double mse = normalized_squared_error / static_cast<double>(pixel_count * 3u);
        return {
            .frame_set   = std::string{frame_set_name},
            .step        = state.step,
            .frame_count = frame_count,
            .pixel_count = pixel_count,
            .mse         = static_cast<float>(mse),
            .psnr        = static_cast<float>(psnr / static_cast<double>(frame_count)),
            .elapsed_ms  = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count(),
        };
    }

    VolumeSnapshot PINFS::sample_volume(const float time, const Vector3<std::uint32_t> resolution) {
        const std::size_t sample_count = static_cast<std::size_t>(resolution.x) * resolution.y * resolution.z;
        VolumeSnapshot result{.resolution = resolution, .density = std::vector<float>(sample_count), .velocity = std::vector<Vector3<float>>(sample_count)};
        ::cuda::device_buffer<float> density{stream, ::cuda::device_default_memory_pool(stream.device()), sample_count, ::cuda::no_init};
        ::cuda::device_buffer<Vector3<float>> velocity{stream, ::cuda::device_default_memory_pool(stream.device()), sample_count, ::cuda::no_init};
        for (std::size_t offset = 0uz; offset < sample_count; offset += configuration.inference_batch_size) {
            const std::uint32_t count       = static_cast<std::uint32_t>(std::min(static_cast<std::size_t>(configuration.inference_batch_size), sample_count - offset));
            const SpacetimePoint* points    = scene.prepare_volume_points(time, resolution, offset, count);
            const DeviceTensor density_out  = dynamic_field.forward({.values = reinterpret_cast<const float*>(points), .width = 4u, .sample_count = count});
            const DeviceTensor velocity_out = velocity_field.forward({.values = reinterpret_cast<const float*>(points), .width = 4u, .sample_count = count});
            dynamic_field.copy_density({.values = density_out.values, .width = density_out.width, .sample_count = density_out.sample_count}, density.data() + offset);
            ::cuda::copy_bytes(stream, ::cuda::std::span<const Vector3<float>>{reinterpret_cast<const Vector3<float>*>(velocity_out.values), count}, ::cuda::std::span<Vector3<float>>{velocity.data() + offset, count});
        }
        ::cuda::copy_bytes(stream, density, ::cuda::std::span<float>{result.density.data(), result.density.size()});
        ::cuda::copy_bytes(stream, velocity, ::cuda::std::span<Vector3<float>>{result.velocity.data(), result.velocity.size()});
        stream.sync();
        return result;
    }

    void PINFS::save(const std::filesystem::path& path) const {
        const NetworkState coarse                        = coarse_field.download();
        const NetworkState dynamic                       = dynamic_field.download();
        const NetworkState velocity                      = velocity_field.download();
        const std::optional<StaticFieldState> stationary = static_field ? std::optional{static_field->download()} : std::nullopt;
        const std::array<std::uint32_t, 2> training{state.seed, state.step};
        std::vector<serialization::safetensors::TensorView> tensors;
        const auto append_network = [&](const std::string_view prefix, const NetworkState& network) {
            tensors.push_back({.name = std::format("{}.parameters", prefix), .dtype = "F32", .shape = {network.parameters.size()}, .data = network.parameters.data(), .byte_count = network.parameters.size() * sizeof(float)});
            tensors.push_back({.name = std::format("{}.optimizer.first_moments", prefix), .dtype = "F32", .shape = {network.first_moments.size()}, .data = network.first_moments.data(), .byte_count = network.first_moments.size() * sizeof(float)});
            tensors.push_back({.name = std::format("{}.optimizer.second_moments", prefix), .dtype = "F32", .shape = {network.second_moments.size()}, .data = network.second_moments.data(), .byte_count = network.second_moments.size() * sizeof(float)});
            tensors.push_back({.name = std::format("{}.optimizer.layer_steps", prefix), .dtype = "U32", .shape = {network.layer_steps.size()}, .data = network.layer_steps.data(), .byte_count = network.layer_steps.size() * sizeof(std::uint32_t)});
        };
        append_network("field.coarse", coarse);
        append_network("field.dynamic", dynamic);
        append_network("field.velocity", velocity);
        if (stationary) {
            append_network("field.static.sdf", stationary->sdf);
            append_network("field.static.color", stationary->color);
            tensors.push_back({.name = "field.static.deviation", .dtype = "F32", .shape = {1u}, .data = &stationary->deviation, .byte_count = sizeof(float)});
            tensors.push_back({.name = "field.static.deviation.optimizer.first_moment", .dtype = "F32", .shape = {1u}, .data = &stationary->deviation_first_moment, .byte_count = sizeof(float)});
            tensors.push_back({.name = "field.static.deviation.optimizer.second_moment", .dtype = "F32", .shape = {1u}, .data = &stationary->deviation_second_moment, .byte_count = sizeof(float)});
            tensors.push_back({.name = "field.static.deviation.optimizer.step", .dtype = "U32", .shape = {1u}, .data = &stationary->deviation_step, .byte_count = sizeof(std::uint32_t)});
        }
        tensors.push_back({.name = "training.state", .dtype = "U32", .shape = {training.size()}, .data = training.data(), .byte_count = training.size() * sizeof(std::uint32_t)});
        serialization::safetensors::write(path, "pinfs", tensors);
    }

    void PINFS::load(const std::filesystem::path& path) {
        const serialization::safetensors::File file = serialization::safetensors::read(path);
        const auto read                             = [&](const std::string_view name, auto& destination) {
            const serialization::safetensors::Tensor& tensor = *std::ranges::find(file.tensors, name, &serialization::safetensors::Tensor::name);
            destination.resize(tensor.data.size() / sizeof(typename std::remove_reference_t<decltype(destination)>::value_type));
            std::memcpy(destination.data(), tensor.data.data(), tensor.data.size());
        };
        const auto read_network = [&](const std::string_view prefix) {
            NetworkState result;
            read(std::format("{}.parameters", prefix), result.parameters);
            read(std::format("{}.optimizer.first_moments", prefix), result.first_moments);
            read(std::format("{}.optimizer.second_moments", prefix), result.second_moments);
            read(std::format("{}.optimizer.layer_steps", prefix), result.layer_steps);
            return result;
        };
        coarse_field.upload(read_network("field.coarse"));
        dynamic_field.upload(read_network("field.dynamic"));
        velocity_field.upload(read_network("field.velocity"));
        if (static_field) {
            StaticFieldState stationary{.sdf = read_network("field.static.sdf"), .color = read_network("field.static.color")};
            std::vector<float> deviation;
            std::vector<float> deviation_first_moment;
            std::vector<float> deviation_second_moment;
            std::vector<std::uint32_t> deviation_step;
            read("field.static.deviation", deviation);
            read("field.static.deviation.optimizer.first_moment", deviation_first_moment);
            read("field.static.deviation.optimizer.second_moment", deviation_second_moment);
            read("field.static.deviation.optimizer.step", deviation_step);
            stationary.deviation               = deviation[0];
            stationary.deviation_first_moment  = deviation_first_moment[0];
            stationary.deviation_second_moment = deviation_second_moment[0];
            stationary.deviation_step          = deviation_step[0];
            static_field->upload(stationary);
        }
        std::vector<std::uint32_t> training;
        read("training.state", training);
        state = {.seed = training[0], .step = training[1]};
        coarse_field.set_fading_step(state.step, configuration.layer_fading_steps);
        dynamic_field.set_fading_step(state.step, configuration.layer_fading_steps);
        velocity_field.set_fading_step(state.step - std::min(state.step, configuration.velocity_delay_steps), configuration.layer_fading_steps);
        if (static_field) static_field->set_fading_step(std::max(1u, state.step), configuration.layer_fading_steps);
    }

    float PINFS::learning_rate() const {
        const std::uint32_t optimizer_step = state.step == 0u ? 0u : state.step - 1u;
        return configuration.learning_rate * std::pow(0.1F, static_cast<float>(optimizer_step) / static_cast<float>(configuration.learning_rate_decay_thousands * 1'000u));
    }

    PINFS::ForwardPass PINFS::forward(const DeviceRays rays, const float time, const bool perturb, const std::uint32_t ray_offset) {
        const float near_distance = configuration.near_distance.value_or(scene.dataset.near_distance);
        const float far_distance  = configuration.far_distance.value_or(scene.dataset.far_distance);
        ForwardPass pass;
        pass.coarse_samples = sampling.coarse(rays, configuration.coarse_samples, near_distance, far_distance, time, state.seed, state.step, ray_offset, perturb);
        if (perturb && configuration.train_warp && state.step >= configuration.velocity_delay_steps) {
            const DeviceTensor velocity = velocity_field.forward({.values = reinterpret_cast<const float*>(pass.coarse_samples.dynamic_points), .width = 4u, .sample_count = pass.coarse_samples.sample_count()});
            sampling.warp(pass.coarse_samples, {.values = velocity.values, .width = velocity.width, .sample_count = velocity.sample_count}, fade(state.step, configuration.velocity_delay_steps + 10'000u, 20'000u) / static_cast<float>(scene.training.time_count), state.seed, state.step, ray_offset);
        }
        pass.coarse_dynamic = coarse_field.forward({.values = reinterpret_cast<const float*>(pass.coarse_samples.dynamic_points), .width = 4u, .sample_count = pass.coarse_samples.sample_count()});
        pass.coarse_output  = rendering.coarse(pass.coarse_samples, {.values = pass.coarse_dynamic.values, .width = pass.coarse_dynamic.width, .sample_count = pass.coarse_dynamic.sample_count}, scene.background.data(), scene.bounds.data());

        const std::uint32_t static_sample_count = static_field ? configuration.importance_samples / 2u : 0u;
        const std::uint32_t pdf_sample_count    = configuration.importance_samples - static_sample_count;
        sampling.sample_pdf(pass.coarse_samples, pass.coarse_output.weights, pdf_sample_count, state.seed, state.step, ray_offset, !perturb);
        if (static_field) {
            DeviceTensor sdf = static_field->infer_sdf(pass.coarse_samples.positions, pass.coarse_samples.sample_count());
            sampling.begin_neus_upsampling(pass.coarse_samples, {.values = sdf.values, .width = sdf.width, .sample_count = sdf.sample_count});
            const std::uint32_t samples_per_stage = static_sample_count / 4u;
            for (const std::uint32_t stage : std::views::iota(0u, 4u)) {
                const bool last            = stage == 3u;
                const RaySamples positions = sampling.neus_upsampling_positions(rays, samples_per_stage, 64.0F * static_cast<float>(1u << stage));
                if (!last) sdf = static_field->infer_sdf(positions.positions, positions.sample_count());
                sampling.commit_neus_upsampling({.values = sdf.values, .width = sdf.width, .sample_count = sdf.sample_count}, last);
            }
        }
        pass.fine_samples = sampling.fine(rays, pass.coarse_samples, pdf_sample_count, static_sample_count, time);
        if (perturb && configuration.train_warp && state.step >= configuration.velocity_delay_steps) {
            const DeviceTensor velocity = velocity_field.forward({.values = reinterpret_cast<const float*>(pass.fine_samples.dynamic_points), .width = 4u, .sample_count = pass.fine_samples.sample_count()});
            sampling.warp(pass.fine_samples, {.values = velocity.values, .width = velocity.width, .sample_count = velocity.sample_count}, fade(state.step, configuration.velocity_delay_steps + 10'000u, 20'000u) / static_cast<float>(scene.training.time_count), state.seed, state.step, ray_offset);
        }
        pass.dynamic = dynamic_field.forward({.values = reinterpret_cast<const float*>(pass.fine_samples.dynamic_points), .width = 4u, .sample_count = pass.fine_samples.sample_count()});
        if (static_field) pass.static_output = static_field->forward(pass.fine_samples.positions, pass.fine_samples.directions, pass.fine_samples.sample_count());
        pass.output = rendering.fine(pass.fine_samples, {.values = pass.dynamic.values, .width = pass.dynamic.width, .sample_count = pass.dynamic.sample_count}, static_field ? &pass.static_output : nullptr, scene.background.data(), scene.bounds.data());
        return pass;
    }
} // namespace physica::reconstruction::pinfs
