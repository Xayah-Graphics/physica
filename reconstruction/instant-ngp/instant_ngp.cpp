module;

#include <nlohmann/json.hpp>
#include <physica/cuda.h>

module physica.reconstruction.instant_ngp;

import std;
import physica.reconstruction.dataset.multiview;
import physica.reconstruction.instant_ngp.scene;
import physica.reconstruction.instant_ngp.network;
import physica.reconstruction.instant_ngp.sampling;
import physica.reconstruction.instant_ngp.rendering;

namespace physica::reconstruction::instant_ngp {
template <NetworkShape NetworkSpec, SamplingShape SamplingSpec, RenderingShape RenderingSpec>
InstantNGP<NetworkSpec, SamplingSpec, RenderingSpec>::InstantNGP(const dataset::multiview::Dataset& dataset, const std::uint32_t device_ordinal, const std::uint32_t training_frame_set, const float scene_scale, const std::uint32_t seed) : state{.seed = seed, .training_frame_set = training_frame_set}, stream{::cuda::devices[device_ordinal]}, scene{dataset, scene_scale, stream}, network{stream, seed}, sampling{stream}, rendering{stream} {}

template <NetworkShape NetworkSpec, SamplingShape SamplingSpec, RenderingShape RenderingSpec>
InstantNGP<NetworkSpec, SamplingSpec, RenderingSpec>::~InstantNGP() noexcept = default;

template <NetworkShape NetworkSpec, SamplingShape SamplingSpec, RenderingShape RenderingSpec>
OptimizationStats InstantNGP<NetworkSpec, SamplingSpec, RenderingSpec>::optimize(const std::uint32_t iterations) {
    const auto start = std::chrono::steady_clock::now();
    const std::uint32_t begin_step = state.step;
    TrainingBatch batch{};
    DeviceFrameSet& frame_set = scene.frame_sets[state.training_frame_set];
    std::uint32_t occupied_cell_count = 0u;

    for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        const DeviceSamples density_samples = sampling.prepare_density_update(frame_set, state.seed, state.step);
        if (density_samples.count != 0u) {
            for (std::uint32_t offset = 0u; offset < density_samples.count; offset += NetworkSpec.training_batch_size) {
                const std::uint32_t count = (std::min)(NetworkSpec.training_batch_size, density_samples.count - offset);
                const DeviceSamples samples{.data = density_samples.data + offset, .count = count};
                sampling.accumulate_density_update(offset, network.infer_density(samples));
            }
            sampling.commit_density_update();
        }
        const TrainingSamples samples = sampling.sample_training(frame_set, state.seed, state.step);
        const NetworkOutput output = network.infer(samples.samples);
        batch = rendering.loss_and_compact(samples, output, frame_set, state.seed, state.step);
        network.forward(batch.samples);
        network.backward(batch.samples, batch.gradients);
        network.step();
        occupied_cell_count = sampling.adapt(batch.generated_sample_count, batch.requested_compacted_sample_count);
        ++state.step;
    }

    return {
        .begin_step = begin_step,
        .end_step = state.step,
        .rays = batch.ray_count,
        .generated_samples = batch.generated_sample_count,
        .compacted_samples = batch.compacted_sample_count,
        .occupied_cells = occupied_cell_count,
        .loss = static_cast<float>(batch.loss_sum * static_cast<double>(batch.compacted_sample_count) / static_cast<double>(NetworkSpec.training_batch_size)),
        .sample_efficiency = static_cast<float>(batch.compacted_sample_count) / static_cast<float>(batch.generated_sample_count),
        .occupancy_ratio = static_cast<float>(occupied_cell_count) / static_cast<float>(SamplingSpec.occupancy_grid_size * SamplingSpec.occupancy_grid_size * SamplingSpec.occupancy_grid_size),
        .elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count(),
    };
}

template <NetworkShape NetworkSpec, SamplingShape SamplingSpec, RenderingShape RenderingSpec>
EvaluationStats InstantNGP<NetworkSpec, SamplingSpec, RenderingSpec>::evaluate(const std::uint32_t frame_set_index) {
    const auto start = std::chrono::steady_clock::now();
    DeviceFrameSet& frame_set = scene.frame_sets[frame_set_index];
    const std::uint64_t pixel_count = static_cast<std::uint64_t>(frame_set.extent.width) * frame_set.extent.height * frame_set.frame_count;
    const std::uint32_t pixels_per_frame = frame_set.extent.width * frame_set.extent.height;
    rendering.begin_evaluation();
    for (std::uint32_t image_index = 0u; image_index < frame_set.frame_count; ++image_index) {
        for (std::uint32_t pixel_offset = 0u; pixel_offset < pixels_per_frame; pixel_offset += SamplingSpec.evaluation_tile_rays) {
            const std::uint32_t tile_pixel_count = (std::min)(SamplingSpec.evaluation_tile_rays, pixels_per_frame - pixel_offset);
            const EvaluationSamples samples = sampling.sample_evaluation(frame_set, image_index, pixel_offset, tile_pixel_count);
            rendering.accumulate_evaluation(samples, network.infer(samples.samples), frame_set);
        }
    }
    const double squared_error = rendering.end_evaluation();
    const double mse = squared_error / (static_cast<double>(pixel_count) * 3.0);
    return {
        .frame_set = frame_set.name,
        .step = state.step,
        .frame_count = frame_set.frame_count,
        .pixel_count = pixel_count,
        .mse = static_cast<float>(mse),
        .psnr = static_cast<float>(-10.0 * std::log10(mse)),
        .elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count(),
    };
}

template <NetworkShape NetworkSpec, SamplingShape SamplingSpec, RenderingShape RenderingSpec>
void InstantNGP<NetworkSpec, SamplingSpec, RenderingSpec>::save(const std::filesystem::path& path) const {
    const NetworkState network_state = network.download();
    const SamplingState sampling_state = sampling.download();
    const std::array<std::uint32_t, 2> training_metadata{state.seed, state.step};
    const std::array<std::uint32_t, 4> sampling_metadata{state.training_frame_set, sampling_state.density_grid_ema_step, sampling_state.rays_per_batch, sampling_state.inference_sample_count};

    struct Tensor final {
        std::string_view name;
        std::string_view dtype;
        std::uint64_t element_count;
        std::uint64_t byte_count;
        const void* data;
    };

    const std::array tensors{
        Tensor{"parameters", "F32", network_state.parameters.size(), network_state.parameters.size() * sizeof(float), network_state.parameters.data()},
        Tensor{"first_moments", "F32", network_state.first_moments.size(), network_state.first_moments.size() * sizeof(float), network_state.first_moments.data()},
        Tensor{"second_moments", "F32", network_state.second_moments.size(), network_state.second_moments.size() * sizeof(float), network_state.second_moments.data()},
        Tensor{"parameter_steps", "U32", network_state.parameter_steps.size(), network_state.parameter_steps.size() * sizeof(std::uint32_t), network_state.parameter_steps.data()},
        Tensor{"density", "F32", sampling_state.density.size(), sampling_state.density.size() * sizeof(float), sampling_state.density.data()},
        Tensor{"training_state", "U32", training_metadata.size(), training_metadata.size() * sizeof(std::uint32_t), training_metadata.data()},
        Tensor{"sampling_state", "U32", sampling_metadata.size(), sampling_metadata.size() * sizeof(std::uint32_t), sampling_metadata.data()},
    };

    nlohmann::json header;
    std::uint64_t offset = 0u;
    for (const Tensor& tensor : tensors) {
        header[std::string{tensor.name}] = {
            {"dtype", tensor.dtype},
            {"shape", nlohmann::json::array({tensor.element_count})},
            {"data_offsets", nlohmann::json::array({offset, offset + tensor.byte_count})},
        };
        offset += tensor.byte_count;
    }

    std::string header_text = header.dump();
    header_text.append((8uz - header_text.size() % 8uz) % 8uz, ' ');
    const std::uint64_t header_size = header_text.size();
    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const Tensor& tensor : tensors) output.write(static_cast<const char*>(tensor.data), static_cast<std::streamsize>(tensor.byte_count));
}

template <NetworkShape NetworkSpec, SamplingShape SamplingSpec, RenderingShape RenderingSpec>
void InstantNGP<NetworkSpec, SamplingSpec, RenderingSpec>::load(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    std::uint64_t header_size = 0u;
    input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    std::string header_text(header_size, '\0');
    input.read(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    const nlohmann::json header = nlohmann::json::parse(header_text);
    const std::uint64_t data_offset = sizeof(header_size) + header_size;

    const auto read_tensor = [&](const std::string_view name, auto& destination) {
        const nlohmann::json& tensor = header.at(name);
        const std::uint64_t begin = tensor.at("data_offsets").at(0).get<std::uint64_t>();
        const std::uint64_t end = tensor.at("data_offsets").at(1).get<std::uint64_t>();
        destination.resize((end - begin) / sizeof(typename std::remove_reference_t<decltype(destination)>::value_type));
        input.seekg(static_cast<std::streamoff>(data_offset + begin));
        input.read(reinterpret_cast<char*>(destination.data()), static_cast<std::streamsize>(end - begin));
    };

    NetworkState network_state;
    SamplingState sampling_state;
    std::vector<std::uint32_t> training_metadata;
    std::vector<std::uint32_t> sampling_metadata;
    read_tensor("parameters", network_state.parameters);
    read_tensor("first_moments", network_state.first_moments);
    read_tensor("second_moments", network_state.second_moments);
    read_tensor("parameter_steps", network_state.parameter_steps);
    read_tensor("density", sampling_state.density);
    read_tensor("training_state", training_metadata);
    read_tensor("sampling_state", sampling_metadata);
    sampling_state.density_grid_ema_step = sampling_metadata[1];
    sampling_state.rays_per_batch = sampling_metadata[2];
    sampling_state.inference_sample_count = sampling_metadata[3];

    network.upload(network_state);
    sampling.upload(sampling_state);
    state = {
        .seed = training_metadata[0],
        .step = training_metadata[1],
        .training_frame_set = sampling_metadata[0],
    };
}

template <NetworkShape NetworkSpec, SamplingShape SamplingSpec, RenderingShape RenderingSpec>
InstantNGPDeviceState InstantNGP<NetworkSpec, SamplingSpec, RenderingSpec>::device_state() const noexcept {
    return {.stream = stream.get(), .network = network.device_state(), .sampling = sampling.device_state()};
}

template struct InstantNGP<nerf_synthetic_network_shape, nerf_synthetic_sampling_shape, nerf_synthetic_rendering_shape>;
} // namespace physica::reconstruction::instant_ngp
