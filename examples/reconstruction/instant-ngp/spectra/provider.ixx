module;

#include <cuda_runtime_api.h>
#include <spectra/sdk/cuda_types.h>
#include <physica/cuda.h>

export module physica.example.reconstruction.instant_ngp.spectra;

import std;
import physica.reconstruction.dataset.nerf_synthetic;
import physica.reconstruction.instant_ngp;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::instant_ngp {
struct Settings final {};

struct Provider final {
    [[no_unique_address]] Settings settings;

    static constexpr auto description = spectra::sdk::describe(
        "physica.example.reconstruction.instant-ngp",
        spectra::sdk::cameras<"training">(),
        spectra::sdk::hash_grid_radiance_field<"field">(),
        spectra::sdk::metric<"step", std::uint32_t>("Step", {}, "Training"),
        spectra::sdk::metric<"loss", float>("Loss", {}, "Training", true),
        spectra::sdk::metric<"psnr", float>("PSNR", "dB", "Training", true),
        spectra::sdk::metric<"sample-efficiency", float>("Sample efficiency", {}, "Training", true),
        spectra::sdk::metric<"occupancy", float>("Occupancy", {}, "Training", true)
    );

    Provider(Settings settings, const std::filesystem::path& assets);
    ~Provider() noexcept;
    Provider(const Provider&) = delete;
    Provider& operator=(const Provider&) = delete;

    void setup(spectra::sdk::cuda::Setup& setup);
    void reset(std::uint64_t seed);
    void step(double seconds);
    void publish(spectra::sdk::cuda::Output& output);

private:
    inline static constexpr float scene_scale = 0.33F;

    reconstruction::dataset::multiview::Dataset dataset;
    reconstruction::instant_ngp::InstantNGP<
        reconstruction::instant_ngp::nerf_synthetic_network_shape,
        reconstruction::instant_ngp::nerf_synthetic_sampling_shape,
        reconstruction::instant_ngp::nerf_synthetic_rendering_shape
    >* instant_ngp{};
    reconstruction::instant_ngp::OptimizationStats training{};
    float psnr = std::numeric_limits<float>::quiet_NaN();
};

Provider::Provider(const Settings source, const std::filesystem::path& assets) : settings(source), dataset(reconstruction::dataset::nerf_synthetic::load(assets / "../../../data/nerf-synthetic/lego", {.frame_sets = {"train"}})) {}

Provider::~Provider() noexcept {
    delete instant_ngp;
}

void Provider::setup(spectra::sdk::cuda::Setup& setup) {
    const reconstruction::dataset::multiview::FrameSet& training_frames = dataset.frame_sets[0];
    const reconstruction::dataset::multiview::Frame& first_frame = training_frames.frames.front();
    std::vector<spectra::sdk::Camera> cameras;
    cameras.reserve(training_frames.frames.size());
    for (const reconstruction::dataset::multiview::Frame& frame : training_frames.frames) {
        const std::array<float, 16>& transform = frame.world_from_camera;
        cameras.push_back({
            .right = {transform[4], transform[8], transform[0]},
            .down = {-transform[5], -transform[9], -transform[1]},
            .forward = {-transform[6], -transform[10], -transform[2]},
            .position = {transform[7] * scene_scale + 0.5F, transform[11] * scene_scale + 0.5F, transform[3] * scene_scale + 0.5F},
            .focal = {frame.intrinsics.focal_x, frame.intrinsics.focal_y},
            .principal = {frame.intrinsics.principal_x, frame.intrinsics.principal_y},
        });
    }

    const spectra::sdk::cuda::CamerasSetup camera_output = setup.cameras<"training">(cameras, first_frame.extent.width, first_frame.extent.height);
    std::size_t pixel_offset = 0uz;
    for (const reconstruction::dataset::multiview::Frame& frame : training_frames.frames) {
        if (cudaMemcpy(camera_output.images.data() + pixel_offset, frame.rgba.data(), frame.rgba.size(), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("CUDA training-image upload failed");
        pixel_offset += frame.rgba.size() / sizeof(spectra::sdk::Rgba8);
    }
    setup.hash_grid_radiance_field<"field">();
}

void Provider::reset(const std::uint64_t seed) {
    auto* replacement = new reconstruction::instant_ngp::InstantNGP<
        reconstruction::instant_ngp::nerf_synthetic_network_shape,
        reconstruction::instant_ngp::nerf_synthetic_sampling_shape,
        reconstruction::instant_ngp::nerf_synthetic_rendering_shape
    >(dataset, 0u, 0u, scene_scale, static_cast<std::uint32_t>(seed));
    delete instant_ngp;
    instant_ngp = replacement;
    training = {};
    psnr = std::numeric_limits<float>::quiet_NaN();
}

void Provider::step(double) {
    training = instant_ngp->optimize(1u);
    psnr = -10.0F * std::log10(training.loss);
}

void Provider::publish(spectra::sdk::cuda::Output& output) {
    const reconstruction::instant_ngp::InstantNGPDeviceState state = instant_ngp->device_state();
    spectra::sdk::cuda::Frame frame = output.begin(state.stream);
    const spectra::sdk::cuda::HashGridRadianceField field = frame.hash_grid_radiance_field<"field">();
    const cudaStream_t stream = static_cast<cudaStream_t>(state.stream);
    const auto copy = [stream](const auto destination, const void* source) {
        if (cudaMemcpyAsync(destination.data(), source, destination.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("CUDA neural-field publish failed");
    };

    copy(field.density_input, state.network.density_input.data());
    copy(field.density_output, state.network.density_output.data());
    copy(field.rgb_input, state.network.color_input.data());
    copy(field.rgb_hidden, state.network.color_hidden.data());
    copy(field.rgb_output, state.network.color_output.data());
    copy(field.hash_grid, state.network.hash_grid.data());
    copy(field.occupancy, state.sampling.occupancy.data());

    frame.metric<"step">().upload(training.end_step);
    frame.metric<"loss">().upload(training.loss);
    frame.metric<"psnr">().upload(psnr);
    frame.metric<"sample-efficiency">().upload(training.sample_efficiency);
    frame.metric<"occupancy">().upload(training.occupancy_ratio);
    frame.commit();
}
} // namespace physica::examples::instant_ngp
