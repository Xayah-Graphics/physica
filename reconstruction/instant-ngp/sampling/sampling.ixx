module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>
#include <cuda/stream>

export module physica.reconstruction.instant_ngp.sampling;

import physica.reconstruction.instant_ngp.network;
import physica.reconstruction.instant_ngp.scene;
import std;

export namespace physica::reconstruction::instant_ngp {
struct SamplingShape final {
    std::uint32_t occupancy_grid_size;
    std::uint32_t ray_step_count;
    std::uint32_t training_batch_granularity;
    std::uint32_t initial_rays_per_batch;
    std::uint32_t evaluation_tile_rays;
    std::uint32_t density_grid_warmup_steps;
    std::uint32_t density_grid_skip_interval;
    std::uint32_t density_grid_max_skip;
    std::uint32_t density_grid_warmup_samples;
    std::uint32_t density_grid_uniform_samples;
    std::uint32_t density_grid_nonuniform_samples;
    bool snap_to_pixel_centers;

    constexpr bool operator==(const SamplingShape&) const = default;
};

inline constexpr SamplingShape nerf_synthetic_sampling_shape{
    .occupancy_grid_size = 128u,
    .ray_step_count = 1024u,
    .training_batch_granularity = 16u * 8u,
    .initial_rays_per_batch = 1u << 12u,
    .evaluation_tile_rays = 4096u,
    .density_grid_warmup_steps = 256u,
    .density_grid_skip_interval = 16u,
    .density_grid_max_skip = 16u,
    .density_grid_warmup_samples = 128u * 128u * 128u,
    .density_grid_uniform_samples = 128u * 128u * 128u / 4u,
    .density_grid_nonuniform_samples = 128u * 128u * 128u / 4u,
    .snap_to_pixel_centers = true,
};

struct Ray final {
    std::array<float, 3> origin{};
};

static_assert(sizeof(Ray) == 3uz * sizeof(float));

struct TrainingSamples final {
    DeviceSamples samples;
    const Ray* rays = nullptr;
    const std::uint32_t* target_pixel_indices = nullptr;
    std::uint32_t* numsteps = nullptr;
    const std::uint32_t* ray_counter = nullptr;
    const std::uint32_t* sample_counter = nullptr;
    std::uint32_t rays_per_batch = 0u;
};

struct EvaluationSamples final {
    DeviceSamples samples;
    const std::uint32_t* numsteps = nullptr;
    std::uint32_t image_index = 0u;
    std::uint32_t pixel_offset = 0u;
    std::uint32_t pixel_count = 0u;
};

struct SamplingState final {
    std::vector<float> density;
    std::uint32_t density_grid_ema_step = 0u;
    std::uint32_t rays_per_batch = 0u;
    std::uint32_t inference_sample_count = 0u;
};

template <SamplingShape Shape, NetworkShape NetworkSpec>
struct Sampling final {
    static_assert(Shape.occupancy_grid_size == 128u, "Only a 128^3 occupancy grid is implemented.");
    static_assert(Shape.ray_step_count == 1024u, "Only 1024 ray-marching steps are implemented.");
    static_assert(Shape.training_batch_granularity == 16u * 8u, "Only the current WMMA batch granularity is implemented.");
    static_assert(Shape.initial_rays_per_batch == 1u << 12u, "Only the current initial ray batch is implemented.");
    static_assert(Shape.evaluation_tile_rays == 4096u, "Only 4096-ray evaluation tiles are implemented.");
    static_assert(Shape.density_grid_warmup_steps == 256u, "Only the current density-grid warmup is implemented.");
    static_assert(Shape.density_grid_skip_interval == 16u, "Only the current density-grid skip interval is implemented.");
    static_assert(Shape.density_grid_max_skip == 16u, "Only the current density-grid maximum skip is implemented.");
    static_assert(Shape.density_grid_warmup_samples == 128u * 128u * 128u, "Only the current density-grid warmup sample count is implemented.");
    static_assert(Shape.density_grid_uniform_samples == 128u * 128u * 128u / 4u, "Only the current uniform density sample count is implemented.");
    static_assert(Shape.density_grid_nonuniform_samples == 128u * 128u * 128u / 4u, "Only the current nonuniform density sample count is implemented.");
    static_assert(Shape.snap_to_pixel_centers, "Only pixel-center ray sampling is implemented.");
    static_assert(NetworkSpec.training_batch_size == 1u << 18u, "Only a 2^18 training batch is implemented.");
    static_assert(NetworkSpec.inference_capacity == (1u << 18u) * 16u, "Only the current sample capacity is implemented.");
    static_assert(sizeof(Sample) == 7uz * sizeof(float));

    inline static constexpr std::uint32_t grid_cell_count = Shape.occupancy_grid_size * Shape.occupancy_grid_size * Shape.occupancy_grid_size;

    explicit Sampling(::cuda::stream_ref stream);

    DeviceSamples prepare_density_update(const DeviceFrameSet& frame_set, std::uint32_t seed, std::uint32_t step);
    void accumulate_density_update(std::uint32_t offset, NetworkOutput output);
    void commit_density_update();
    TrainingSamples sample_training(const DeviceFrameSet& frame_set, std::uint32_t seed, std::uint32_t step);
    EvaluationSamples sample_evaluation(const DeviceFrameSet& frame_set, std::uint32_t image_index, std::uint32_t pixel_offset, std::uint32_t pixel_count);
    std::uint32_t adapt(std::uint32_t generated_sample_count, std::uint32_t requested_compacted_sample_count);
    SamplingState download() const;
    void upload(const SamplingState& state);

private:
    ::cuda::stream_ref stream;
    ::cuda::device_buffer<Sample> samples;
    ::cuda::device_buffer<Ray> rays;
    ::cuda::device_buffer<std::uint32_t> target_pixel_indices;
    ::cuda::device_buffer<std::uint32_t> numsteps;
    ::cuda::device_buffer<std::uint32_t> ray_counter;
    ::cuda::device_buffer<std::uint32_t> sample_counter;
    ::cuda::device_buffer<std::uint8_t> occupancy;
    ::cuda::device_buffer<float> density;
    ::cuda::device_buffer<float> density_scratch;
    ::cuda::device_buffer<std::uint32_t> density_indices;
    ::cuda::device_buffer<float> density_mean;
    ::cuda::device_buffer<std::uint32_t> occupied_cell_counter;
    ::cuda::device_buffer<std::uint32_t> evaluation_numsteps;
    ::cuda::device_buffer<std::uint32_t> evaluation_sample_counter;
    std::uint32_t density_grid_ema_step = 0u;
    ::cuda::host_buffer<std::uint32_t> occupied_cell_count;
    std::uint32_t rays_per_batch = Shape.initial_rays_per_batch;
    std::uint32_t inference_sample_count = NetworkSpec.inference_capacity;
};

extern template struct Sampling<nerf_synthetic_sampling_shape, nerf_synthetic_network_shape>;
} // namespace physica::reconstruction::instant_ngp
