module;

#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>

export module physica.reconstruction.instant_ngp.scene;

import std;
import physica.reconstruction.dataset.multiview;

export namespace physica::reconstruction::instant_ngp {
struct DeviceCamera final {
    std::array<float, 3> x{};
    std::array<float, 3> y{};
    std::array<float, 3> z{};
    std::array<float, 3> origin{};
};

static_assert(sizeof(DeviceCamera) == 12uz * sizeof(float));

struct DeviceFrameSet final {
    std::string name;
    dataset::multiview::Extent extent;
    dataset::multiview::Intrinsics intrinsics;
    std::uint32_t frame_count = 0u;
    ::cuda::device_buffer<std::uint8_t> pixels;
    ::cuda::device_buffer<DeviceCamera> cameras;
};

struct Scene final {
    std::vector<DeviceFrameSet> frame_sets;

    Scene(const dataset::multiview::Dataset& dataset, float scene_scale, ::cuda::stream_ref stream);
};

Scene::Scene(const dataset::multiview::Dataset& dataset, const float scale, const ::cuda::stream_ref stream) {
    constexpr float scene_offset = 0.5F;
    frame_sets.reserve(dataset.frame_sets.size());
    for (const dataset::multiview::FrameSet& frame_set : dataset.frame_sets) {
        const dataset::multiview::Frame& first = frame_set.frames.front();
        const std::size_t pixel_count = first.rgba.size() * frame_set.frames.size();
        ::cuda::host_buffer<std::uint8_t> pixels{stream, ::cuda::pinned_default_memory_pool(), pixel_count, ::cuda::no_init};
        ::cuda::host_buffer<DeviceCamera> cameras{stream, ::cuda::pinned_default_memory_pool(), frame_set.frames.size(), ::cuda::no_init};

        std::size_t frame_index = 0uz;
        for (const dataset::multiview::Frame& frame : frame_set.frames) {
            std::ranges::copy(frame.rgba, pixels.data() + frame_index * first.rgba.size());
            const std::array<float, 16>& transform = frame.world_from_camera;
            cameras.data()[frame_index] = {
                .x = {transform[4], transform[8], transform[0]},
                .y = {-transform[5], -transform[9], -transform[1]},
                .z = {-transform[6], -transform[10], -transform[2]},
                .origin = {transform[7] * scale + scene_offset, transform[11] * scale + scene_offset, transform[3] * scale + scene_offset},
            };
            ++frame_index;
        }

        DeviceFrameSet device_frame_set{
            .name = frame_set.name,
            .extent = first.extent,
            .intrinsics = first.intrinsics,
            .frame_count = static_cast<std::uint32_t>(frame_set.frames.size()),
            .pixels = ::cuda::device_buffer<std::uint8_t>{stream, ::cuda::device_default_memory_pool(stream.device()), pixels.size(), ::cuda::no_init},
            .cameras = ::cuda::device_buffer<DeviceCamera>{stream, ::cuda::device_default_memory_pool(stream.device()), cameras.size(), ::cuda::no_init},
        };

        ::cuda::copy_bytes(stream, pixels, device_frame_set.pixels);
        ::cuda::copy_bytes(stream, cameras, device_frame_set.cameras);
        stream.sync();
        frame_sets.push_back(std::move(device_frame_set));
    }
}
} // namespace physica::reconstruction::instant_ngp
