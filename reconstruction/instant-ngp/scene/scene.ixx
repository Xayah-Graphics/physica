module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>
#include <cuda/stream>

export module physica.reconstruction.instant_ngp.scene;

import physica.reconstruction.dataset.multiview;
import std;

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
} // namespace physica::reconstruction::instant_ngp
