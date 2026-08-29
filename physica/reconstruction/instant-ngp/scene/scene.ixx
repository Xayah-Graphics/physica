module;

#include <physica/cuda.h>

export module physica.reconstruction.instant_ngp.scene;

import std;
import physica.reconstruction.dataset.multiview;
export import physica.math;

export namespace physica::reconstruction::instant_ngp {
    struct DeviceCamera final {
        Vector3<float> x{};
        Vector3<float> y{};
        Vector3<float> z{};
        Vector3<float> origin{};
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
