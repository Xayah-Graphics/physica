module;

#include <physica/cuda.h>

module physica.reconstruction.instant_ngp.scene;

import std;

namespace physica::reconstruction::instant_ngp {
    Scene::Scene(const dataset::multiview::Dataset& dataset, const float scale, const ::cuda::stream_ref stream) {
        constexpr float scene_offset = 0.5F;
        frame_sets.reserve(dataset.frame_sets.size());
        for (const dataset::multiview::FrameSet& frame_set : dataset.frame_sets) {
            const dataset::multiview::Frame& first = frame_set.frames.front();
            const std::size_t pixel_count          = first.rgba.size() * frame_set.frames.size();
            ::cuda::host_buffer<std::uint8_t> pixels{stream, ::cuda::pinned_default_memory_pool(), pixel_count, ::cuda::no_init};
            ::cuda::host_buffer<DeviceCamera> cameras{stream, ::cuda::pinned_default_memory_pool(), frame_set.frames.size(), ::cuda::no_init};

            std::size_t frame_index = 0uz;
            for (const dataset::multiview::Frame& frame : frame_set.frames) {
                std::ranges::copy(frame.rgba, pixels.data() + frame_index * first.rgba.size());
                const Matrix4<float>& transform = frame.world_from_camera;
                cameras.data()[frame_index]     = {
                        .x      = {transform(1uz, 0uz), transform(2uz, 0uz), transform(0uz, 0uz)},
                        .y      = {-transform(1uz, 1uz), -transform(2uz, 1uz), -transform(0uz, 1uz)},
                        .z      = {-transform(1uz, 2uz), -transform(2uz, 2uz), -transform(0uz, 2uz)},
                        .origin = {transform(1uz, 3uz) * scale + scene_offset, transform(2uz, 3uz) * scale + scene_offset, transform(0uz, 3uz) * scale + scene_offset},
                };
                ++frame_index;
            }

            DeviceFrameSet device_frame_set{
                .name        = frame_set.name,
                .extent      = first.extent,
                .intrinsics  = first.intrinsics,
                .frame_count = static_cast<std::uint32_t>(frame_set.frames.size()),
                .pixels      = ::cuda::device_buffer<std::uint8_t>{stream, ::cuda::device_default_memory_pool(stream.device()), pixels.size(), ::cuda::no_init},
                .cameras     = ::cuda::device_buffer<DeviceCamera>{stream, ::cuda::device_default_memory_pool(stream.device()), cameras.size(), ::cuda::no_init},
            };

            ::cuda::copy_bytes(stream, pixels, device_frame_set.pixels);
            ::cuda::copy_bytes(stream, cameras, device_frame_set.cameras);
            stream.sync();
            frame_sets.push_back(std::move(device_frame_set));
        }
    }
} // namespace physica::reconstruction::instant_ngp
