export module physica.reconstruction.dataset.multiview;

import std;

export namespace physica::reconstruction::dataset::multiview {
    struct Extent final {
        std::uint32_t width  = 0u;
        std::uint32_t height = 0u;
    };

    struct Intrinsics final {
        float focal_x     = 0.0F;
        float focal_y     = 0.0F;
        float principal_x = 0.0F;
        float principal_y = 0.0F;
    };

    struct Frame final {
        std::string name;
        std::vector<std::uint8_t> rgba;
        std::array<float, 16> world_from_camera{};
        Extent extent;
        Intrinsics intrinsics;
        float time               = 0.0F;
        std::uint32_t view_index = 0u;
        std::uint32_t time_index = 0u;
    };

    struct FrameSet final {
        std::string name;
        std::vector<Frame> frames;
        std::uint32_t view_count = 0u;
        std::uint32_t time_count = 0u;
    };

    struct Dataset final {
        std::vector<FrameSet> frame_sets;
    };
} // namespace physica::reconstruction::dataset::multiview
