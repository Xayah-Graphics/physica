export module physica.reconstruction.dataset.pinf;

export import physica.reconstruction.dataset.multiview;
import std;

export namespace physica::reconstruction::dataset::pinf {
    enum class Resolution : std::uint8_t {
        full,
        half,
        quarter,
    };

    struct LoadRequest final {
        std::vector<std::string> frame_sets{};
        Resolution resolution = Resolution::full;
    };

    struct Dataset final {
        multiview::Dataset multiview;
        Matrix4<float> voxel_transform;
        Vector3<float> voxel_scale;
        Vector3<float> render_center;
        Vector3<float> background;
        float near_distance = 0.0F;
        float far_distance  = 0.0F;
        float phi           = 0.0F;
        char rotation_axis  = 'Y';
    };

    bool is_dataset(const std::filesystem::path& path);
    Dataset load(const std::filesystem::path& path, LoadRequest request);
} // namespace physica::reconstruction::dataset::pinf
