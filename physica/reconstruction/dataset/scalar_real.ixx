export module physica.reconstruction.dataset.scalar_real;

export import physica.reconstruction.dataset.multiview;
import std;

namespace physica::reconstruction::dataset::scalar_real {
    export struct Video final {
        std::string frame_set;
        std::string file_name;
        Matrix4<float> world_from_camera        = {};
        std::uint32_t width                     = 0u;
        std::uint32_t height                    = 0u;
        std::uint32_t frame_count               = 0u;
        std::uint32_t frame_rate                = 0u;
        std::uint32_t view_index                = 0u;
        float focal                             = 0.0f;
    };

    export struct Dataset final {
        multiview::Dataset multiview;
        std::vector<Video> videos;
        Matrix4<float> voxel_matrix  = {};
        Vector3<float> voxel_scale   = {};
        Vector3<float> render_center = {};
        float near                   = 0.0f;
        float far                    = 0.0f;
        float phi                    = 0.0f;
        char rotation_axis           = 'Y';
    };

    export struct LoadRequest final {
        std::vector<std::string> frame_sets = {};
    };

    export bool is_dataset(const std::filesystem::path& path);
    export Dataset load(const std::filesystem::path& path, LoadRequest request);
} // namespace physica::reconstruction::dataset::scalar_real
