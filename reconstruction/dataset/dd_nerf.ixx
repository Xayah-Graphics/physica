export module physica.reconstruction.dataset.dd_nerf;

export import physica.reconstruction.dataset.multiview;
import std;

namespace physica::reconstruction::dataset::dd_nerf {
    export struct LoadRequest final {
        std::vector<std::string> frame_sets = {};
    };

    export bool is_dataset(const std::filesystem::path& path);
    export multiview::Dataset load(const std::filesystem::path& path, LoadRequest request);
}
