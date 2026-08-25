module;

#include <nlohmann/json.hpp>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

module physica.reconstruction.dataset.dd_nerf;

import std;

namespace physica::reconstruction::dataset::dd_nerf {
    bool is_dataset(const std::filesystem::path& path) {
        return std::filesystem::is_regular_file(path / "cameras.json") && std::filesystem::is_directory(path / "images");
    }

    multiview::Dataset load(const std::filesystem::path& path, const LoadRequest request) {
        const nlohmann::json json         = nlohmann::json::parse(std::ifstream{path / "cameras.json", std::ios::binary}, nullptr, true, true);
        const nlohmann::json& frames_json = json.at("frames");
        const std::uint32_t width         = json.at("w").get<std::uint32_t>();
        const std::uint32_t height        = json.at("h").get<std::uint32_t>();
        const multiview::Intrinsics intrinsics{
            .focal_x     = json.at("fl_x").get<float>(),
            .focal_y     = json.at("fl_y").get<float>(),
            .principal_x = json.at("cx").get<float>(),
            .principal_y = json.at("cy").get<float>(),
        };

        multiview::Dataset dataset;
        dataset.frame_sets.reserve(request.frame_sets.size());
        for (const std::string& frame_set_name : request.frame_sets) {
            multiview::FrameSet frame_set{.name = frame_set_name, .time_count = 1u};
            if (frame_set_name == "train") frame_set.frames.reserve(frames_json.size());
            else frame_set.frames.reserve(frames_json.size() / 10uz + 1uz);
            dataset.frame_sets.push_back(std::move(frame_set));
        }

        for (const nlohmann::json& frame_json : frames_json) {
            const std::filesystem::path hash_path = frame_json.at("file_path").get<std::string>();
            const std::string hash_text           = hash_path.filename().string();
            std::uint64_t hash                    = 14695981039346656037ull;
            for (const unsigned char byte : hash_text) {
                hash ^= static_cast<std::uint64_t>(byte);
                hash *= 1099511628211ull;
            }

            const std::uint64_t split             = hash % 10ull;
            const std::string_view frame_set_name = split == 0ull ? "test" : (split == 1ull ? "validation" : "train");
            multiview::FrameSet* target_frame_set = nullptr;
            for (multiview::FrameSet& frame_set : dataset.frame_sets) {
                if (frame_set.name == frame_set_name) {
                    target_frame_set = std::addressof(frame_set);
                    break;
                }
            }
            if (target_frame_set == nullptr) continue;

            std::filesystem::path image_path = frame_json.at("file_path").get<std::string>();
            image_path.make_preferred();
            if (image_path.is_relative()) image_path = path / image_path;
            image_path = image_path.lexically_normal();

            int image_width     = 0;
            int image_height    = 0;
            int component_count = 0;
            const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw_pixels{stbi_load(image_path.string().c_str(), &image_width, &image_height, &component_count, 4), stbi_image_free};

            std::array<float, 16> world_from_camera{};
            const nlohmann::json& transform_matrix = frame_json.at("transform_matrix");
            for (const std::size_t row : std::views::iota(0uz, 4uz))
                for (const std::size_t column : std::views::iota(0uz, 4uz)) world_from_camera[row * 4uz + column] = transform_matrix.at(row).at(column).get<float>();

            target_frame_set->frames.push_back(multiview::Frame{
                .name              = image_path.filename().string(),
                .rgba              = std::vector<std::uint8_t>{raw_pixels.get(), raw_pixels.get() + static_cast<std::size_t>(width) * height * 4uz},
                .world_from_camera = world_from_camera,
                .extent            = {.width = width, .height = height},
                .intrinsics        = intrinsics,
                .view_index        = static_cast<std::uint32_t>(target_frame_set->frames.size()),
            });
        }

        for (multiview::FrameSet& frame_set : dataset.frame_sets) frame_set.view_count = static_cast<std::uint32_t>(frame_set.frames.size());
        return dataset;
    }
} // namespace physica::reconstruction::dataset::dd_nerf
