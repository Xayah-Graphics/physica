module;

#include <nlohmann/json.hpp>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

module physica.reconstruction.dataset.nerf_synthetic;

import std;

namespace physica::reconstruction::dataset::nerf_synthetic {
    bool is_dataset(const std::filesystem::path& path) {
        return std::filesystem::is_regular_file(path / "transforms_train.json") || std::filesystem::is_regular_file(path / "transforms_val.json") || std::filesystem::is_regular_file(path / "transforms_test.json");
    }

    multiview::Dataset load(const std::filesystem::path& path, const LoadRequest request) {
        multiview::Dataset dataset;
        dataset.frame_sets.reserve(request.frame_sets.size());

        for (const std::string& frame_set_name : request.frame_sets) {
            const std::string file_name            = frame_set_name == "validation" ? "transforms_val.json" : std::format("transforms_{}.json", frame_set_name);
            const std::filesystem::path json_path  = path / file_name;
            const std::filesystem::path split_root = json_path.parent_path();
            const nlohmann::json json              = nlohmann::json::parse(std::ifstream{json_path, std::ios::binary}, nullptr, true, true);
            const float camera_angle_x             = json.at("camera_angle_x").get<float>();
            const nlohmann::json& frames_json      = json.at("frames");

            multiview::FrameSet frame_set{
                .name       = frame_set_name,
                .frames     = std::vector<multiview::Frame>(frames_json.size()),
                .view_count = static_cast<std::uint32_t>(frames_json.size()),
                .time_count = 1u,
            };
            const std::vector<std::size_t> indices = std::views::iota(0uz, frames_json.size()) | std::ranges::to<std::vector<std::size_t>>();

            std::for_each(std::execution::par, indices.begin(), indices.end(), [&](const std::size_t frame_index) {
                const nlohmann::json& frame_json = frames_json.at(frame_index);
                std::filesystem::path image_path = frame_json.at("file_path").get<std::string>();
                image_path.make_preferred();
                if (image_path.extension().empty()) image_path.replace_extension(".png");
                if (image_path.is_relative()) image_path = split_root / image_path;
                image_path = image_path.lexically_normal();

                int width           = 0;
                int height          = 0;
                int component_count = 0;
                const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw_pixels{stbi_load(image_path.string().c_str(), &width, &height, &component_count, 4), stbi_image_free};
                const std::uint32_t width_u  = static_cast<std::uint32_t>(width);
                const std::uint32_t height_u = static_cast<std::uint32_t>(height);
                const std::size_t rgba_size  = static_cast<std::size_t>(width_u) * height_u * 4uz;

                std::array<float, 16> world_from_camera{};
                const nlohmann::json& transform_matrix = frame_json.at("transform_matrix");
                for (const std::size_t row : std::views::iota(0uz, 4uz))
                    for (const std::size_t column : std::views::iota(0uz, 4uz)) world_from_camera[row * 4uz + column] = transform_matrix.at(row).at(column).get<float>();

                const float focal_length      = 0.5F * static_cast<float>(width_u) / std::tan(camera_angle_x * 0.5F);
                frame_set.frames[frame_index] = multiview::Frame{
                    .name              = image_path.filename().string(),
                    .rgba              = std::vector<std::uint8_t>{raw_pixels.get(), raw_pixels.get() + rgba_size},
                    .world_from_camera = world_from_camera,
                    .extent            = {.width = width_u, .height = height_u},
                    .intrinsics =
                        {
                            .focal_x     = focal_length,
                            .focal_y     = focal_length,
                            .principal_x = static_cast<float>(width_u) * 0.5F,
                            .principal_y = static_cast<float>(height_u) * 0.5F,
                        },
                    .view_index = static_cast<std::uint32_t>(frame_index),
                };
            });

            dataset.frame_sets.push_back(std::move(frame_set));
        }
        return dataset;
    }
} // namespace physica::reconstruction::dataset::nerf_synthetic
