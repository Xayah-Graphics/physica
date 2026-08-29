module;

#include "ffmpeg.h"

#include <nlohmann/json.hpp>

module physica.reconstruction.dataset.pinf;

import std;

namespace physica::reconstruction::dataset::pinf {
    namespace {
        struct Video final {
            std::string file_name;
            std::vector<Matrix4<float>> world_from_camera;
            std::uint32_t frame_count = 0u;
            std::uint32_t view_index  = 0u;
        };
    } // namespace

    bool is_dataset(const std::filesystem::path& path) {
        return std::filesystem::is_regular_file(path / "info.json");
    }

    Dataset load(const std::filesystem::path& path, const LoadRequest request) {
        const nlohmann::json json = nlohmann::json::parse(std::ifstream{path / "info.json", std::ios::binary}, nullptr, true, true);
        Dataset result{
            .background    = {},
            .near_distance = json.at("near").get<float>(),
            .far_distance  = json.at("far").get<float>(),
            .phi           = json.at("phi").get<float>(),
            .rotation_axis = json.at("rot").get<std::string>()[0],
        };

        const nlohmann::json& render_center = json.at("render_center");
        const nlohmann::json& background    = json.at("frame_bkg_color");
        for (const std::size_t component : std::views::iota(0uz, 3uz)) {
            result.render_center[component] = render_center.at(component).get<float>();
            result.background[component]    = background.at(component).get<float>();
        }

        const nlohmann::json& voxel_scale = json.at("voxel_scale");
        if (voxel_scale.is_array())
            for (const std::size_t component : std::views::iota(0uz, 3uz)) result.voxel_scale[component] = voxel_scale.at(component).get<float>();
        else result.voxel_scale = {voxel_scale.get<float>(), voxel_scale.get<float>(), voxel_scale.get<float>()};

        const nlohmann::json& voxel_transform = json.at("voxel_matrix");
        constexpr std::array<std::size_t, 4> source_columns{2uz, 1uz, 0uz, 3uz};
        for (const std::size_t row : std::views::iota(0uz, 4uz))
            for (const std::size_t column : std::views::iota(0uz, 4uz)) result.voxel_transform(row, column) = voxel_transform.at(row).at(source_columns[column]).get<float>();

        const std::uint32_t resolution_divisor = request.resolution == Resolution::full ? 1u : request.resolution == Resolution::half ? 2u : 4u;
        constexpr std::array<std::string_view, 3> frame_set_names{"train", "validation", "test"};
        constexpr std::array<std::string_view, 3> json_names{"train_videos", "val_videos", "test_videos"};

        result.multiview.frame_sets.reserve(request.frame_sets.size());
        for (const std::string& frame_set_name : request.frame_sets) {
            const std::size_t frame_set_index = static_cast<std::size_t>(std::ranges::find(frame_set_names, frame_set_name) - frame_set_names.begin());
            const nlohmann::json& videos_json = json.at(json_names[frame_set_index]);
            multiview::FrameSet frame_set{
                .name       = frame_set_name,
                .view_count = static_cast<std::uint32_t>(videos_json.size()),
            };
            frame_set.frames.reserve(videos_json.size() * videos_json.front().at("frame_num").get<std::uint32_t>());

            for (const std::size_t view_index : std::views::iota(0uz, videos_json.size())) {
                const nlohmann::json& video_json = videos_json.at(view_index);
                Video video{
                    .file_name   = video_json.at("file_name").get<std::string>(),
                    .frame_count = video_json.at("frame_num").get<std::uint32_t>(),
                    .view_index  = static_cast<std::uint32_t>(view_index),
                };
                video.world_from_camera.resize(video.frame_count);
                if (video_json.contains("transform_matrix")) {
                    const nlohmann::json& transform = video_json.at("transform_matrix");
                    Matrix4<float> matrix;
                    for (const std::size_t row : std::views::iota(0uz, 4uz))
                        for (const std::size_t column : std::views::iota(0uz, 4uz)) matrix(row, column) = transform.at(row).at(column).get<float>();
                    std::ranges::fill(video.world_from_camera, matrix);
                } else {
                    const nlohmann::json& transforms = video_json.at("transform_matrix_list");
                    for (const std::size_t frame_index : std::views::iota(0uz, video.world_from_camera.size()))
                        for (const std::size_t row : std::views::iota(0uz, 4uz))
                            for (const std::size_t column : std::views::iota(0uz, 4uz)) video.world_from_camera[frame_index](row, column) = transforms.at(frame_index).at(row).at(column).get<float>();
                }

                VideoDecoder decoder{(path / video.file_name).string().c_str(), resolution_divisor};
                const std::uint32_t width  = decoder.width;
                const std::uint32_t height = decoder.height;
                const float focal          = 0.5F * static_cast<float>(width) / std::tan(0.5F * video_json.at("camera_angle_x").get<float>());
                frame_set.time_count       = video.frame_count;

                const auto append_frame = [&](const std::uint32_t time_index) {
                    multiview::Frame frame{
                        .name              = std::format("{}#{:04}", video.file_name, time_index),
                        .rgba              = std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height * 4uz),
                        .world_from_camera = video.world_from_camera[time_index],
                        .extent            = {.width = width, .height = height},
                        .intrinsics =
                            {
                                .focal_x     = focal,
                                .focal_y     = focal,
                                .principal_x = static_cast<float>(width) * 0.5F,
                                .principal_y = static_cast<float>(height) * 0.5F,
                            },
                        .time       = static_cast<float>(time_index) / static_cast<float>(video.frame_count),
                        .view_index = video.view_index,
                        .time_index = time_index,
                    };
                    decoder.read(frame.rgba.data());
                    frame_set.frames.push_back(std::move(frame));
                };

                for (const std::uint32_t time_index : std::views::iota(0u, video.frame_count)) append_frame(time_index);
            }
            result.multiview.frame_sets.push_back(std::move(frame_set));
        }
        return result;
    }
} // namespace physica::reconstruction::dataset::pinf
