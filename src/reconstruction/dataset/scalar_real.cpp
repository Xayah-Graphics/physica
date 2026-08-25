module;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 5202)
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <nlohmann/json.hpp>
module physica.reconstruction.dataset.scalar_real;
import std;

namespace physica::reconstruction::dataset::scalar_real {
    bool is_dataset(const std::filesystem::path& path) {
        return std::filesystem::is_regular_file(path / "info.json");
    }

    Dataset load(const std::filesystem::path& path, const LoadRequest request) {
        Dataset dataset = {};
        dataset.multiview.frame_sets.reserve(request.frame_sets.size());
        for (const std::string& frame_set_name : request.frame_sets) {
            dataset.multiview.frame_sets.push_back(multiview::FrameSet{.name = frame_set_name});
        }

        const std::filesystem::path info_path = path / "info.json";
        const nlohmann::json json             = nlohmann::json::parse(std::ifstream{info_path, std::ios::binary}, nullptr, true, true);

        dataset.near                         = json.at("near").get<float>();
        dataset.far                          = json.at("far").get<float>();
        dataset.phi                          = json.at("phi").get<float>();
        const std::string rotation_axis_text = json.at("rot").get<std::string>();
        dataset.rotation_axis                = rotation_axis_text[0];

        const nlohmann::json& render_center_json = json.at("render_center");
        for (std::size_t i = 0uz; i < 3uz; ++i) dataset.render_center[i] = render_center_json.at(i).get<float>();

        const nlohmann::json& voxel_scale_json = json.at("voxel_scale");
        for (std::size_t i = 0uz; i < 3uz; ++i) {
            dataset.voxel_scale[i] = voxel_scale_json.at(i).get<float>();
        }

        const nlohmann::json& voxel_json = json.at("voxel_matrix");
        for (std::size_t row = 0uz; row < 4uz; ++row)
            for (std::size_t column = 0uz; column < 4uz; ++column) {
                dataset.voxel_matrix[row * 4uz + column] = voxel_json.at(row).at(column).get<float>();
            }

        const std::array<std::string_view, 2> split_json_names = {"train_videos", "test_videos"};
        const std::array<std::string_view, 2> frame_set_names  = {"train", "test"};
        for (const std::size_t split_index : std::views::iota(0uz, split_json_names.size())) {
            multiview::FrameSet* frame_destination = nullptr;
            for (multiview::FrameSet& frame_set : dataset.multiview.frame_sets) {
                if (frame_set.name == frame_set_names[split_index]) {
                    frame_destination = std::addressof(frame_set);
                    break;
                }
            }
            if (frame_destination == nullptr) continue;

            const nlohmann::json& videos_json = json.at(split_json_names[split_index]);

            std::uint32_t split_frame_count = 0u;
            std::uint32_t video_index       = 0u;
            for (const nlohmann::json& video_json : videos_json) {
                Video video                = {};
                video.frame_set            = frame_set_names[split_index];
                video.file_name            = video_json.at("file_name").get<std::string>();
                video.frame_count          = video_json.at("frame_num").get<std::uint32_t>();
                video.frame_rate           = video_json.at("frame_rate").get<std::uint32_t>();
                video.view_index           = video_index;
                const float camera_angle_x = video_json.at("camera_angle_x").get<float>();

                const nlohmann::json& transform_matrix_json = video_json.at("transform_matrix");
                for (std::size_t row = 0uz; row < 4uz; ++row)
                    for (std::size_t column = 0uz; column < 4uz; ++column) {
                        video.world_from_camera[row * 4uz + column] = transform_matrix_json.at(row).at(column).get<float>();
                    }

                const std::filesystem::path video_path = path / video.file_name;
                AVFormatContext* format_context        = nullptr;
                AVCodecContext* codec_context          = nullptr;
                SwsContext* sws_context                = nullptr;
                AVPacket* packet                       = nullptr;
                AVFrame* frame                         = nullptr;
                AVFrame* rgb_frame                     = nullptr;
                const auto release_decode_state        = [&]() noexcept {
                    if (rgb_frame != nullptr && rgb_frame->data[0] != nullptr) av_freep(&rgb_frame->data[0]);
                    if (rgb_frame != nullptr) av_frame_free(&rgb_frame);
                    if (frame != nullptr) av_frame_free(&frame);
                    if (packet != nullptr) av_packet_free(&packet);
                    if (sws_context != nullptr) sws_freeContext(sws_context);
                    if (codec_context != nullptr) avcodec_free_context(&codec_context);
                    if (format_context != nullptr) avformat_close_input(&format_context);
                };
                try {
                    avformat_open_input(&format_context, video_path.string().c_str(), nullptr, nullptr);
                    avformat_find_stream_info(format_context, nullptr);
                    int video_stream_index = -1;
                    for (unsigned int stream_index = 0u; stream_index < format_context->nb_streams; ++stream_index) {
                        if (format_context->streams[stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                            video_stream_index = static_cast<int>(stream_index);
                            break;
                        }
                    }

                    AVStream* stream       = format_context->streams[video_stream_index];
                    const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
                    codec_context          = avcodec_alloc_context3(decoder);
                    avcodec_parameters_to_context(codec_context, stream->codecpar);
                    avcodec_open2(codec_context, decoder, nullptr);

                    video.width  = static_cast<std::uint32_t>(codec_context->width);
                    video.height = static_cast<std::uint32_t>(codec_context->height);
                    video.focal  = 0.5f * static_cast<float>(video.width) / std::tan(0.5f * camera_angle_x);

                    split_frame_count = video.frame_count;

                    sws_context = sws_getContext(codec_context->width, codec_context->height, codec_context->pix_fmt, codec_context->width, codec_context->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);

                    packet    = av_packet_alloc();
                    frame     = av_frame_alloc();
                    rgb_frame = av_frame_alloc();
                    av_image_alloc(rgb_frame->data, rgb_frame->linesize, codec_context->width, codec_context->height, AV_PIX_FMT_RGB24, 32);

                    frame_destination->frames.reserve(frame_destination->frames.size() + video.frame_count);
                    const auto append_frame = [&](const std::uint32_t decoded_frame_index) {
                        sws_scale(sws_context, frame->data, frame->linesize, 0, codec_context->height, rgb_frame->data, rgb_frame->linesize);
                        multiview::Frame output_frame{
                            .name              = std::format("{}#{:04}", video.file_name, decoded_frame_index),
                            .rgba              = std::vector<std::uint8_t>(static_cast<std::size_t>(video.width) * video.height * 4uz),
                            .world_from_camera = video.world_from_camera,
                            .extent            = {.width = video.width, .height = video.height},
                            .intrinsics =
                                {
                                    .focal_x     = video.focal,
                                    .focal_y     = video.focal,
                                    .principal_x = static_cast<float>(video.width) * 0.5f,
                                    .principal_y = static_cast<float>(video.height) * 0.5f,
                                },
                            .time       = video.frame_count > 1u ? static_cast<float>(decoded_frame_index) / static_cast<float>(video.frame_count - 1u) : 0.0f,
                            .view_index = video.view_index,
                            .time_index = decoded_frame_index,
                        };
                        for (std::uint32_t row = 0u; row < video.height; ++row) {
                            const std::uint8_t* row_begin = rgb_frame->data[0] + static_cast<std::ptrdiff_t>(row) * rgb_frame->linesize[0];
                            std::uint8_t* row_out         = output_frame.rgba.data() + static_cast<std::uint64_t>(row) * video.width * 4ull;
                            for (std::uint32_t pixel = 0u; pixel < video.width; ++pixel) {
                                row_out[pixel * 4ull + 0ull] = row_begin[pixel * 3ull + 0ull];
                                row_out[pixel * 4ull + 1ull] = row_begin[pixel * 3ull + 1ull];
                                row_out[pixel * 4ull + 2ull] = row_begin[pixel * 3ull + 2ull];
                                row_out[pixel * 4ull + 3ull] = 255u;
                            }
                        }
                        frame_destination->frames.push_back(std::move(output_frame));
                    };
                    std::uint32_t decoded_frame_index = 0u;
                    while (av_read_frame(format_context, packet) >= 0) {
                        if (packet->stream_index == video_stream_index) {
                            avcodec_send_packet(codec_context, packet);
                            while (true) {
                                const int receive_status = avcodec_receive_frame(codec_context, frame);
                                if (receive_status == AVERROR(EAGAIN) || receive_status == AVERROR_EOF) break;
                                append_frame(decoded_frame_index);
                                ++decoded_frame_index;
                            }
                        }
                        av_packet_unref(packet);
                    }

                    avcodec_send_packet(codec_context, nullptr);
                    while (true) {
                        const int receive_status = avcodec_receive_frame(codec_context, frame);
                        if (receive_status == AVERROR_EOF || receive_status == AVERROR(EAGAIN)) break;
                        append_frame(decoded_frame_index);
                        ++decoded_frame_index;
                    }
                } catch (...) {
                    release_decode_state();
                    throw;
                }

                release_decode_state();
                dataset.videos.push_back(std::move(video));
                ++video_index;
            }
            frame_destination->view_count = video_index;
            frame_destination->time_count = split_frame_count;
        }

        return dataset;
    }
} // namespace physica::reconstruction::dataset::scalar_real
