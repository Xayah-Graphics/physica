#include "ffmpeg.h"

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

namespace physica::reconstruction::dataset {
    void VideoDecoder::FormatDeleter::operator()(AVFormatContext* source) const noexcept {
        avformat_close_input(&source);
    }

    void VideoDecoder::CodecDeleter::operator()(AVCodecContext* source) const noexcept {
        avcodec_free_context(&source);
    }

    void VideoDecoder::ConversionDeleter::operator()(SwsContext* const source) const noexcept {
        sws_freeContext(source);
    }

    void VideoDecoder::PacketDeleter::operator()(AVPacket* source) const noexcept {
        av_packet_free(&source);
    }

    void VideoDecoder::FrameDeleter::operator()(AVFrame* source) const noexcept {
        av_frame_free(&source);
    }

    VideoDecoder::VideoDecoder(const std::filesystem::path& path, const unsigned resolution_divisor) : packet{av_packet_alloc()}, decoded{av_frame_alloc()} {
        const std::string path_string = path.string();
        avformat_open_input(std::out_ptr(format), path_string.c_str(), nullptr, nullptr);
        avformat_find_stream_info(format.get(), nullptr);
        while (format->streams[stream_index]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) ++stream_index;
        AVStream* stream       = format->streams[stream_index];
        const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        codec.reset(avcodec_alloc_context3(decoder));
        avcodec_parameters_to_context(codec.get(), stream->codecpar);
        avcodec_open2(codec.get(), decoder, nullptr);

        width  = static_cast<unsigned>(codec->width) / resolution_divisor;
        height = static_cast<unsigned>(codec->height) / resolution_divisor;
        conversion.reset(sws_getContext(codec->width, codec->height, codec->pix_fmt, static_cast<int>(width), static_cast<int>(height), AV_PIX_FMT_RGBA, SWS_AREA, nullptr, nullptr, nullptr));
    }

    VideoDecoder::~VideoDecoder() noexcept = default;

    void VideoDecoder::read(const std::span<std::uint8_t> rgba) {
        for (;;) {
            if (avcodec_receive_frame(codec.get(), decoded.get()) == 0) break;
            if (flushing) continue;

            bool packet_ready = false;
            while (av_read_frame(format.get(), packet.get()) >= 0) {
                packet_ready = packet->stream_index == stream_index;
                if (packet_ready) break;
                av_packet_unref(packet.get());
            }
            if (packet_ready) {
                avcodec_send_packet(codec.get(), packet.get());
                av_packet_unref(packet.get());
            } else {
                avcodec_send_packet(codec.get(), nullptr);
                flushing = true;
            }
        }

        AVFrame destination{};
        destination.format = AV_PIX_FMT_RGBA;
        destination.width  = static_cast<int>(width);
        destination.height = static_cast<int>(height);
        av_image_fill_arrays(destination.data, destination.linesize, rgba.data(), AV_PIX_FMT_RGBA, destination.width, destination.height, 1);
        sws_scale_frame(conversion.get(), &destination, decoded.get());
    }
} // namespace physica::reconstruction::dataset
