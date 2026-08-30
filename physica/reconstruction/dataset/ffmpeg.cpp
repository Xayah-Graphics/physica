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
    VideoDecoder::VideoDecoder(const char* path, const unsigned resolution_divisor) : packet{av_packet_alloc()}, decoded{av_frame_alloc()} {
        avformat_open_input(&format, path, nullptr, nullptr);
        avformat_find_stream_info(format, nullptr);
        while (format->streams[stream_index]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) ++stream_index;
        AVStream* stream       = format->streams[stream_index];
        const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        codec                  = avcodec_alloc_context3(decoder);
        avcodec_parameters_to_context(codec, stream->codecpar);
        avcodec_open2(codec, decoder, nullptr);

        width      = static_cast<unsigned>(codec->width) / resolution_divisor;
        height     = static_cast<unsigned>(codec->height) / resolution_divisor;
        conversion = sws_getContext(codec->width, codec->height, codec->pix_fmt, static_cast<int>(width), static_cast<int>(height), AV_PIX_FMT_RGBA, SWS_AREA, nullptr, nullptr, nullptr);
    }

    VideoDecoder::~VideoDecoder() noexcept {
        av_frame_free(&decoded);
        av_packet_free(&packet);
        sws_freeContext(conversion);
        avcodec_free_context(&codec);
        avformat_close_input(&format);
    }

    void VideoDecoder::read(unsigned char* rgba) {
        for (;;) {
            if (avcodec_receive_frame(codec, decoded) == 0) break;
            if (flushing) continue;

            bool packet_ready = false;
            while (av_read_frame(format, packet) >= 0) {
                packet_ready = packet->stream_index == stream_index;
                if (packet_ready) break;
                av_packet_unref(packet);
            }
            if (packet_ready) {
                avcodec_send_packet(codec, packet);
                av_packet_unref(packet);
            } else {
                avcodec_send_packet(codec, nullptr);
                flushing = true;
            }
        }

        unsigned char* destination[4]{};
        int destination_lines[4]{};
        av_image_fill_arrays(destination, destination_lines, rgba, AV_PIX_FMT_RGBA, static_cast<int>(width), static_cast<int>(height), 1);
        sws_scale(conversion, decoded->data, decoded->linesize, 0, codec->height, destination, destination_lines);
    }
} // namespace physica::reconstruction::dataset
