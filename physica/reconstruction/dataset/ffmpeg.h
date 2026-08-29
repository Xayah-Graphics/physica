#ifndef PHYSICA_RECONSTRUCTION_DATASET_FFMPEG_H
#define PHYSICA_RECONSTRUCTION_DATASET_FFMPEG_H

struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace physica::reconstruction::dataset {
    struct VideoDecoder final {
        unsigned width  = 0u;
        unsigned height = 0u;

        VideoDecoder(const char* path, unsigned resolution_divisor);
        ~VideoDecoder() noexcept;

        VideoDecoder(const VideoDecoder&)            = delete;
        VideoDecoder& operator=(const VideoDecoder&) = delete;
        VideoDecoder(VideoDecoder&&)                 = delete;
        VideoDecoder& operator=(VideoDecoder&&)      = delete;

        void read(unsigned char* rgba);

    private:
        AVFormatContext* format = nullptr;
        AVCodecContext* codec   = nullptr;
        SwsContext* conversion  = nullptr;
        AVPacket* packet        = nullptr;
        AVFrame* decoded        = nullptr;
        int stream_index        = 0;
        bool flushing           = false;
    };
} // namespace physica::reconstruction::dataset

#endif
