#ifndef PHYSICA_RECONSTRUCTION_DATASET_FFMPEG_H
#define PHYSICA_RECONSTRUCTION_DATASET_FFMPEG_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace physica::reconstruction::dataset {
    struct VideoDecoder final {
        unsigned width  = 0u;
        unsigned height = 0u;

        VideoDecoder(const std::filesystem::path& path, unsigned resolution_divisor);
        ~VideoDecoder() noexcept;

        VideoDecoder(const VideoDecoder&)            = delete;
        VideoDecoder& operator=(const VideoDecoder&) = delete;
        VideoDecoder(VideoDecoder&&)                 = delete;
        VideoDecoder& operator=(VideoDecoder&&)      = delete;

        void read(std::span<std::uint8_t> rgba);

    private:
        struct FormatDeleter final {
            void operator()(AVFormatContext* format) const noexcept;
        };
        struct CodecDeleter final {
            void operator()(AVCodecContext* codec) const noexcept;
        };
        struct ConversionDeleter final {
            void operator()(SwsContext* conversion) const noexcept;
        };
        struct PacketDeleter final {
            void operator()(AVPacket* packet) const noexcept;
        };
        struct FrameDeleter final {
            void operator()(AVFrame* frame) const noexcept;
        };

        std::unique_ptr<AVFormatContext, FormatDeleter> format;
        std::unique_ptr<AVCodecContext, CodecDeleter> codec;
        std::unique_ptr<SwsContext, ConversionDeleter> conversion;
        std::unique_ptr<AVPacket, PacketDeleter> packet;
        std::unique_ptr<AVFrame, FrameDeleter> decoded;
        int stream_index = 0;
        bool flushing    = false;
    };
} // namespace physica::reconstruction::dataset

#endif
