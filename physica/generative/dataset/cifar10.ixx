export module physica.generative.dataset.cifar10;

import std;

export namespace physica::generative {
    struct Cifar10TrainingSet final {
        inline static constexpr std::uint32_t width            = 32u;
        inline static constexpr std::uint32_t height           = 32u;
        inline static constexpr std::uint32_t channel_count    = 3u;
        inline static constexpr std::uint32_t image_byte_count = width * height * channel_count;
        inline static constexpr std::uint32_t image_count      = 50'000u;

        std::vector<std::uint8_t> images;
        std::vector<std::uint8_t> labels;

        explicit Cifar10TrainingSet(const std::filesystem::path& directory);
    };
} // namespace physica::generative
