module physica.generative.dataset.cifar10;

import std;

namespace physica::generative {
    Cifar10TrainingSet::Cifar10TrainingSet(const std::filesystem::path& directory)
        : images(static_cast<std::size_t>(image_count) * image_byte_count), labels(image_count) {
        std::array<std::uint8_t, image_byte_count + 1u> record{};
        for (std::uint32_t batch = 0u; batch < 5u; ++batch) {
            std::ifstream file{directory / ("data_batch_" + std::to_string(batch + 1u) + ".bin"), std::ios::binary};
            if (!file) throw std::runtime_error{"Unable to open the CIFAR-10 training batch."};
            for (std::uint32_t image = 0u; image < 10'000u; ++image) {
                file.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(record.size()));
                const std::size_t index = static_cast<std::size_t>(batch) * 10'000uz + image;
                labels[index] = record[0];
                std::ranges::copy(record | std::views::drop(1), images.begin() + static_cast<std::ptrdiff_t>(index * image_byte_count));
            }
        }
    }
} // namespace physica::generative
