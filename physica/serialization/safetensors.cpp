module;

#include <nlohmann/json.hpp>

module physica.serialization.safetensors;

import std;

namespace physica::serialization::safetensors {
    void write(const std::filesystem::path& path, const std::string_view system, const std::span<const TensorView> tensors) {
        nlohmann::json header;
        header["__metadata__"] = {
            {"physica.format_version", "1"},
            {"physica.project_version", PHYSICA_PROJECT_VERSION},
            {"physica.system", system},
        };
        std::uint64_t offset{};
        for (const TensorView& tensor : tensors) {
            header[tensor.name] = {
                {"dtype", tensor.dtype},
                {"shape", tensor.shape},
                {"data_offsets", nlohmann::json::array({offset, offset + tensor.byte_count})},
            };
            offset += tensor.byte_count;
        }
        std::string text = header.dump();
        text.append((8uz - text.size() % 8uz) % 8uz, ' ');
        const std::uint64_t header_size = text.size();
        std::ofstream output{path, std::ios::binary | std::ios::trunc};
        output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        for (const TensorView& tensor : tensors) output.write(static_cast<const char*>(tensor.data), static_cast<std::streamsize>(tensor.byte_count));
    }

    File read(const std::filesystem::path& path) {
        std::ifstream input{path, std::ios::binary};
        std::uint64_t header_size{};
        input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        std::string text(header_size, '\0');
        input.read(text.data(), static_cast<std::streamsize>(text.size()));
        const nlohmann::json header     = nlohmann::json::parse(text);
        const std::uint64_t data_offset = sizeof(header_size) + header_size;
        File result;
        result.metadata = header.at("__metadata__").get<std::map<std::string, std::string>>();
        result.tensors.reserve(header.size() - 1uz);
        for (const auto& [name, description] : header.items()) {
            if (name == "__metadata__") continue;
            const std::uint64_t begin = description.at("data_offsets").at(0).get<std::uint64_t>();
            const std::uint64_t end   = description.at("data_offsets").at(1).get<std::uint64_t>();
            Tensor tensor{
                .name  = name,
                .dtype = description.at("dtype").get<std::string>(),
                .shape = description.at("shape").get<std::vector<std::uint64_t>>(),
                .data  = std::vector<std::byte>(end - begin),
            };
            input.seekg(static_cast<std::streamoff>(data_offset + begin));
            input.read(reinterpret_cast<char*>(tensor.data.data()), static_cast<std::streamsize>(tensor.data.size()));
            result.tensors.push_back(std::move(tensor));
        }
        return result;
    }
} // namespace physica::serialization::safetensors
