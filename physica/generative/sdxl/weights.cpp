module;
#if defined(_WIN32)
#include <Windows.h>
#elif defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "../../neural/inference-kernels.h"
#include "kernels.h"
#include <physica/cuda.h>

#include <nlohmann/json.hpp>

module physica.generative.sdxl.weights;
import std;
import physica.neural.inference_runtime;

namespace physica::generative::sdxl {
    Checkpoint::Checkpoint(const std::filesystem::path& path, neural::InferenceRuntime& execution, Weights& storage) : runtime{execution}, weights{storage} {
#if defined(_WIN32)
        file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "SDXL checkpoint open"};
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "SDXL checkpoint mapping"};
        view = static_cast<const std::byte*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
        if (!view) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "SDXL checkpoint view"};
#elif defined(__linux__)
        mapped_size = std::filesystem::file_size(path);
        const int file = ::open(path.c_str(), O_RDONLY);
        if (file == -1) throw std::system_error{errno, std::generic_category(), "SDXL checkpoint open"};
        view = static_cast<const std::byte*>(::mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, file, 0));
        const int mapping_error = errno;
        ::close(file);
        if (view == MAP_FAILED) throw std::system_error{mapping_error, std::generic_category(), "SDXL checkpoint mapping"};
#endif
        std::uint64_t header_size;
        std::memcpy(&header_size, view, sizeof(header_size));
        index = nlohmann::json::parse(reinterpret_cast<const char*>(view + 8), reinterpret_cast<const char*>(view + 8 + header_size));
        data  = view + 8 + header_size;
    }

    Checkpoint::~Checkpoint() {
        runtime.stream.sync();
#if defined(_WIN32)
        UnmapViewOfFile(view);
        CloseHandle(mapping);
        CloseHandle(file);
#elif defined(__linux__)
        ::munmap(const_cast<std::byte*>(view), mapped_size);
#endif
    }

    neural::TensorView Checkpoint::tensor(const std::string& name, const neural::Scalar scalar, const bool convolution, const bool transpose) {
        const auto& entry       = index.at(name);
        const auto shape        = entry.at("shape").get<std::vector<int>>();
        const auto offsets      = entry.at("data_offsets").get<std::array<std::size_t, 2>>();
        const std::size_t count = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<>{});
        const std::unordered_map<std::string, neural::Scalar> types{{"F16", neural::Scalar::f16}, {"F32", neural::Scalar::f32}, {"BF16", neural::Scalar::bf16}};
        const neural::Scalar source_scalar = types.at(entry.at("dtype").get<std::string>());
        auto& destination                  = weights.storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), count * (scalar == neural::Scalar::f32 ? 4uz : 2uz), ::cuda::no_init);
        neural::TensorView result{destination.data(), 1, 1, shape.size() >= 2 ? shape[0] : 1, shape.size() >= 2 ? shape[1] : shape[0], scalar};
        if (!convolution && !transpose && source_scalar == scalar) {
            ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{data + offsets[0], destination.size()}, destination);
        } else {
            ::cuda::device_buffer<std::byte> source{runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), offsets[1] - offsets[0], ::cuda::no_init};
            ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{data + offsets[0], source.size()}, source);
            if (convolution) {
                result = neural::TensorView{destination.data(), shape[0], shape[2], shape[3], shape[1], scalar};
                neural::kernels::convert_layout(runtime.stream, result.data, source.data(), result.n, result.h * result.w, result.c, int(source_scalar), int(scalar));
            } else if (transpose) {
                neural::kernels::convert_layout(runtime.stream, result.data, source.data(), 1, shape[1], shape[0], int(source_scalar), int(scalar));
                result.w = shape[1];
                result.c = shape[0];
            } else neural::kernels::convert(runtime.stream, result.data, source.data(), count, int(source_scalar), int(scalar));
        }
        return result;
    }

    neural::Linear Checkpoint::linear(const std::string& prefix, const neural::Scalar scalar, const bool bias) {
        return {tensor(prefix + ".weight", scalar), bias ? tensor(prefix + ".bias", scalar) : neural::TensorView{}};
    }

    neural::Linear Checkpoint::qkv(const std::span<const std::string> prefixes, const neural::Scalar scalar, const bool bias) {
        neural::Linear result;
        for (const bool is_bias : {false, true}) {
            if (is_bias && !bias) continue;
            const std::string suffix = is_bias ? ".bias" : ".weight";
            const auto shape         = index.at(prefixes[0] + suffix).at("shape").get<std::vector<int>>();
            const std::size_t count  = std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies<>{});
            auto& destination        = weights.storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), count * prefixes.size() * (scalar == neural::Scalar::f32 ? 4uz : 2uz), ::cuda::no_init);
            auto packed              = ::cuda::device_buffer<std::byte>{runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), scalar == neural::Scalar::f16 ? 0uz : count * prefixes.size() * 2, ::cuda::no_init};
            std::byte* target        = scalar == neural::Scalar::f16 ? destination.data() : packed.data();
            for (int i = 0; i < prefixes.size(); ++i) {
                const auto offset = index.at(prefixes[i] + suffix).at("data_offsets")[0].get<std::size_t>();
                ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::byte>{data + offset, count * 2}, ::cuda::std::span<std::byte>{target + i * count * 2, count * 2});
            }
            if (scalar != neural::Scalar::f16) neural::kernels::convert(runtime.stream, destination.data(), packed.data(), count * prefixes.size(), 1, static_cast<int>(scalar));
            if (is_bias) result.bias = neural::TensorView{destination.data(), 1, 1, 1, shape[0] * static_cast<int>(prefixes.size()), scalar};
            else result.weight = neural::TensorView{destination.data(), 1, 1, shape[0] * static_cast<int>(prefixes.size()), shape[1], scalar};
        }
        return result;
    }

    neural::Norm Checkpoint::norm(const std::string& prefix, const neural::Scalar scalar, const float epsilon) {
        return {tensor(prefix + ".weight", scalar), tensor(prefix + ".bias", scalar), epsilon};
    }
    neural::Conv Checkpoint::convolution(const std::string& prefix, const neural::Scalar scalar, const int stride, const int padding) {
        return {tensor(prefix + ".weight", scalar, true), tensor(prefix + ".bias", scalar), stride, padding};
    }
} // namespace physica::generative::sdxl
