module;
#include <physica/cuda.h>

#include <nlohmann/json.hpp>

export module physica.generative.sdxl.weights;
import std;
import physica.neural.inference_runtime;

export namespace physica::generative::sdxl {
    struct Weights final {
        std::deque<::cuda::device_buffer<std::byte>> storage;
    };

    struct Checkpoint final {
        neural::InferenceRuntime& runtime;

        Checkpoint(const std::filesystem::path& path, neural::InferenceRuntime& runtime, Weights& weights);
        ~Checkpoint();
        Checkpoint(const Checkpoint&)            = delete;
        Checkpoint& operator=(const Checkpoint&) = delete;
        neural::TensorView tensor(const std::string& name, neural::Scalar scalar, bool convolution = false, bool transpose = false);
        neural::Linear linear(const std::string& prefix, neural::Scalar scalar, bool bias = true);
        neural::Linear qkv(std::span<const std::string> prefixes, neural::Scalar scalar, bool bias);
        neural::Norm norm(const std::string& prefix, neural::Scalar scalar, float epsilon = 1.0e-5F);
        neural::Conv convolution(const std::string& prefix, neural::Scalar scalar, int stride = 1, int padding = 1);

    private:
        void* file{};
        void* mapping{};
        const std::byte* view{};
        const std::byte* data{};
        nlohmann::json index;
        Weights& weights;
    };
} // namespace physica::generative::sdxl
