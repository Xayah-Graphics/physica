module;
#include <cuda_fp16.h>
#include <physica/cuda.h>

export module physica.generative.sdxl.unet;
import std;
import physica.neural.inference_runtime;
import physica.generative.sdxl.workspace;
import physica.generative.sdxl.weights;

export namespace physica::generative::sdxl {
    struct Residual final {
        neural::Norm norm1;
        neural::Conv conv1;
        neural::Linear time;
        neural::Norm norm2;
        neural::Conv conv2;
        neural::Conv shortcut;

        Residual() = default;
        Residual(Checkpoint& source, const std::string& prefix, bool timed);
        void forward(neural::TensorView output, neural::TensorView input, neural::TensorView projected_time, const int* step, neural::InferenceRuntime& runtime, const Workspace& scratch, bool prepared_statistics = false) const;
    };
    struct Transformer final {
        neural::Norm norm1;
        neural::Linear qkv;
        neural::Linear self_output;
        neural::Norm norm2;
        neural::Linear query;
        neural::Linear kv;
        neural::Linear cross_output;
        neural::Norm norm3;
        neural::Linear expand;
        neural::Linear contract;

        Transformer(Checkpoint& source, const std::string& prefix);
        void forward(neural::TensorView values, neural::TensorView key, neural::TensorView value, const std::int32_t* lengths, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
    struct SpatialTransformer final {
        neural::Norm norm;
        neural::Linear input;
        std::vector<Transformer> blocks;
        neural::Linear output;

        SpatialTransformer() = default;
        SpatialTransformer(Checkpoint& source, const std::string& prefix, int depth);
        void forward(neural::TensorView values, neural::TensorView sequence, std::span<const std::array<neural::TensorView, 2>> context, const std::int32_t* lengths, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
    struct UNetStage final {
        std::vector<Residual> residuals;
        std::vector<SpatialTransformer> transformers;
        neural::Conv resize;
    };
    struct UNetState final {
        ::cuda::std::span<__half> current;
        ::cuda::std::span<__half> sequence;
        std::vector<::cuda::device_buffer<__half>> skips;
        std::vector<::cuda::device_buffer<__half>> time_storage;
        std::vector<neural::TensorView> time;
        std::vector<::cuda::device_buffer<__half>> context_storage;
        std::vector<std::array<neural::TensorView, 2>> context;
        ::cuda::device_buffer<std::int32_t> lengths;
        int height;
        int width;

        UNetState(::cuda::stream_ref stream, int height, int width);
    };
    struct UNet final {
        neural::Linear time_input;
        neural::Linear time_output;
        neural::Linear label_input;
        neural::Linear label_output;
        neural::Conv input;
        std::array<UNetStage, 3> down;
        Residual middle_input;
        SpatialTransformer middle;
        Residual middle_output;
        std::array<UNetStage, 3> up;
        neural::Norm norm;
        neural::Conv output;

        explicit UNet(Checkpoint& source);
        void prepare(UNetState& state, neural::TensorView context, neural::TensorView condition, const float* times, int steps, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
        void forward(neural::TensorView output, neural::TensorView input, const int* step, UNetState& state, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
} // namespace physica::generative::sdxl
