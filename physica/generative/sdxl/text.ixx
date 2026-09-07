module;
#include <cuda_fp16.h>
#include <physica/cuda.h>

export module physica.generative.sdxl.text;
import std;
import physica.neural.inference_runtime;
import physica.generative.sdxl.workspace;
import physica.generative.sdxl.weights;
import physica.generative.sdxl.inputs;

export namespace physica::generative::sdxl {
    struct ClipBlock final {
        neural::Norm norm1;
        neural::Linear qkv;
        neural::Linear projection;
        neural::Norm norm2;
        neural::Linear expand;
        neural::Linear contract;
    };
    struct Encoding final {
        ::cuda::device_buffer<__half> sequence;
        ::cuda::device_buffer<__half> pooled;
    };
    struct Clip final {
        int width;
        bool large;
        neural::TensorView token;
        neural::TensorView position;
        std::vector<ClipBlock> blocks;
        neural::Norm final_norm;
        neural::Linear projection;
        ::cuda::device_buffer<__half> empty;

        Clip(Checkpoint& checkpoint, bool large);
        void forward(neural::TensorView sequence, const std::int32_t* ids, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
        void pool(neural::TensorView pooled, neural::TensorView sequence, const std::int32_t* eos, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
        void prepare_empty(neural::InferenceRuntime& runtime, const Workspace& scratch);
        Encoding encode(const Tokens& positive, const Tokens& negative, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
} // namespace physica::generative::sdxl
