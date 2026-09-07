module;
#include <physica/cuda.h>

export module physica.generative.sdxl.vae;
import std;
import physica.neural.inference_runtime;
import physica.generative.sdxl.workspace;
import physica.generative.sdxl.weights;
import physica.generative.sdxl.unet;

export namespace physica::generative::sdxl {
    struct VaeStage final {
        std::vector<Residual> blocks;
        neural::Conv resize;
    };
    struct VAE final {
        neural::Conv post_quant;
        neural::Conv input;
        Residual middle_input;
        neural::Norm attention_norm;
        neural::Linear qkv;
        neural::Conv projection;
        Residual middle_output;
        std::array<VaeStage, 4> up;
        neural::Norm norm;
        neural::Conv output;

        explicit VAE(Checkpoint& source);
        void forward(neural::TensorView output, neural::TensorView latent, neural::TensorView current, neural::InferenceRuntime& runtime, const Workspace& scratch) const;
    };
} // namespace physica::generative::sdxl
