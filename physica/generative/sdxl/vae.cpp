module;
#include "../../neural/inference-kernels.h"
#include "kernels.h"
#include <physica/cuda.h>

module physica.generative.sdxl.vae;
import std;
import physica.neural.inference_runtime;
import physica.generative.sdxl.workspace;
import physica.generative.sdxl.weights;
import physica.generative.sdxl.unet;

namespace physica::generative::sdxl {
    VAE::VAE(Checkpoint& source) {
        const std::string root = "first_stage_model.decoder.";
        post_quant             = source.convolution("first_stage_model.post_quant_conv", neural::Scalar::bf16, 1, 0);
        input                  = source.convolution(root + "conv_in", neural::Scalar::bf16);
        middle_input           = Residual{source, root + "mid.block_1", false};
        attention_norm         = source.norm(root + "mid.attn_1.norm", neural::Scalar::bf16, 1.0e-6F);
        qkv                    = source.qkv(std::array{root + "mid.attn_1.q", root + "mid.attn_1.k", root + "mid.attn_1.v"}, neural::Scalar::bf16, true);
        projection             = source.convolution(root + "mid.attn_1.proj_out", neural::Scalar::bf16, 1, 0);
        middle_output          = Residual{source, root + "mid.block_2", false};
        for (int level = 0; level < 4; ++level) {
            const std::string prefix = root + "up." + std::to_string(3 - level);
            for (int i = 0; i < 3; ++i) up[level].blocks.emplace_back(source, prefix + ".block." + std::to_string(i), false);
            if (level < 3) up[level].resize = source.convolution(prefix + ".upsample.conv", neural::Scalar::bf16);
        }
        norm   = source.norm(root + "norm_out", neural::Scalar::bf16, 1.0e-6F);
        output = source.convolution(root + "conv_out", neural::Scalar::bf16);
    }

    void VAE::forward(const neural::TensorView result, const neural::TensorView latent, neural::TensorView current, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        const neural::TensorView scaled    = scratch.normalized.reshape(1, latent.h, latent.w, 4, neural::Scalar::bf16);
        const neural::TensorView quantized = scratch.output.reshape(1, latent.h, latent.w, 4, neural::Scalar::bf16);
        kernels::latent_decode(runtime.stream, scaled.data, static_cast<const float*>(latent.data), static_cast<int>(latent.elements()));
        runtime.convolution(quantized, scaled, post_quant);
        current.n      = 1;
        current.h      = latent.h;
        current.w      = latent.w;
        current.c      = 512;
        current.scalar = neural::Scalar::bf16;
        runtime.convolution(current, quantized, input);
        middle_input.forward(current, current, {}, nullptr, runtime, scratch);
        const neural::TensorView normalized = scratch.normalized.reshape(1, current.h, current.w, 512, neural::Scalar::bf16);
        const neural::TensorView packed     = scratch.hidden.reshape(1, current.h, current.w, 1536, neural::Scalar::bf16);
        const neural::TensorView attended   = scratch.output.reshape(1, current.h, current.w, 512, neural::Scalar::bf16);
        runtime.group_norm(normalized, current, attention_norm, scratch.statistics, false);
        runtime.linear(packed, normalized, qkv);
        neural::TensorView key   = packed;
        neural::TensorView value = packed;
        key.data                 = static_cast<std::byte*>(packed.data) + 1024;
        value.data               = static_cast<std::byte*>(packed.data) + 2048;
        runtime.attention(attended, packed, key, value, 1, 1536, 1536);
        runtime.convolution(current, attended, projection, current);
        middle_output.forward(current, current, {}, nullptr, runtime, scratch);
        for (const auto& stage : up) {
            for (const auto& block : stage.blocks) {
                neural::TensorView next = current;
                next.c                  = block.conv1.weight.n;
                block.forward(next, current, {}, nullptr, runtime, scratch);
                current = next;
            }
            if (stage.resize.weight.data) {
                const neural::TensorView resized = scratch.combined.reshape(1, current.h * 2, current.w * 2, current.c, neural::Scalar::bf16);
                neural::kernels::resize(runtime.stream, resized.data, current.data, 1, current.h, current.w, current.c, 2);
                current.h *= 2;
                current.w *= 2;
                runtime.convolution(current, resized, stage.resize);
            }
        }
        const neural::TensorView activated = scratch.normalized.reshape(1, current.h, current.w, 128, neural::Scalar::bf16);
        runtime.group_norm(activated, current, norm, scratch.statistics, true);
        runtime.convolution(result, activated, output);
    }
} // namespace physica::generative::sdxl
