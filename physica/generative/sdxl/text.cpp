module;
#include "../../neural/inference-kernels.h"
#include "kernels.h"
#include <cuda_fp16.h>
#include <physica/cuda.h>

module physica.generative.sdxl.text;
import std;
import physica.neural.inference_runtime;
import physica.generative.sdxl.workspace;
import physica.generative.sdxl.weights;
import physica.generative.sdxl.inputs;

namespace physica::generative::sdxl {
    Clip::Clip(Checkpoint& source, const bool is_large) : width{is_large ? 1280 : 768}, large{is_large}, empty{source.runtime.stream, ::cuda::device_default_memory_pool(source.runtime.stream.device()), 77uz * width, ::cuda::no_init} {
        const std::string root = large ? "conditioner.embedders.1.model." : "conditioner.embedders.0.transformer.text_model.";
        token                  = source.tensor(root + (large ? "token_embedding.weight" : "embeddings.token_embedding.weight"), neural::Scalar::f16);
        position               = source.tensor(root + (large ? "positional_embedding" : "embeddings.position_embedding.weight"), neural::Scalar::f16);
        for (int index = 0; index < (large ? 32 : 11); ++index) {
            const std::string prefix = root + (large ? "transformer.resblocks." : "encoder.layers.") + std::to_string(index) + ".";
            ClipBlock block;
            block.norm1 = source.norm(prefix + (large ? "ln_1" : "layer_norm1"), neural::Scalar::f16);
            if (large) block.qkv = {source.tensor(prefix + "attn.in_proj_weight", neural::Scalar::f16), source.tensor(prefix + "attn.in_proj_bias", neural::Scalar::f16)};
            else block.qkv = source.qkv(std::array{prefix + "self_attn.q_proj", prefix + "self_attn.k_proj", prefix + "self_attn.v_proj"}, neural::Scalar::f16, true);
            block.projection = source.linear(prefix + (large ? "attn.out_proj" : "self_attn.out_proj"), neural::Scalar::f16);
            block.norm2      = source.norm(prefix + (large ? "ln_2" : "layer_norm2"), neural::Scalar::f16);
            block.expand     = source.linear(prefix + (large ? "mlp.c_fc" : "mlp.fc1"), neural::Scalar::f16);
            block.contract   = source.linear(prefix + (large ? "mlp.c_proj" : "mlp.fc2"), neural::Scalar::f16);
            blocks.push_back(block);
        }
        if (large) {
            final_norm = source.norm(root + "ln_final", neural::Scalar::f16);
            projection = {source.tensor(root + "text_projection", neural::Scalar::f16, false, true), {}};
        }
    }

    void Clip::forward(const neural::TensorView sequence, const std::int32_t* ids, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        kernels::embedding(runtime.stream, static_cast<__half*>(sequence.data), static_cast<const __half*>(token.data), static_cast<const __half*>(position.data), ids, sequence.n * 77, width);
        const neural::TensorView normalized = scratch.normalized.reshape(sequence.n, 1, 77, width, neural::Scalar::f16);
        const neural::TensorView qkv        = scratch.hidden.reshape(sequence.n, 1, 77, width * 3, neural::Scalar::f16);
        const neural::TensorView attention  = scratch.output.reshape(sequence.n, 1, 77, width, neural::Scalar::f16);
        const neural::TensorView expanded   = scratch.hidden.reshape(sequence.n, 1, 77, width * 4, neural::Scalar::f16);
        for (int index = 0; index < (large ? 31 : 11); ++index) {
            const auto& block = blocks[index];
            runtime.layer_norm(normalized, sequence, block.norm1);
            runtime.linear(qkv, normalized, block.qkv);
            neural::TensorView key   = qkv;
            neural::TensorView value = qkv;
            key.data                 = static_cast<__half*>(qkv.data) + width;
            value.data               = static_cast<__half*>(qkv.data) + 2 * width;
            runtime.attention(attention, qkv, key, value, width / 64, width * 3, width * 3, true);
            runtime.linear(sequence, attention, block.projection, sequence);
            runtime.layer_norm(normalized, sequence, block.norm2);
            runtime.linear(expanded, normalized, block.expand);
            neural::kernels::activation(runtime.stream, expanded.data, expanded.data, expanded.elements(), 1, large ? 2 : 1);
            runtime.linear(sequence, expanded, block.contract, sequence);
        }
    }

    void Clip::pool(const neural::TensorView pooled, const neural::TensorView sequence, const std::int32_t* eos, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        // Only the first block's EOS output contributes to pooled conditioning.
        const auto& last                  = blocks.back();
        const neural::TensorView norm     = scratch.normalized.reshape(1, 1, 77, width, neural::Scalar::f16);
        const neural::TensorView kv       = scratch.hidden.reshape(1, 1, 77, width * 2, neural::Scalar::f16);
        const neural::TensorView residual = scratch.output.reshape(1, 1, 1, width, neural::Scalar::f16);
        const neural::TensorView query    = scratch.shortcut.reshape(1, 1, 1, width, neural::Scalar::f16);
        const neural::TensorView attended = scratch.combined.reshape(1, 1, 1, width, neural::Scalar::f16);
        neural::TensorView first          = sequence;
        first.n                           = 1;
        runtime.layer_norm(norm, first, last.norm1);
        neural::Linear q_layer  = last.qkv;
        q_layer.weight.w        = width;
        q_layer.bias.c          = width;
        neural::Linear kv_layer = last.qkv;
        kv_layer.weight.data    = static_cast<__half*>(last.qkv.weight.data) + width * width;
        kv_layer.weight.w       = 2 * width;
        kv_layer.bias.data      = static_cast<__half*>(last.qkv.bias.data) + width;
        kv_layer.bias.c         = 2 * width;
        runtime.linear(kv, norm, kv_layer);
        kernels::gather_rows(runtime.stream, static_cast<__half*>(residual.data), static_cast<const __half*>(norm.data), eos, 1, width);
        runtime.linear(query, residual, q_layer);
        neural::TensorView value = kv;
        value.data               = static_cast<__half*>(kv.data) + width;
        runtime.attention(attended, query, kv, value, width / 64, width, width * 2, true, eos);
        const neural::TensorView single = scratch.normalized.reshape(1, 1, 1, width, neural::Scalar::f16);
        kernels::gather_rows(runtime.stream, static_cast<__half*>(residual.data), static_cast<const __half*>(sequence.data), eos, 1, width);
        runtime.linear(residual, attended, last.projection, residual);
        runtime.layer_norm(single, residual, last.norm2);
        const neural::TensorView hidden = scratch.hidden.reshape(1, 1, 1, width * 4, neural::Scalar::f16);
        runtime.linear(hidden, single, last.expand);
        neural::kernels::activation(runtime.stream, hidden.data, hidden.data, hidden.elements(), 1, 2);
        runtime.linear(residual, hidden, last.contract, residual);
        runtime.layer_norm(single, residual, final_norm);
        runtime.linear(pooled, single, projection);
    }

    void Clip::prepare_empty(neural::InferenceRuntime& runtime, const Workspace& scratch) {
        std::array<std::int32_t, 77> tokens;
        tokens.fill(large ? 0 : 49407);
        tokens[0] = 49406;
        tokens[1] = 49407;
        auto ids  = ::cuda::device_buffer<std::int32_t>{runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), tokens.size(), ::cuda::no_init};
        ::cuda::copy_bytes(runtime.stream, ::cuda::std::span<const std::int32_t>{tokens.data(), tokens.size()}, ids);
        forward(neural::TensorView{empty.data(), 1, 1, 77, width, neural::Scalar::f16}, ids.data(), runtime, scratch);
        runtime.stream.sync();
    }
    Encoding Clip::encode(const Tokens& positive, const Tokens& negative, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        const auto stream       = runtime.stream;
        const auto pool         = ::cuda::device_default_memory_pool(stream.device());
        const std::size_t count = positive.ids.size() + negative.ids.size();
        Encoding result{::cuda::device_buffer<__half>{stream, pool, count * width, ::cuda::no_init}, ::cuda::device_buffer<__half>{stream, pool, large ? 2560uz : 0uz, ::cuda::no_init}};
        ::cuda::host_buffer<std::int32_t> host_ids{stream, ::cuda::pinned_default_memory_pool(), count + 2, ::cuda::no_init};
        ::cuda::host_buffer<float> host_weights{stream, ::cuda::pinned_default_memory_pool(), count, ::cuda::no_init};
        std::ranges::copy(positive.ids, host_ids.data());
        std::ranges::copy(negative.ids, host_ids.data() + positive.ids.size());
        for (std::size_t i = 0; i < count; ++i)
            if (host_ids.data()[i] == -1) host_ids.data()[i] = large ? 0 : 49407;
        host_ids.data()[count]     = static_cast<int>(std::ranges::find(positive.ids, 49407) - positive.ids.begin());
        host_ids.data()[count + 1] = static_cast<int>(std::ranges::find(negative.ids, 49407) - negative.ids.begin());
        std::ranges::copy(positive.weights, host_weights.data());
        std::ranges::copy(negative.weights, host_weights.data() + positive.weights.size());
        ::cuda::device_buffer<std::int32_t> ids{stream, pool, count + 2, ::cuda::no_init};
        ::cuda::device_buffer<float> weights{stream, pool, count, ::cuda::no_init};
        ::cuda::copy_bytes(stream, host_ids, ids);
        ::cuda::copy_bytes(stream, host_weights, weights);
        const neural::TensorView sequence{result.sequence.data(), positive.chunks + negative.chunks, 1, 77, width};
        forward(sequence, ids.data(), runtime, scratch);
        if (large) {
            this->pool({result.pooled.data(), 1, 1, 1, width}, {result.sequence.data(), 1, 1, 77, width}, ids.data() + count, runtime, scratch);
            this->pool({result.pooled.data() + width, 1, 1, 1, width}, {result.sequence.data() + positive.ids.size() * width, 1, 1, 77, width}, ids.data() + count + 1, runtime, scratch);
        }
        kernels::weighted_text(stream, result.sequence.data(), result.sequence.data(), empty.data(), weights.data(), static_cast<int>(count), width);
        return result;
    }
} // namespace physica::generative::sdxl
