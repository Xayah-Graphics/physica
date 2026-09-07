module;
#include "../../neural/inference-kernels.h"
#include "kernels.h"
#include <cuda_fp16.h>
#include <physica/cuda.h>

module physica.generative.sdxl.unet;
import std;
import physica.neural.inference_runtime;
import physica.generative.sdxl.workspace;
import physica.generative.sdxl.weights;

namespace physica::generative::sdxl {
    Residual::Residual(Checkpoint& source, const std::string& prefix, const bool timed) {
        const neural::Scalar type = timed ? neural::Scalar::f16 : neural::Scalar::bf16;
        norm1                     = source.norm(prefix + (timed ? ".in_layers.0" : ".norm1"), type, timed ? 1.0e-5F : 1.0e-6F);
        conv1                     = source.convolution(prefix + (timed ? ".in_layers.2" : ".conv1"), type);
        if (timed) time = source.linear(prefix + ".emb_layers.1", type);
        norm2 = source.norm(prefix + (timed ? ".out_layers.0" : ".norm2"), type, timed ? 1.0e-5F : 1.0e-6F);
        conv2 = source.convolution(prefix + (timed ? ".out_layers.3" : ".conv2"), type);
        if (conv1.weight.c != conv1.weight.n) shortcut = source.convolution(prefix + (timed ? ".skip_connection" : ".nin_shortcut"), type, 1, 0);
    }

    void Residual::forward(const neural::TensorView output, const neural::TensorView input, const neural::TensorView projected_time, const int* step, neural::InferenceRuntime& runtime, const Workspace& scratch, const bool prepared_statistics) const {
        const neural::TensorView normalized = scratch.normalized.reshape(input.n, input.h, input.w, input.c, input.scalar);
        const neural::TensorView hidden     = scratch.hidden.reshape(input.n, input.h, input.w, output.c, input.scalar);
        const neural::TensorView activated  = scratch.normalized.reshape(input.n, input.h, input.w, output.c, input.scalar);
        runtime.group_norm(normalized, input, norm1, scratch.statistics, true, prepared_statistics);
        runtime.convolution(hidden, normalized, conv1);
        runtime.group_norm(activated, hidden, norm2, scratch.statistics, true, false, projected_time, step);
        neural::TensorView skip = input;
        if (shortcut.weight.data) {
            skip = scratch.shortcut.reshape(input.n, input.h, input.w, output.c, input.scalar);
            runtime.convolution(skip, input, shortcut);
        }
        runtime.convolution(output, activated, conv2, skip);
    }

    Transformer::Transformer(Checkpoint& source, const std::string& prefix) {
        norm1        = source.norm(prefix + ".norm1", neural::Scalar::f16);
        qkv          = source.qkv(std::array{prefix + ".attn1.to_q", prefix + ".attn1.to_k", prefix + ".attn1.to_v"}, neural::Scalar::f16, false);
        self_output  = source.linear(prefix + ".attn1.to_out.0", neural::Scalar::f16);
        norm2        = source.norm(prefix + ".norm2", neural::Scalar::f16);
        query        = source.linear(prefix + ".attn2.to_q", neural::Scalar::f16, false);
        kv           = source.qkv(std::array{prefix + ".attn2.to_k", prefix + ".attn2.to_v"}, neural::Scalar::f16, false);
        cross_output = source.linear(prefix + ".attn2.to_out.0", neural::Scalar::f16);
        norm3        = source.norm(prefix + ".norm3", neural::Scalar::f16);
        expand       = source.linear(prefix + ".ff.net.0.proj", neural::Scalar::f16);
        contract     = source.linear(prefix + ".ff.net.2", neural::Scalar::f16);
    }

    void Transformer::forward(const neural::TensorView values, const neural::TensorView cross_key, const neural::TensorView cross_value, const std::int32_t* lengths, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        const int width                     = values.c;
        const neural::TensorView normalized = scratch.normalized.reshape(values.n, values.h, values.w, width);
        const neural::TensorView packed     = scratch.hidden.reshape(values.n, values.h, values.w, width * 3);
        const neural::TensorView attended   = scratch.output.reshape(values.n, values.h, values.w, width);
        runtime.layer_norm(normalized, values, norm1);
        runtime.linear(packed, normalized, qkv);
        neural::TensorView k = packed;
        neural::TensorView v = packed;
        k.data               = static_cast<std::byte*>(packed.data) + width * 2;
        v.data               = static_cast<std::byte*>(packed.data) + width * 4;
        runtime.attention(attended, packed, k, v, width / 64, width * 3, width * 3);
        runtime.linear(normalized, attended, self_output);
        neural::kernels::residual_norm(runtime.stream, normalized.data, values.data, normalized.data, norm2.weight.data, norm2.bias.data, values.n * values.h * values.w, width, norm2.epsilon);
        const neural::TensorView q = scratch.hidden.reshape(values.n, values.h, values.w, width);
        runtime.linear(q, normalized, query);
        runtime.attention(attended, q, cross_key, cross_value, width / 64, width, width * 2, false, nullptr, lengths);
        runtime.linear(normalized, attended, cross_output);
        neural::kernels::residual_norm(runtime.stream, normalized.data, values.data, normalized.data, norm3.weight.data, norm3.bias.data, values.n * values.h * values.w, width, norm3.epsilon);
        const neural::TensorView activated = scratch.output.reshape(values.n, values.h, values.w, width * 4);
        runtime.geglu(activated, normalized, expand);
        runtime.linear(values, activated, contract, values);
    }

    SpatialTransformer::SpatialTransformer(Checkpoint& source, const std::string& prefix, const int depth) {
        norm  = source.norm(prefix + ".norm", neural::Scalar::f16, 1.0e-6F);
        input = source.linear(prefix + ".proj_in", neural::Scalar::f16);
        for (int i = 0; i < depth; ++i) blocks.emplace_back(source, prefix + ".transformer_blocks." + std::to_string(i));
        output = source.linear(prefix + ".proj_out", neural::Scalar::f16);
    }

    void SpatialTransformer::forward(const neural::TensorView values, neural::TensorView sequence, const std::span<const std::array<neural::TensorView, 2>> context, const std::int32_t* lengths, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        const neural::TensorView normalized = scratch.normalized.reshape(values.n, values.h, values.w, values.c);
        sequence.n                          = values.n;
        sequence.h                          = values.h;
        sequence.w                          = values.w;
        sequence.c                          = values.c;
        runtime.group_norm(normalized, values, norm, scratch.statistics, false);
        runtime.linear(sequence, normalized, input);
        for (std::size_t i = 0; i < blocks.size(); ++i) blocks[i].forward(sequence, context[i][0], context[i][1], lengths, runtime, scratch);
        runtime.linear(values, sequence, output, values);
    }

    UNetState::UNetState(const ::cuda::stream_ref stream, const int h, const int w) : lengths{stream, ::cuda::device_default_memory_pool(stream.device()), 2, ::cuda::no_init}, height{h}, width{w} {
        skips.emplace_back(stream, ::cuda::device_default_memory_pool(stream.device()), 2uz * h * w * 320, ::cuda::no_init);
        for (int level = 0; level < 3; ++level) {
            for (int i = 0; i < 2; ++i) skips.emplace_back(stream, ::cuda::device_default_memory_pool(stream.device()), 2uz * (h >> level) * (w >> level) * (320 << level), ::cuda::no_init);
            if (level < 2) skips.emplace_back(stream, ::cuda::device_default_memory_pool(stream.device()), 2uz * (h >> (level + 1)) * (w >> (level + 1)) * (320 << level), ::cuda::no_init);
        }
    }

    UNet::UNet(Checkpoint& source) {
        const std::string root = "model.diffusion_model.";
        time_input             = source.linear(root + "time_embed.0", neural::Scalar::f16);
        time_output            = source.linear(root + "time_embed.2", neural::Scalar::f16);
        label_input            = source.linear(root + "label_emb.0.0", neural::Scalar::f16);
        label_output           = source.linear(root + "label_emb.0.2", neural::Scalar::f16);
        input                  = source.convolution(root + "input_blocks.0.0", neural::Scalar::f16);
        const std::array<int, 3> depths{0, 2, 10};
        int index = 1;
        for (int level = 0; level < 3; ++level) {
            for (int i = 0; i < 2; ++i, ++index) {
                const std::string prefix = root + "input_blocks." + std::to_string(index);
                down[level].residuals.emplace_back(source, prefix + ".0", true);
                if (depths[level]) down[level].transformers.emplace_back(source, prefix + ".1", depths[level]);
            }
            if (level < 2) down[level].resize = source.convolution(root + "input_blocks." + std::to_string(index++) + ".0.op", neural::Scalar::f16, 2);
        }
        middle_input  = Residual{source, root + "middle_block.0", true};
        middle        = SpatialTransformer{source, root + "middle_block.1", 10};
        middle_output = Residual{source, root + "middle_block.2", true};
        index         = 0;
        for (int level = 0; level < 3; ++level) {
            for (int i = 0; i < 3; ++i, ++index) {
                const std::string prefix = root + "output_blocks." + std::to_string(index);
                up[level].residuals.emplace_back(source, prefix + ".0", true);
                if (depths[2 - level]) up[level].transformers.emplace_back(source, prefix + ".1", depths[2 - level]);
                if (i == 2 && level < 2) up[level].resize = source.convolution(prefix + ".2.conv", neural::Scalar::f16);
            }
        }
        norm   = source.norm(root + "out.0", neural::Scalar::f16);
        output = source.convolution(root + "out.2", neural::Scalar::f16);
    }

    void UNet::prepare(UNetState& state, const neural::TensorView context, const neural::TensorView condition, const float* times, const int steps, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        auto embeddings                        = ::cuda::device_buffer<__half>{runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), steps * 320uz, ::cuda::no_init};
        const neural::TensorView expanded      = scratch.hidden.reshape(steps, 1, 1, 1280);
        const neural::TensorView time          = scratch.output.reshape(steps, 1, 1, 1280);
        const neural::TensorView label         = scratch.hidden.reshape(2, 1, 1, 1280);
        const neural::TensorView labels_hidden = scratch.normalized.reshape(2, 1, 1, 1280);
        const neural::TensorView summed        = scratch.combined.reshape(steps * 2, 1, 1, 1280);
        kernels::time_embedding(runtime.stream, embeddings.data(), times, steps, 320, 1);
        runtime.linear(expanded, {embeddings.data(), steps, 1, 1, 320}, time_input);
        neural::kernels::activation(runtime.stream, expanded.data, expanded.data, expanded.elements(), 1, 0);
        runtime.linear(time, expanded, time_output);
        runtime.linear(labels_hidden, condition, label_input);
        neural::kernels::activation(runtime.stream, labels_hidden.data, labels_hidden.data, labels_hidden.elements(), 1, 0);
        runtime.linear(label, labels_hidden, label_output);
        kernels::time_condition(runtime.stream, summed.data, time.data, label.data, steps);
        const auto prepare_residual = [&](const Residual& residual) {
            const int width = residual.time.weight.w;
            auto& storage   = state.time_storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), steps * 2uz * width, ::cuda::no_init);
            const neural::TensorView projected{storage.data(), steps * 2, 1, 1, width};
            runtime.linear(projected, summed, residual.time);
            state.time.push_back(projected);
        };
        const auto prepare_transformer = [&](const SpatialTransformer& transformer) {
            for (const auto& block : transformer.blocks) {
                const int width = block.kv.weight.w / 2;
                auto& storage   = state.context_storage.emplace_back(runtime.stream, ::cuda::device_default_memory_pool(runtime.stream.device()), context.n * context.h * context.w * std::size_t(width) * 2, ::cuda::no_init);
                const neural::TensorView packed{storage.data(), context.n, context.h, context.w, width * 2};
                runtime.linear(packed, context, block.kv);
                const std::array pair{neural::TensorView{storage.data(), context.n, context.h, context.w, width}, neural::TensorView{storage.data() + width, context.n, context.h, context.w, width}};
                state.context.push_back(pair);
            }
        };
        for (const auto& stage : down) {
            for (const auto& residual : stage.residuals) prepare_residual(residual);
            for (const auto& transformer : stage.transformers) prepare_transformer(transformer);
        }
        prepare_residual(middle_input);
        prepare_transformer(middle);
        prepare_residual(middle_output);
        for (const auto& stage : up) {
            for (const auto& residual : stage.residuals) prepare_residual(residual);
            for (const auto& transformer : stage.transformers) prepare_transformer(transformer);
        }
    }

    void UNet::forward(const neural::TensorView result, const neural::TensorView model_input, const int* step, UNetState& state, neural::InferenceRuntime& runtime, const Workspace& scratch) const {
        neural::TensorView current{state.skips[0].data(), 2, state.height, state.width, 320};
        const neural::TensorView sequence{state.sequence.data(), 2, 1, 1, 1};
        std::size_t skip_index{};
        std::size_t time_index{};
        std::size_t context_index{};
        std::array<neural::TensorView, 9> skips;
        const auto save_skip = [&] {
            skips[skip_index] = current;
            ++skip_index;
        };
        const auto transform = [&](const SpatialTransformer& transformer) {
            transformer.forward(current, sequence, std::span{state.context}.subspan(context_index, transformer.blocks.size()), state.lengths.data(), runtime, scratch);
            context_index += transformer.blocks.size();
        };
        runtime.convolution(current, model_input, input);
        save_skip();
        for (const auto& stage : down) {
            for (std::size_t i = 0; i < stage.residuals.size(); ++i) {
                neural::TensorView next = current;
                next.data               = state.skips[skip_index].data();
                next.c                  = stage.residuals[i].conv1.weight.n;
                stage.residuals[i].forward(next, current, state.time[time_index++], step, runtime, scratch);
                current = next;
                if (!stage.transformers.empty()) transform(stage.transformers[i]);
                save_skip();
            }
            if (stage.resize.weight.data) {
                const neural::TensorView resized{state.skips[skip_index].data(), 2, current.h / 2, current.w / 2, current.c};
                runtime.convolution(resized, current, stage.resize);
                current = resized;
                save_skip();
            }
        }
        const neural::TensorView middle_result{state.current.data(), 2, current.h, current.w, current.c};
        middle_input.forward(middle_result, current, state.time[time_index++], step, runtime, scratch);
        current = middle_result;
        transform(middle);
        middle_output.forward(current, current, state.time[time_index++], step, runtime, scratch);
        for (const auto& stage : up) {
            for (std::size_t i = 0; i < stage.residuals.size(); ++i) {
                const neural::TensorView skip   = skips[--skip_index];
                const neural::TensorView joined = scratch.combined.reshape(2, current.h, current.w, current.c + skip.c);
                neural::kernels::group_moments(runtime.stream, scratch.statistics, current.data, skip.data, joined.data, nullptr, nullptr, 2, current.h * current.w, current.c + skip.c, current.c, 1);
                current.c = stage.residuals[i].conv1.weight.n;
                stage.residuals[i].forward(current, joined, state.time[time_index++], step, runtime, scratch, true);
                if (!stage.transformers.empty()) transform(stage.transformers[i]);
            }
            if (stage.resize.weight.data) {
                const neural::TensorView resized = scratch.combined.reshape(2, current.h * 2, current.w * 2, current.c);
                neural::kernels::resize(runtime.stream, resized.data, current.data, 2, current.h, current.w, current.c, 1);
                current.h *= 2;
                current.w *= 2;
                runtime.convolution(current, resized, stage.resize);
            }
        }
        const neural::TensorView activated = scratch.normalized.reshape(2, current.h, current.w, current.c);
        runtime.group_norm(activated, current, norm, scratch.statistics, true);
        runtime.convolution(result, activated, output);
    }
} // namespace physica::generative::sdxl
