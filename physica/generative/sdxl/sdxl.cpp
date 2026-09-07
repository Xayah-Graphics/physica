module;
#include "../../neural/inference-kernels.h"
#include "kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <physica/cuda.h>

module physica.generative.sdxl;

import std;

namespace physica::generative::sdxl {
    Model::Model(const ::cuda::stream_ref stream, const std::filesystem::path& file, const std::filesystem::path& cache_directory) : runtime{stream, cache_directory}, checkpoint{file, runtime, weights}, clip_l{checkpoint, false}, clip_g{checkpoint, true}, unet{checkpoint}, vae{checkpoint}, training{stream, ::cuda::device_default_memory_pool(stream.device()), 1000, ::cuda::no_init} {
        const ClipWorkspaceLayout layout{77};
        ::cuda::device_buffer<std::byte> workspace{stream, ::cuda::device_default_memory_pool(stream.device()), layout.bytes, ::cuda::no_init};
        const Workspace scratch = layout.view(workspace.data());
        kernels::training_sigmas(stream, training.data());
        clip_l.prepare_empty(runtime, scratch);
        clip_g.prepare_empty(runtime, scratch);
        stream.sync();
    }

    Output::Output(const ::cuda::stream_ref stream, const int w, const int h) : pixels{stream, ::cuda::pinned_default_memory_pool(), std::size_t(w) * h * 3, ::cuda::no_init}, latent{stream, ::cuda::pinned_default_memory_pool(), std::size_t(w / 8) * (h / 8) * 4, ::cuda::no_init}, width{w}, height{h} {}

    Inference::Inference(Model& network, Parameters options)
        : parameters{std::move(options)}, model{network}, unet_layout{parameters.height / 8, parameters.width / 8, parameters.steps}, vae_layout{parameters.height, parameters.width}, workspace{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device())}, unet{model.runtime.stream, parameters.height / 8, parameters.width / 8}, schedule{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), parameters.steps + 1uz, ::cuda::no_init}, seed{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), 1, ::cuda::no_init}, operator_workspace{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device())}, latent{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), std::size_t(parameters.width / 8) * (parameters.height / 8) * 4, ::cuda::no_init}, step{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), 1, ::cuda::no_init},
          input{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), latent.size() * 2, ::cuda::no_init}, epsilon{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), latent.size() * 2, ::cuda::no_init}, decoder{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), std::size_t(parameters.width) * parameters.height * 256, ::cuda::no_init}, decoded{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), std::size_t(parameters.width) * parameters.height * 3, ::cuda::no_init}, image{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), decoded.size(), ::cuda::no_init}, scaled_latent{model.runtime.stream, ::cuda::device_default_memory_pool(model.runtime.stream.device()), latent.size(), ::cuda::no_init},
          outputs{Output{model.runtime.stream, parameters.width, parameters.height}, Output{model.runtime.stream, parameters.width, parameters.height}} {
        const auto started = std::chrono::steady_clock::now();
        const auto stream  = model.runtime.stream;
        const auto pool    = ::cuda::device_default_memory_pool(stream.device());
        // The sampling and decoder phases reuse the same activation storage.
        auto* activation                   = reinterpret_cast<__half*>(decoder.data());
        const std::size_t current_elements = 2uz * (parameters.height / 8) * (parameters.width / 8) * 640;
        unet.current                       = {activation, current_elements};
        unet.sequence                      = {activation + current_elements, current_elements / 2};
        model.runtime.begin_preparation();
        const Tokens positive = model.tokenizer.encode(parameters.positive, -1);
        const Tokens negative = model.tokenizer.encode(parameters.negative, -1);
        const ClipWorkspaceLayout clip_layout{positive.ids.size() + negative.ids.size()};
        workspace                      = ::cuda::device_buffer<std::byte>{stream, pool, std::max({clip_layout.bytes, unet_layout.bytes, vae_layout.bytes}), ::cuda::no_init};
        const Workspace text_workspace = clip_layout.view(workspace.data());
        const auto l                   = model.clip_l.encode(positive, negative, model.runtime, text_workspace);
        const auto g                   = model.clip_g.encode(positive, negative, model.runtime, text_workspace);
        const int length               = std::max(positive.chunks, negative.chunks) * 77;
        ::cuda::device_buffer<__half> context{stream, pool, 2uz * length * 2048, ::cuda::no_init};
        ::cuda::device_buffer<__half> condition{stream, pool, 2 * 2816, ::cuda::no_init};
        kernels::conditioning(stream, context.data(), l.sequence.data(), g.sequence.data(), positive.chunks, negative.chunks, length);
        const std::array<std::int32_t, 2> lengths{positive.chunks * 77, negative.chunks * 77};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::int32_t>{lengths.data(), lengths.size()}, unet.lengths);
        stream.sync();
        text_seconds                  = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto precompute_started = std::chrono::steady_clock::now();
        const std::array<float, 6> sizes{float(parameters.height), float(parameters.width), 0, 0, float(parameters.height), float(parameters.width)};
        ::cuda::device_buffer<float> geometry_input{stream, pool, sizes.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> geometry{stream, pool, 1536, ::cuda::no_init};
        ::cuda::device_buffer<float> times{stream, pool, std::size_t(parameters.steps), ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{sizes.data(), sizes.size()}, geometry_input);
        kernels::time_embedding(stream, geometry.data(), geometry_input.data(), 6, 256, 0);
        kernels::conditions(stream, condition.data(), g.pooled.data(), geometry.data());
        kernels::prepare_schedule(stream, schedule.data(), times.data(), model.training.data(), parameters.steps);
        model.unet.prepare(unet, {context.data(), 2, 1, length, 2048}, {condition.data(), 2, 1, 1, 2816}, times.data(), parameters.steps, model.runtime, unet_layout.view(workspace.data()));
        stream.sync();
        precompute_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - precompute_started).count();
        ::cuda::fill_bytes(stream, seed, 0u);
        kernels::initialize(stream, latent.data(), input.data(), step.data(), seed.data(), schedule.data(), static_cast<int>(latent.size()));
        denoise();
        decode();
        operator_workspace = model.runtime.finish_preparation();
        neural::check(cudaEventCreate(&begin));
        neural::check(cudaEventCreate(&initialized));
        neural::check(cudaEventCreate(&sampled));
        neural::check(cudaEventCreate(&decoded_event));
        neural::check(cudaEventCreate(&copied));
        neural::check(cudaGraphCreate(&graph, 0));
        neural::check(cudaGraphConditionalHandleCreate(&loop, graph, 1, cudaGraphCondAssignDefault));
        neural::check(cudaStreamBeginCaptureToGraph(stream.get(), graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
        kernels::initialize(stream, latent.data(), input.data(), step.data(), seed.data(), schedule.data(), static_cast<int>(latent.size()));
        neural::check(cudaEventRecordWithFlags(initialized, stream.get(), cudaEventRecordExternal));
        cudaStreamCaptureStatus status;
        const cudaGraphNode_t* dependencies{};
        std::size_t dependency_count{};
        neural::check(cudaStreamGetCaptureInfo(stream.get(), &status, nullptr, nullptr, &dependencies, nullptr, &dependency_count));
        const std::vector<cudaGraphNode_t> prefix{dependencies, dependencies + dependency_count};
        cudaGraph_t captured{};
        neural::check(cudaStreamEndCapture(stream.get(), &captured));
        cudaGraphNodeParams node{};
        node.type               = cudaGraphNodeTypeConditional;
        node.conditional.handle = loop;
        node.conditional.type   = cudaGraphCondTypeWhile;
        node.conditional.size   = 1;
        cudaGraphNode_t while_node{};
        neural::check(cudaGraphAddNode(&while_node, graph, prefix.data(), nullptr, prefix.size(), &node));
        const cudaGraph_t body = node.conditional.phGraph_out[0];
        neural::check(cudaStreamBeginCaptureToGraph(stream.get(), body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
        denoise();
        kernels::advance(stream, step.data(), parameters.steps, loop);
        neural::check(cudaStreamEndCapture(stream.get(), &captured));
        neural::check(cudaStreamBeginCaptureToGraph(stream.get(), graph, &while_node, nullptr, 1, cudaStreamCaptureModeThreadLocal));
        neural::check(cudaEventRecordWithFlags(sampled, stream.get(), cudaEventRecordExternal));
        decode();
        neural::kernels::layout(stream, scaled_latent.data(), latent.data(), 1, parameters.height / 8, parameters.width / 8, 4, 0, false);
        neural::check(cudaEventRecordWithFlags(decoded_event, stream.get(), cudaEventRecordExternal));
        neural::check(cudaStreamEndCapture(stream.get(), &captured));
        neural::check(cudaGraphInstantiate(&executable, graph, 0));
        cache_hits     = model.runtime.cache_hits;
        cache_misses   = model.runtime.cache_misses;
        tuning_seconds = model.runtime.tuning_seconds;
        std::size_t free_bytes;
        std::size_t total_bytes;
        neural::check(cudaMemGetInfo(&free_bytes, &total_bytes));
        resident_bytes  = total_bytes - free_bytes;
        prepare_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    Inference::~Inference() {
        model.runtime.stream.sync();
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
        cudaEventDestroy(begin);
        cudaEventDestroy(initialized);
        cudaEventDestroy(sampled);
        cudaEventDestroy(decoded_event);
        cudaEventDestroy(copied);
    }

    const Output& Inference::generate(const std::uint64_t seed) {
        Output& result    = outputs[output_index++ % 2];
        result.seed       = seed;
        const auto stream = model.runtime.stream;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&seed, 1}, this->seed);
        neural::check(cudaEventRecord(begin, stream.get()));
        neural::check(cudaGraphLaunch(executable, stream.get()));
        ::cuda::copy_bytes(stream, image, result.pixels);
        ::cuda::copy_bytes(stream, scaled_latent, result.latent);
        neural::check(cudaEventRecord(copied, stream.get()));
        neural::check(cudaEventSynchronize(copied));
        float milliseconds;
        neural::check(cudaEventElapsedTime(&milliseconds, begin, initialized));
        result.initialize_seconds = milliseconds * 0.001;
        neural::check(cudaEventElapsedTime(&milliseconds, initialized, sampled));
        result.sample_seconds = milliseconds * 0.001;
        neural::check(cudaEventElapsedTime(&milliseconds, sampled, decoded_event));
        result.decode_seconds = milliseconds * 0.001;
        neural::check(cudaEventElapsedTime(&milliseconds, decoded_event, copied));
        result.transfer_seconds = milliseconds * 0.001;
        return result;
    }

    void Inference::denoise() {
        const int height = parameters.height / 8;
        const int width  = parameters.width / 8;
        model.unet.forward({epsilon.data(), 2, height, width, 4}, {input.data(), 2, height, width, 4}, step.data(), unet, model.runtime, unet_layout.view(workspace.data()));
        kernels::euler(model.runtime.stream, latent.data(), input.data(), epsilon.data(), schedule.data(), step.data(), parameters.cfg, height * width * 4);
    }

    void Inference::decode() {
        model.vae.forward({decoded.data(), 1, parameters.height, parameters.width, 3, neural::Scalar::bf16}, {latent.data(), 1, parameters.height / 8, parameters.width / 8, 4, neural::Scalar::f32}, {decoder.data(), 1, 1, 1, 1, neural::Scalar::bf16}, model.runtime, vae_layout.view(workspace.data()));
        kernels::pixels(model.runtime.stream, image.data(), decoded.data(), parameters.height * parameters.width * 3);
    }
} // namespace physica::generative::sdxl
