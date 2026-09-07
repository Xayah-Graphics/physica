#include <cudnn.h>
#include <cudnn_frontend_version.h>
#include <cutlass/version.h>
#include <physica/cuda.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <nlohmann/json.hpp>

import std;
import physica.generative.sdxl;
import physica.generative.sdxl.inputs;
import physica.serialization.safetensors;

namespace {
    const physica::generative::sdxl::Parameters parameters{
        .positive = R"prompt(petite, realistic, photorealistic, black hair, long hair, 
blue serafuku, red neckerchief, blue sailor collar, long sleeves, pleated skirt, blue beret, 
skinny, black thighhighs, slim legs,
1girl, indoors, solo, portrait, looking at viewer, 

(anime CG, realistic painting style), masterpiece, best quality, amazing quality, newest, very aesthetic,newest, highres, year 2025, high resolution, excellent, medium resolution, clean coloring,soft shading, )prompt",
        .negative = R"prompt(see-through thighhighs, red rope,

worst quality,normal quality,quality,lowres,anatomical nonsense,bad anatomy,bad hands, mutated hands,interlocked fingers,extra fingers,watermark,low resolution, old,transparent,low logo, text, username, signature, early, watermark, signature, jpeg artifacts, username, censored, lowres, logo, text, sketch, multiple views, monochrome, thick thighs, )prompt",
        .width    = 1024,
        .height   = 1536,
        .steps    = 50,
        .cfg      = 4.5F,
    };
    constexpr std::array<std::uint64_t, 1> seeds{659915289870415ull};

    void save(const physica::generative::sdxl::Output& output, const std::filesystem::path& root, const double load_seconds, const physica::generative::sdxl::Inference& inference) {
        const auto started     = std::chrono::steady_clock::now();
        const auto& parameters = inference.parameters;
        const auto directory   = root / std::to_string(output.seed);
        std::filesystem::create_directories(directory);
        const auto image = directory / "image.png";
        if (!stbi_write_png(reinterpret_cast<const char*>(image.u8string().c_str()), output.width, output.height, 3, output.pixels.data(), output.width * 3)) throw std::runtime_error{"SDXL PNG write failed"};
        const std::uint64_t bytes = std::uint64_t(output.width / 8) * (output.height / 8) * 4 * sizeof(float);
        const std::array tensors{physica::serialization::safetensors::TensorView{"latent", "F32", {1, 4, std::uint64_t(output.height / 8), std::uint64_t(output.width / 8)}, output.latent.data(), bytes}};
        physica::serialization::safetensors::write(directory / "latent.safetensors", "sdxl", tensors);
        const double save_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        nlohmann::json run{{"seed", output.seed}, {"width", output.width}, {"height", output.height}, {"steps", parameters.steps}, {"cfg", parameters.cfg}, {"positive", parameters.positive}, {"negative", parameters.negative}, {"sampler", "euler"}, {"scheduler", "simple"}, {"denoise", 1.0}, {"precision", {{"clip", "float16"}, {"unet", "float16"}, {"vae", "bfloat16"}, {"sampling", "float32"}}}, {"toolchain", {{"cuda_runtime", CUDART_VERSION}, {"cudnn", cudnnGetVersion()}, {"cudnn_frontend", CUDNN_FRONTEND_VERSION}, {"cutlass", CUTLASS_VERSION}, {"architecture", "sm_120a"}}}, {"tokenizer", {{"implementation", physica::generative::sdxl::Tokenizer::implementation}, {"vocabulary_sha256", physica::generative::sdxl::Tokenizer::vocabulary_sha256}}},
            {"timing", {{"load_seconds", load_seconds}, {"prepare_seconds", inference.prepare_seconds}, {"text_seconds", inference.text_seconds}, {"precompute_seconds", inference.precompute_seconds}, {"tuning_seconds", inference.tuning_seconds}, {"initialize_seconds", output.initialize_seconds}, {"sample_seconds", output.sample_seconds}, {"decode_seconds", output.decode_seconds}, {"transfer_seconds", output.transfer_seconds}, {"save_seconds", save_seconds}}}, {"latent_scale", 0.13025}, {"latent_layout", "NCHW"}, {"device_memory_used_bytes", inference.resident_bytes}, {"rng", "philox4x32_10_box_muller_v1"}, {"cache", {{"hits", inference.cache_hits}, {"misses", inference.cache_misses}}}, {"sampling_execution", "cuda_graph_while"}};
        std::ofstream report{directory / "run.json"};
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report << run.dump(2) << '\n';
        std::println("SAVE seed={} {:.3f}s", output.seed, save_seconds);
    }
} // namespace

int main(int, char** argv) try {
    const std::filesystem::path checkpoint{argv[1]};
    const std::filesystem::path output{argv[2]};
    const auto started = std::chrono::steady_clock::now();
    std::println("LOAD SDXL / RTX 5090 / CUDA 13.3");
    std::fflush(stdout);
    ::cuda::stream stream{::cuda::devices[0]};
    physica::generative::sdxl::Model model{stream, checkpoint, std::filesystem::absolute(argv[0]).parent_path() / "sdxl-cache"};
    const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::println("LOAD {:.3f}s", load_seconds);
    std::println("PREPARE text, K/V, time conditions, operator plans and CUDA graph");
    std::fflush(stdout);
    physica::generative::sdxl::Inference inference{model, parameters};
    std::println("READY {:.3f}s device memory {:.2f} GiB", inference.prepare_seconds, inference.resident_bytes / double(1ull << 30));
    std::fflush(stdout);
    std::array<std::future<void>, 2> saves;
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        if (saves[i % 2].valid()) saves[i % 2].get();
        const auto& result = inference.generate(seeds[i]);
        std::println("GENERATE seed={} sample={:.3f}s decode={:.3f}s transfer={:.3f}s", result.seed, result.sample_seconds, result.decode_seconds, result.transfer_seconds);
        std::fflush(stdout);
        saves[i % 2] = std::async(std::launch::async, [&result, &output, &inference, load_seconds] { save(result, output, load_seconds, inference); });
    }
    for (auto& task : saves)
        if (task.valid()) task.get();
} catch (const std::exception& error) {
    std::println(stderr, "SDXL: {}", error.what());
    return 1;
}
