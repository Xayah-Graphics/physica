#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <physica/cuda.h>
#include <stb_image_write.h>

import std;
import physica.generative.dataset.cifar10;
import physica.generative.flow_matching;
import physica.generative.flow_matching.sampling;

namespace {
    template <typename Type>
    Type parse_number(const std::string_view text) {
        Type value{};
        std::from_chars(text.data(), text.data() + text.size(), value);
        return value;
    }

    physica::generative::flow_matching::SamplingSolver parse_solver(const std::string_view name) {
        if (name == "euler") return physica::generative::flow_matching::SamplingSolver::euler;
        if (name == "heun") return physica::generative::flow_matching::SamplingSolver::heun;
        if (name == "rk4") return physica::generative::flow_matching::SamplingSolver::rk4;
        throw std::runtime_error{"Unknown ODE solver."};
    }

    std::string_view solver_name(const physica::generative::flow_matching::SamplingSolver solver) {
        switch (solver) {
        case physica::generative::flow_matching::SamplingSolver::euler: return "euler";
        case physica::generative::flow_matching::SamplingSolver::heun: return "heun";
        case physica::generative::flow_matching::SamplingSolver::rk4: return "rk4";
        }
        std::unreachable();
    }

    std::string format_duration(const double seconds) {
        const std::chrono::hh_mm_ss duration{std::chrono::seconds{static_cast<std::uint64_t>(seconds)}};
        return std::format("{:02}:{:02}:{:02}", duration.hours().count(), duration.minutes().count(), duration.seconds().count());
    }

    std::string elapsed_since(const std::chrono::steady_clock::time_point begin) {
        return format_duration(std::chrono::duration<double>{std::chrono::steady_clock::now() - begin}.count());
    }

    void write_png(const std::filesystem::path& path, const physica::generative::flow_matching::SamplingResult& result) {
        stbi_write_png(path.string().c_str(), static_cast<int>(result.width), static_cast<int>(result.height), 4, result.rgba.data(), static_cast<int>(result.width * 4u));
    }

    void write_png(const std::filesystem::path& path, const std::uint32_t width, const std::uint32_t height, const std::span<const std::uint8_t> rgba) {
        stbi_write_png(path.string().c_str(), static_cast<int>(width), static_cast<int>(height), 4, rgba.data(), static_cast<int>(width * 4u));
    }

    std::filesystem::path latest_checkpoint(const std::filesystem::path& output) {
        if (std::filesystem::exists(output / "final.safetensors")) return output / "final.safetensors";
        std::filesystem::path latest;
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator{output / "checkpoints"})
            if (entry.path().extension() == ".safetensors" && (latest.empty() || entry.path().filename() > latest.filename())) latest = entry.path();
        return latest;
    }

    int train(const std::span<char*> arguments) {
        const std::chrono::steady_clock::time_point command_begin = std::chrono::steady_clock::now();
        const std::filesystem::path dataset_directory             = arguments[2];
        const std::filesystem::path output_directory              = arguments[3];
        const std::uint64_t end_step                              = arguments.size() > 4uz ? parse_number<std::uint64_t>(arguments[4]) : 400'000u;
        const bool resume                                         = arguments.size() > 5uz && std::string_view{arguments[5]} == "resume";
        constexpr int device_ordinal                              = 0;
        constexpr std::uint64_t seed                              = 42u;
        std::print("Physica / Flow Matching\n\n"
                   "  mode       train\n"
                   "  dataset    {}\n"
                   "  device     cuda:{}\n"
                   "  target     {} steps\n"
                   "  batch      256\n"
                   "  ema        500000-sample half-life, 0.05 ramp\n"
                   "  seed       {}\n"
                   "  resume     {}\n"
                   "  output     {}\n\n",
            dataset_directory.string(), device_ordinal, end_step, seed, resume ? "yes" : "no", output_directory.string());
        std::println("{}  {:<10} initializing CUDA training runtime", format_duration(0.0), "START");
        std::fflush(stdout);
        std::filesystem::create_directories(output_directory / "checkpoints");
        std::filesystem::create_directories(output_directory / "samples");
        physica::generative::Cifar10TrainingSet dataset{dataset_directory};
        physica::generative::flow_matching::Trainer trainer{dataset, device_ordinal, seed};
        if (resume) {
            const std::filesystem::path checkpoint = latest_checkpoint(output_directory);
            trainer.load(checkpoint);
            std::println("{}  {:<10} restored step {} from {}", elapsed_since(command_begin), "RESUME", trainer.state.step, checkpoint.string());
            std::fflush(stdout);
        }
        std::println("{}  {:<10} step {} -> {}", elapsed_since(command_begin), "READY", trainer.state.step, end_step);
        std::fflush(stdout);
        const bool csv_exists = std::filesystem::exists(output_directory / "training.csv");
        std::ofstream csv{output_directory / "training.csv", std::ios::app};
        if (!csv_exists) std::println(csv, "step,loss,samples_per_second,elapsed_seconds");
        double previous_elapsed_seconds = trainer.state.elapsed_seconds;
        double smoothed_seconds_per_step{};
        while (trainer.state.step < end_step) {
            const std::uint64_t iteration_count                                     = std::min<std::uint64_t>(100u, end_step - trainer.state.step);
            const physica::generative::flow_matching::TrainingStatistics statistics = trainer.optimize(iteration_count);
            const std::uint64_t step                                                = statistics.step;
            const float loss                                                        = statistics.average_loss;
            const double samples_per_second                                         = statistics.samples_per_second;
            const double elapsed_seconds                                            = trainer.state.elapsed_seconds;
            const double seconds_per_step                                           = (elapsed_seconds - previous_elapsed_seconds) / static_cast<double>(iteration_count);
            smoothed_seconds_per_step                                               = smoothed_seconds_per_step == 0.0 ? seconds_per_step : std::lerp(smoothed_seconds_per_step, seconds_per_step, 0.2);
            const double eta_seconds                                                = smoothed_seconds_per_step * static_cast<double>(end_step - step);
            previous_elapsed_seconds                                                = elapsed_seconds;
            std::println(csv, "{},{},{},{}", step, loss, samples_per_second, elapsed_seconds);
            csv.flush();
            std::println("{}  {:<10} {:>7} / {:>7}  {:>6.2f}%  loss {:.6f}  {:>6.1f} samples/s  eta {}", elapsed_since(command_begin), "TRAIN", step, end_step, static_cast<double>(step) * 100.0 / static_cast<double>(end_step), loss, samples_per_second, format_duration(eta_seconds));
            std::fflush(stdout);
            if (step % 1'000u == 0u) {
                const physica::generative::flow_matching::SamplingRequest request;
                const physica::generative::flow_matching::SamplingResult parameter_result           = trainer.sample(request, physica::generative::flow_matching::ParameterSource::parameters);
                const physica::generative::flow_matching::SamplingResult exponential_average_result = trainer.sample(request, physica::generative::flow_matching::ParameterSource::exponential_average);
                const std::filesystem::path parameter_preview_path                                  = output_directory / "samples" / std::format("step-{:06}-parameters.png", step);
                const std::filesystem::path exponential_average_preview_path                        = output_directory / "samples" / std::format("step-{:06}-ema.png", step);
                write_png(parameter_preview_path, parameter_result);
                write_png(exponential_average_preview_path, exponential_average_result);
                std::println("{}  {:<10} step {}  parameters + ema  {} {}  {} NFE", elapsed_since(command_begin), "SAMPLE", step, solver_name(request.solver), request.step_count, parameter_result.nfe);
                std::println("{}  {:<10} {}", elapsed_since(command_begin), "WRITE", parameter_preview_path.string());
                std::println("{}  {:<10} {}", elapsed_since(command_begin), "WRITE", exponential_average_preview_path.string());
                std::fflush(stdout);
            }
            if (step % 50'000u == 0u) {
                const std::filesystem::path checkpoint = output_directory / "checkpoints" / std::format("step-{:06}.safetensors", step);
                trainer.save(checkpoint);
                std::println("{}  {:<10} step {}  {:.1f} MiB  {}", elapsed_since(command_begin), "SAVE", step, static_cast<double>(std::filesystem::file_size(checkpoint)) / (1uz << 20u), checkpoint.string());
                std::fflush(stdout);
            }
        }
        const std::filesystem::path final_checkpoint = output_directory / "final.safetensors";
        trainer.save(final_checkpoint);
        std::println("{}  {:<10} {:.1f} MiB  {}", elapsed_since(command_begin), "SAVE", static_cast<double>(std::filesystem::file_size(final_checkpoint)) / (1uz << 20u), final_checkpoint.string());
        const std::string elapsed = elapsed_since(command_begin);
        std::println("{}  {:<10} step {}  elapsed {}", elapsed, "DONE", trainer.state.step, elapsed);
        std::fflush(stdout);
        return 0;
    }

    int sample(const std::span<char*> arguments) {
        const std::chrono::steady_clock::time_point command_begin = std::chrono::steady_clock::now();
        physica::generative::flow_matching::SamplingRequest request;
        if (arguments.size() > 4uz && std::string_view{arguments[4]} != "all") request.class_index = parse_number<std::uint32_t>(arguments[4]);
        if (arguments.size() > 5uz) request.solver = parse_solver(arguments[5]);
        if (arguments.size() > 6uz) request.step_count = parse_number<std::uint32_t>(arguments[6]);
        if (arguments.size() > 7uz) request.guidance = parse_number<float>(arguments[7]);
        if (arguments.size() > 8uz) request.seed = parse_number<std::uint64_t>(arguments[8]);
        const std::string class_name = request.class_index ? std::format("{}", *request.class_index) : "all";
        std::print("Physica / Flow Matching\n\n"
                   "  mode       sample\n"
                   "  checkpoint {}\n"
                   "  device     cuda:0\n"
                   "  class      {}\n"
                   "  solver     {}\n"
                   "  steps      {}\n"
                   "  guidance   {:.1f}\n"
                   "  seed       {}\n"
                   "  output     {}\n\n",
            arguments[2], class_name, solver_name(request.solver), request.step_count, request.guidance, request.seed, arguments[3]);
        std::println("{}  {:<10} loading EMA checkpoint", format_duration(0.0), "START");
        std::fflush(stdout);
        physica::generative::flow_matching::Sampler sampler{arguments[2], 0};
        std::println("{}  {:<10} checkpoint loaded", elapsed_since(command_begin), "READY");
        std::println("{}  {:<10} generating 100 images", elapsed_since(command_begin), "SAMPLE");
        std::fflush(stdout);
        const physica::generative::flow_matching::SamplingResult result = sampler.sample(request);
        write_png(arguments[3], result);
        std::println("{}  {:<10} {} images  {}x{}  {}", elapsed_since(command_begin), "SAVE", result.labels.size(), result.width, result.height, arguments[3]);
        const std::string elapsed = elapsed_since(command_begin);
        std::println("{}  {:<10} {} NFE  elapsed {}", elapsed, "DONE", result.nfe, elapsed);
        std::fflush(stdout);
        return 0;
    }

    int sample_fid(const std::span<char*> arguments) {
        const std::chrono::steady_clock::time_point command_begin = std::chrono::steady_clock::now();
        const std::filesystem::path output_directory              = arguments[3];
        std::filesystem::create_directories(output_directory);
        physica::generative::flow_matching::SamplingRequest request;
        if (arguments.size() > 4uz) request.solver = parse_solver(arguments[4]);
        if (arguments.size() > 5uz) request.step_count = parse_number<std::uint32_t>(arguments[5]);
        if (arguments.size() > 6uz) request.guidance = parse_number<float>(arguments[6]);
        if (arguments.size() > 7uz) request.seed = parse_number<std::uint64_t>(arguments[7]);
        const std::uint64_t base_seed = request.seed;
        std::print("Physica / Flow Matching\n\n"
                   "  mode       sample-fid\n"
                   "  checkpoint {}\n"
                   "  device     cuda:0\n"
                   "  images     50000\n"
                   "  solver     {}\n"
                   "  steps      {}\n"
                   "  guidance   {:.1f}\n"
                   "  seed       {}\n"
                   "  output     {}\n\n",
            arguments[2], solver_name(request.solver), request.step_count, request.guidance, request.seed, output_directory.string());
        std::println("{}  {:<10} loading EMA checkpoint", format_duration(0.0), "START");
        std::fflush(stdout);
        physica::generative::flow_matching::Sampler sampler{arguments[2], 0};
        std::println("{}  {:<10} checkpoint loaded", elapsed_since(command_begin), "READY");
        std::ofstream manifest{output_directory / "manifest.csv"};
        std::println(manifest, "index,class,seed,solver,steps,nfe,guidance");
        const std::chrono::steady_clock::time_point generation_begin = std::chrono::steady_clock::now();
        std::println("{}  {:<10} generating 50000 images", elapsed_since(command_begin), "START");
        std::fflush(stdout);
        for (std::uint32_t batch = 0u; batch < 500u; ++batch) {
            request.seed                                                    = base_seed + batch;
            const physica::generative::flow_matching::SamplingResult result = sampler.sample(request);
            for (std::uint32_t image = 0u; image < 100u; ++image) {
                std::array<std::uint8_t, 32uz * 32uz * 4uz> pixels{};
                const std::uint32_t grid_x = image % 10u;
                const std::uint32_t grid_y = image / 10u;
                for (std::uint32_t y = 0u; y < 32u; ++y) {
                    const std::size_t source = (static_cast<std::size_t>(grid_y * 32u + y) * result.width + grid_x * 32u) * 4uz;
                    std::ranges::copy_n(result.rgba.begin() + static_cast<std::ptrdiff_t>(source), 32uz * 4uz, pixels.begin() + static_cast<std::ptrdiff_t>(y * 32u * 4u));
                }
                const std::uint32_t index            = batch * 100u + image;
                const std::filesystem::path filename = std::format("{:05}.png", index);
                write_png(output_directory / filename, 32u, 32u, pixels);
                std::println(manifest, "{},{},{},{},{},{},{}", index, static_cast<std::uint32_t>(result.labels[image]), request.seed, solver_name(request.solver), request.step_count, result.nfe, request.guidance);
            }
            manifest.flush();
            const std::uint32_t completed_batches = batch + 1u;
            if (completed_batches % 10u == 0u) {
                const std::uint32_t completed_images = completed_batches * 100u;
                const double generation_seconds      = std::chrono::duration<double>{std::chrono::steady_clock::now() - generation_begin}.count();
                const double images_per_second       = static_cast<double>(completed_images) / generation_seconds;
                const double eta_seconds             = static_cast<double>(50'000u - completed_images) / images_per_second;
                std::println("{}  {:<10} {:>5} / 50000  {:>6.2f}%  {:>6.1f} images/s  eta {}", elapsed_since(command_begin), "FID", completed_images, static_cast<double>(completed_images) / 500.0, images_per_second, format_duration(eta_seconds));
                std::fflush(stdout);
            }
        }
        std::println("{}  {:<10} {}", elapsed_since(command_begin), "SAVE", (output_directory / "manifest.csv").string());
        const std::string elapsed = elapsed_since(command_begin);
        std::println("{}  {:<10} 50000 images  elapsed {}", elapsed, "DONE", elapsed);
        std::fflush(stdout);
        return 0;
    }
} // namespace

int main(const int count, char** const source_arguments) {
    try {
        const std::span<char*> arguments{source_arguments, static_cast<std::size_t>(count)};
        const std::string_view command = arguments[1];
        if (command == "train") return train(arguments);
        if (command == "sample") return sample(arguments);
        if (command == "sample-fid") return sample_fid(arguments);
        throw std::runtime_error{"Unknown Flow Matching command."};
    } catch (const std::exception& exception) {
        std::println(stderr, "{:<12} {}", "ERROR", exception.what());
        std::fflush(stderr);
        return 1;
    }
}
