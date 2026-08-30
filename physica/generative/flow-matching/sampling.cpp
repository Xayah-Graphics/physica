module;

#include "kernels.h"
#include <physica/cuda.h>

module physica.generative.flow_matching.sampling;

import std;
import physica.generative.flow_matching.model;

namespace physica::generative::flow_matching {
    SamplingRuntime::SamplingRuntime(const ::cuda::stream_ref source_stream, FlowDiT& source_model)
        : stream{source_stream}, model{source_model}, model_workspace_layout{batch}, model_workspace{stream, ::cuda::device_default_memory_pool(stream.device()), model_workspace_layout.byte_count, ::cuda::no_init}, state{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, times{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, null_labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init}, conditional_velocity{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, unconditional_velocity{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, velocity{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init},
          intermediate{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, first{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, second{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, third{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, fourth{stream, ::cuda::device_default_memory_pool(stream.device()), value_count, ::cuda::no_init}, image_rgba{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(batch) * 32uz * 32uz * 4uz, ::cuda::no_init} {
        ::cuda::fill_bytes(stream, null_labels, 10u);
    }

    SamplingResult SamplingRuntime::sample(const float* const parameters, const SamplingRequest& request) {
        kernels::make_sampling_noise(stream, state.data(), request.seed, batch);
        kernels::make_labels(stream, labels.data(), batch, request.class_index.value_or(10u));
        const float step_size = 1.0F / static_cast<float>(request.step_count);
        std::uint32_t nfe{};
        for (std::uint32_t step = 0u; step < request.step_count; ++step) {
            const float time = static_cast<float>(step) * step_size;
            if (request.solver == SamplingSolver::euler) {
                evaluate(parameters, state.data(), time, request.guidance, velocity.data());
                kernels::euler_step(stream, state.data(), velocity.data(), step_size, value_count);
                ++nfe;
            } else if (request.solver == SamplingSolver::heun) {
                evaluate(parameters, state.data(), time, request.guidance, first.data());
                kernels::heun_predict(stream, state.data(), first.data(), intermediate.data(), step_size, value_count);
                evaluate(parameters, intermediate.data(), time + step_size, request.guidance, second.data());
                kernels::heun_step(stream, state.data(), first.data(), second.data(), step_size, value_count);
                nfe += 2u;
            } else {
                evaluate(parameters, state.data(), time, request.guidance, first.data());
                kernels::rk4_intermediate(stream, state.data(), first.data(), intermediate.data(), 0.5F * step_size, value_count);
                evaluate(parameters, intermediate.data(), time + 0.5F * step_size, request.guidance, second.data());
                kernels::rk4_intermediate(stream, state.data(), second.data(), intermediate.data(), 0.5F * step_size, value_count);
                evaluate(parameters, intermediate.data(), time + 0.5F * step_size, request.guidance, third.data());
                kernels::rk4_intermediate(stream, state.data(), third.data(), intermediate.data(), step_size, value_count);
                evaluate(parameters, intermediate.data(), time + step_size, request.guidance, fourth.data());
                kernels::rk4_step(stream, state.data(), first.data(), second.data(), third.data(), fourth.data(), step_size, value_count);
                nfe += 4u;
            }
        }

        kernels::unpatchify(stream, state.data(), image_rgba.data(), batch);
        std::vector<std::uint8_t> image_pixels(image_rgba.size());
        std::vector<std::uint8_t> image_labels(batch);
        ::cuda::copy_bytes(stream, image_rgba, ::cuda::std::span<std::uint8_t>{image_pixels.data(), image_pixels.size()});
        ::cuda::copy_bytes(stream, labels, ::cuda::std::span<std::uint8_t>{image_labels.data(), image_labels.size()});
        stream.sync();
        SamplingResult result{
            .width  = 320u,
            .height = 320u,
            .nfe    = nfe,
            .labels = std::move(image_labels),
            .rgba   = std::vector<std::uint8_t>(320uz * 320uz * 4uz),
        };
        for (std::uint32_t image = 0u; image < batch; ++image)
            for (std::uint32_t y = 0u; y < 32u; ++y) {
                const std::uint32_t grid_x    = image % 10u;
                const std::uint32_t grid_y    = image / 10u;
                const std::size_t source      = (static_cast<std::size_t>(image) * 32uz * 32uz + static_cast<std::size_t>(y) * 32uz) * 4uz;
                const std::size_t destination = (static_cast<std::size_t>(grid_y * 32u + y) * result.width + grid_x * 32u) * 4uz;
                std::ranges::copy_n(image_pixels.begin() + static_cast<std::ptrdiff_t>(source), 32uz * 4uz, result.rgba.begin() + static_cast<std::ptrdiff_t>(destination));
            }
        return result;
    }

    void SamplingRuntime::evaluate(const float* const parameters, const float* const input, const float time, const float guidance, float* const output) {
        kernels::make_sampling_time(stream, times.data(), time, batch);
        model.forward(parameters, input, times.data(), labels.data(), model_workspace.data(), model_workspace_layout);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{reinterpret_cast<const float*>(model_workspace.data() + model_workspace_layout.velocity), value_count}, conditional_velocity);
        model.forward(parameters, input, times.data(), null_labels.data(), model_workspace.data(), model_workspace_layout);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{reinterpret_cast<const float*>(model_workspace.data() + model_workspace_layout.velocity), value_count}, unconditional_velocity);
        kernels::combine_guidance(stream, conditional_velocity.data(), unconditional_velocity.data(), output, guidance, value_count);
    }
} // namespace physica::generative::flow_matching
