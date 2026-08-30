module;

#include <physica/cuda.h>

export module physica.generative.flow_matching.sampling;

import std;
import physica.generative.flow_matching.model;

export namespace physica::generative::flow_matching {
    enum class SamplingSolver : std::uint8_t {
        euler,
        heun,
        rk4,
    };

    struct SamplingRequest final {
        SamplingSolver solver{SamplingSolver::heun};
        std::uint32_t step_count{25u};
        float guidance{2.0F};
        std::uint64_t seed{42u};
        std::optional<std::uint32_t> class_index;
    };

    struct SamplingResult final {
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t nfe;
        std::vector<std::uint8_t> labels;
        std::vector<std::uint8_t> rgba;
    };

    struct SamplingRuntime final {
        SamplingRuntime(::cuda::stream_ref stream, FlowDiT& model);

        SamplingResult sample(const float* parameters, const SamplingRequest& request);

    private:
        inline static constexpr std::uint32_t batch     = 100u;
        inline static constexpr std::size_t value_count = static_cast<std::size_t>(batch) * 256uz * 12uz;

        ::cuda::stream_ref stream;
        FlowDiT& model;
        FlowDiTWorkspaceLayout model_workspace_layout;
        ::cuda::device_buffer<std::uint8_t> model_workspace;
        ::cuda::device_buffer<float> state;
        ::cuda::device_buffer<float> times;
        ::cuda::device_buffer<std::uint8_t> labels;
        ::cuda::device_buffer<std::uint8_t> null_labels;
        ::cuda::device_buffer<float> conditional_velocity;
        ::cuda::device_buffer<float> unconditional_velocity;
        ::cuda::device_buffer<float> velocity;
        ::cuda::device_buffer<float> intermediate;
        ::cuda::device_buffer<float> first;
        ::cuda::device_buffer<float> second;
        ::cuda::device_buffer<float> third;
        ::cuda::device_buffer<float> fourth;
        ::cuda::device_buffer<std::uint8_t> image_rgba;

        void evaluate(const float* parameters, const float* input, float time, float guidance, float* output);
    };
} // namespace physica::generative::flow_matching
