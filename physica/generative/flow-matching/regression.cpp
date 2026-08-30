#include "../../neural/transformer-kernels.h"
#include "kernels.h"
#include <physica/cuda.h>

import std;
import physica.generative.dataset.cifar10;
import physica.generative.flow_matching;
import physica.generative.flow_matching.model;
import physica.generative.flow_matching.sampling;
import physica.neural.matmul;
import physica.neural.training_state;
import physica.neural.transformer;
import physica.serialization.safetensors;

namespace {
    void expect(const bool condition, const std::string_view message) {
        if (!condition) throw std::runtime_error{std::string{message}};
    }

    void test_storage() {
        const std::filesystem::path path = std::filesystem::temp_directory_path() / "physica-flow-matching-storage.safetensors";
        const std::array<float, 4u> floats{1.0F, -2.0F, 3.5F, 4.0F};
        const std::array<std::uint32_t, 2u> steps{3u, 9u};
        const std::array<std::uint64_t, 3u> integers{2u, 4u, 8u};
        const std::array<physica::serialization::safetensors::TensorView, 3u> tensors{
            physica::serialization::safetensors::TensorView{"model.parameters", "F32", {2u, 2u}, floats.data(), sizeof(floats)},
            physica::serialization::safetensors::TensorView{"optimizer.parameter_steps", "U32", {steps.size()}, steps.data(), sizeof(steps)},
            physica::serialization::safetensors::TensorView{"training.state", "U64", {integers.size()}, integers.data(), sizeof(integers)},
        };
        physica::serialization::safetensors::write(path, "flow-matching-test", tensors);
        const physica::serialization::safetensors::File file = physica::serialization::safetensors::read(path);
        expect(file.metadata.at("physica.format_version") == "1", "storage format version");
        expect(!file.metadata.at("physica.project_version").empty(), "storage project version");
        expect(file.metadata.at("physica.system") == "flow-matching-test", "storage system");
        expect(file.tensors[0].shape == std::vector<std::uint64_t>{2u, 2u}, "storage shape");
        expect(std::memcmp(file.tensors[0].data.data(), floats.data(), sizeof(floats)) == 0, "storage float round trip");
        expect(std::memcmp(file.tensors[1].data.data(), steps.data(), sizeof(steps)) == 0, "storage U32 round trip");
        expect(std::memcmp(file.tensors[2].data.data(), integers.data(), sizeof(integers)) == 0, "storage U64 round trip");
        std::filesystem::remove(path);
    }

    void test_rng_and_ode(const ::cuda::stream_ref stream) {
        constexpr std::size_t count = 256uz * 12uz;
        ::cuda::device_buffer<float> first{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init};
        ::cuda::device_buffer<float> second{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init};
        ::cuda::device_buffer<float> different{stream, ::cuda::device_default_memory_pool(stream.device()), count, ::cuda::no_init};
        physica::generative::flow_matching::kernels::make_sampling_noise(stream, first.data(), 42u, 1u);
        physica::generative::flow_matching::kernels::make_sampling_noise(stream, second.data(), 42u, 1u);
        physica::generative::flow_matching::kernels::make_sampling_noise(stream, different.data(), 43u, 1u);
        std::vector<float> first_host(count);
        std::vector<float> second_host(count);
        std::vector<float> different_host(count);
        ::cuda::copy_bytes(stream, first, ::cuda::std::span<float>{first_host.data(), first_host.size()});
        ::cuda::copy_bytes(stream, second, ::cuda::std::span<float>{second_host.data(), second_host.size()});
        ::cuda::copy_bytes(stream, different, ::cuda::std::span<float>{different_host.data(), different_host.size()});
        stream.sync();
        expect(std::memcmp(first_host.data(), second_host.data(), count * sizeof(float)) == 0, "Philox tuple repeatability");
        expect(std::memcmp(first_host.data(), different_host.data(), count * sizeof(float)) != 0, "Philox seed domain separation");

        constexpr std::size_t ode_count = 8uz;
        const std::array<float, ode_count> zeros{};
        const std::array<float, ode_count> ones{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        ::cuda::device_buffer<float> state{stream, ::cuda::device_default_memory_pool(stream.device()), ode_count, ::cuda::no_init};
        ::cuda::device_buffer<float> velocity{stream, ::cuda::device_default_memory_pool(stream.device()), ode_count, ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{zeros.data(), zeros.size()}, state);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{ones.data(), ones.size()}, velocity);
        physica::generative::flow_matching::kernels::euler_step(stream, state.data(), velocity.data(), 0.25F, ode_count);
        std::array<float, ode_count> result{};
        ::cuda::copy_bytes(stream, state, ::cuda::std::span<float>{result.data(), result.size()});
        stream.sync();
        for (const float value : result) expect(value == 0.25F, "Euler constant field");
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{zeros.data(), zeros.size()}, state);
        physica::generative::flow_matching::kernels::heun_step(stream, state.data(), velocity.data(), velocity.data(), 0.25F, ode_count);
        ::cuda::copy_bytes(stream, state, ::cuda::std::span<float>{result.data(), result.size()});
        stream.sync();
        for (const float value : result) expect(value == 0.25F, "Heun constant field");
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{zeros.data(), zeros.size()}, state);
        physica::generative::flow_matching::kernels::rk4_step(stream, state.data(), velocity.data(), velocity.data(), velocity.data(), velocity.data(), 0.25F, ode_count);
        ::cuda::copy_bytes(stream, state, ::cuda::std::span<float>{result.data(), result.size()});
        stream.sync();
        for (const float value : result) expect(value == 0.25F, "RK4 constant field");

        ::cuda::device_buffer<float> intermediate{stream, ::cuda::device_default_memory_pool(stream.device()), ode_count, ::cuda::no_init};
        ::cuda::device_buffer<float> second_stage{stream, ::cuda::device_default_memory_pool(stream.device()), ode_count, ::cuda::no_init};
        ::cuda::device_buffer<float> third_stage{stream, ::cuda::device_default_memory_pool(stream.device()), ode_count, ::cuda::no_init};
        ::cuda::device_buffer<float> fourth_stage{stream, ::cuda::device_default_memory_pool(stream.device()), ode_count, ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{ones.data(), ones.size()}, state);
        physica::generative::flow_matching::kernels::euler_step(stream, state.data(), velocity.data(), 0.25F, ode_count);
        ::cuda::copy_bytes(stream, state, ::cuda::std::span<float>{result.data(), result.size()});
        stream.sync();
        for (const float value : result) expect(value == 1.25F, "Euler linear field");
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{ones.data(), ones.size()}, state);
        physica::generative::flow_matching::kernels::heun_predict(stream, state.data(), velocity.data(), intermediate.data(), 0.25F, ode_count);
        physica::generative::flow_matching::kernels::heun_step(stream, state.data(), velocity.data(), intermediate.data(), 0.25F, ode_count);
        ::cuda::copy_bytes(stream, state, ::cuda::std::span<float>{result.data(), result.size()});
        stream.sync();
        for (const float value : result) expect(value == 1.28125F, "Heun linear field");
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{ones.data(), ones.size()}, state);
        physica::generative::flow_matching::kernels::rk4_intermediate(stream, state.data(), velocity.data(), intermediate.data(), 0.125F, ode_count);
        ::cuda::copy_bytes(stream, intermediate, second_stage);
        physica::generative::flow_matching::kernels::rk4_intermediate(stream, state.data(), second_stage.data(), intermediate.data(), 0.125F, ode_count);
        ::cuda::copy_bytes(stream, intermediate, third_stage);
        physica::generative::flow_matching::kernels::rk4_intermediate(stream, state.data(), third_stage.data(), fourth_stage.data(), 0.25F, ode_count);
        physica::generative::flow_matching::kernels::rk4_step(stream, state.data(), velocity.data(), second_stage.data(), third_stage.data(), fourth_stage.data(), 0.25F, ode_count);
        ::cuda::copy_bytes(stream, state, ::cuda::std::span<float>{result.data(), result.size()});
        stream.sync();
        for (const float value : result) expect(std::abs(value - 1.28401697F) < 1.0e-7F, "RK4 linear field");
    }

    void test_matmul_and_optimizer(const ::cuda::stream_ref stream) {
        physica::neural::MatmulRuntime matmul{stream, {.workspace_byte_count = 64uz * 1024uz * 1024uz, .tuning_byte_count = 8uz * 1024uz * 1024uz, .tuning_bias_byte_count = 64uz * 1024uz}};
        const std::array<float, 6u> a_host{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        const std::array<float, 6u> b_host{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        ::cuda::device_buffer<float> a{stream, ::cuda::device_default_memory_pool(stream.device()), a_host.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> b{stream, ::cuda::device_default_memory_pool(stream.device()), b_host.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> output{stream, ::cuda::device_default_memory_pool(stream.device()), 4uz, ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{a_host.data(), a_host.size()}, a);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{b_host.data(), b_host.size()}, b);
        matmul.execute({a.data(), b.data(), output.data(), 2u, 2u, 3u});
        std::array<float, 4u> output_host{};
        ::cuda::copy_bytes(stream, output, ::cuda::std::span<float>{output_host.data(), output_host.size()});
        stream.sync();
        expect(output_host == std::array<float, 4u>{22.0F, 28.0F, 49.0F, 64.0F}, "TF32 cuBLASLt matmul");
        ::cuda::device_buffer<float> bias_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), 2uz, ::cuda::no_init};
        matmul.execute({a.data(), b.data(), output.data(), 2u, 2u, 3u, true, false, physica::neural::MatmulEpilogue::bias_gradient, bias_gradient.data()});
        std::array<float, 2u> bias_gradient_host{};
        ::cuda::copy_bytes(stream, bias_gradient, ::cuda::std::span<float>{bias_gradient_host});
        stream.sync();
        expect(bias_gradient_host == std::array<float, 2u>{9.0F, 12.0F}, "cuBLASLt bias gradient epilogue");

        std::array<float, 32uz * 32uz> gelu_input_host{};
        std::array<float, 32uz * 32uz> gelu_upstream_host{};
        std::array<float, 32uz * 32uz> gelu_weight_host{};
        std::array<float, 32u> gelu_bias_host{};
        constexpr std::array<float, 5u> gelu_inputs{-4.0F, -1.0F, 0.0F, 1.0F, 4.0F};
        constexpr std::array<float, 5u> gelu_upstreams{0.25F, -0.5F, 1.0F, 0.75F, -0.25F};
        for (std::size_t index = 0uz; index < gelu_input_host.size(); ++index) {
            gelu_input_host[index]    = gelu_inputs[index % gelu_inputs.size()];
            gelu_upstream_host[index] = gelu_upstreams[index % gelu_upstreams.size()];
        }
        for (std::size_t index = 0uz; index < 32uz; ++index) gelu_weight_host[index * 32uz + index] = 1.0F;
        ::cuda::device_buffer<float> gelu_input{stream, ::cuda::device_default_memory_pool(stream.device()), gelu_input_host.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> gelu_upstream{stream, ::cuda::device_default_memory_pool(stream.device()), gelu_upstream_host.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> gelu_weight{stream, ::cuda::device_default_memory_pool(stream.device()), gelu_weight_host.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> gelu_bias{stream, ::cuda::device_default_memory_pool(stream.device()), gelu_bias_host.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> gelu_output{stream, ::cuda::device_default_memory_pool(stream.device()), gelu_input_host.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> gelu_auxiliary{stream, ::cuda::device_default_memory_pool(stream.device()), gelu_input_host.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> gelu_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), gelu_input_host.size(), ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{gelu_input_host}, gelu_input);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{gelu_upstream_host}, gelu_upstream);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{gelu_weight_host}, gelu_weight);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{gelu_bias_host}, gelu_bias);
        matmul.execute({gelu_input.data(), gelu_weight.data(), gelu_output.data(), 32u, 32u, 32u, false, false, physica::neural::MatmulEpilogue::gelu_aux_bias, gelu_bias.data(), 0.0F, gelu_auxiliary.data()});
        matmul.execute({gelu_upstream.data(), gelu_weight.data(), gelu_gradient.data(), 32u, 32u, 32u, false, true, physica::neural::MatmulEpilogue::gelu_gradient, nullptr, 0.0F, gelu_auxiliary.data()});
        std::array<float, gelu_input_host.size()> gelu_output_host{};
        std::array<float, gelu_input_host.size()> gelu_auxiliary_host{};
        std::array<float, gelu_input_host.size()> gelu_gradient_host{};
        ::cuda::copy_bytes(stream, gelu_output, ::cuda::std::span<float>{gelu_output_host});
        ::cuda::copy_bytes(stream, gelu_auxiliary, ::cuda::std::span<float>{gelu_auxiliary_host});
        ::cuda::copy_bytes(stream, gelu_gradient, ::cuda::std::span<float>{gelu_gradient_host});
        stream.sync();
        constexpr float gelu_scale = 0.79788456080286535588F;
        for (std::size_t index = 0uz; index < gelu_input_host.size(); ++index) {
            const float value                = gelu_input_host[index];
            const float cubic                = value * value * value;
            const float hyperbolic_tangent   = std::tanh(gelu_scale * (value + 0.044715F * cubic));
            const float reference_output     = 0.5F * value * (1.0F + hyperbolic_tangent);
            const float reference_derivative = 0.5F * (1.0F + hyperbolic_tangent) + 0.5F * value * (1.0F - hyperbolic_tangent * hyperbolic_tangent) * gelu_scale * (1.0F + 3.0F * 0.044715F * value * value);
            expect(gelu_auxiliary_host[index] == value, "cuBLASLt GELU auxiliary stores preactivation");
            expect(std::abs(gelu_output_host[index] - reference_output) < 2.0e-5F, "cuBLASLt GELU forward epilogue");
            expect(std::abs(gelu_gradient_host[index] - gelu_upstream_host[index] * reference_derivative) < 2.0e-5F, "cuBLASLt GELU backward epilogue");
        }

        constexpr std::uint32_t reduction       = 65'536u;
        constexpr std::size_t reduction_a_count = static_cast<std::size_t>(reduction) * 256uz;
        constexpr std::size_t reduction_b_count = static_cast<std::size_t>(reduction) * 12uz;
        std::vector<float> reduction_a_host(reduction_a_count);
        std::vector<float> reduction_b_host(reduction_b_count);
        for (std::size_t index = 0uz; index < reduction_a_count; ++index) reduction_a_host[index] = static_cast<float>(static_cast<std::int32_t>(index * 17uz % 257uz) - 128) / 128.0F;
        for (std::size_t index = 0uz; index < reduction_b_count; ++index) reduction_b_host[index] = static_cast<float>(static_cast<std::int32_t>(index * 29uz % 251uz) - 125) / 125.0F;
        ::cuda::device_buffer<float> reduction_a{stream, ::cuda::device_default_memory_pool(stream.device()), reduction_a_count, ::cuda::no_init};
        ::cuda::device_buffer<float> reduction_b{stream, ::cuda::device_default_memory_pool(stream.device()), reduction_b_count, ::cuda::no_init};
        ::cuda::device_buffer<float> reduction_first{stream, ::cuda::device_default_memory_pool(stream.device()), 256uz * 12uz, ::cuda::no_init};
        ::cuda::device_buffer<float> reduction_second{stream, ::cuda::device_default_memory_pool(stream.device()), 256uz * 12uz, ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{reduction_a_host.data(), reduction_a_host.size()}, reduction_a);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{reduction_b_host.data(), reduction_b_host.size()}, reduction_b);
        matmul.execute({reduction_a.data(), reduction_b.data(), reduction_first.data(), 256u, 12u, reduction, true, false});
        matmul.execute({reduction_a.data(), reduction_b.data(), reduction_second.data(), 256u, 12u, reduction, true, false});
        std::array<float, 256uz * 12uz> reduction_first_host{};
        std::array<float, 256uz * 12uz> reduction_second_host{};
        ::cuda::copy_bytes(stream, reduction_first, ::cuda::std::span<float>{reduction_first_host.data(), reduction_first_host.size()});
        ::cuda::copy_bytes(stream, reduction_second, ::cuda::std::span<float>{reduction_second_host.data(), reduction_second_host.size()});
        stream.sync();
        expect(std::memcmp(reduction_first_host.data(), reduction_second_host.data(), sizeof(reduction_first_host)) == 0, "cuBLASLt large reduction determinism");
        matmul.execute({reduction_b.data(), reduction_a.data(), reduction_first.data(), 12u, 256u, reduction, true, false});
        matmul.execute({reduction_b.data(), reduction_a.data(), reduction_second.data(), 12u, 256u, reduction, true, false});
        ::cuda::copy_bytes(stream, reduction_first, ::cuda::std::span<float>{reduction_first_host.data(), reduction_first_host.size()});
        ::cuda::copy_bytes(stream, reduction_second, ::cuda::std::span<float>{reduction_second_host.data(), reduction_second_host.size()});
        stream.sync();
        expect(std::memcmp(reduction_first_host.data(), reduction_second_host.data(), sizeof(reduction_first_host)) == 0, "cuBLASLt transposed large reduction determinism");

        const std::array<float, 2u> initial{1.0F, -2.0F};
        const std::array<float, 2u> gradients{0.25F, -0.5F};
        physica::neural::ParameterBuffer direct{stream, initial.size()};
        direct.initialize(initial);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{gradients.data(), gradients.size()}, direct.gradients);
        direct.step({}, 1u, 0u, 256u);
        const physica::neural::ParameterState first_step = direct.download();
        expect(std::abs(first_step.parameters[0] - 0.9999F) < 1.0e-7F, "AdamW parameter update");
        expect(std::abs(first_step.first_moments[0] - 0.025F) < 1.0e-7F, "AdamW first moment");
        expect(std::abs(first_step.second_moments[0] - 0.0000625F) < 1.0e-9F, "AdamW second moment");
        expect(std::memcmp(first_step.parameters.data(), first_step.ema.data(), initial.size() * sizeof(float)) == 0, "EMA first update follows parameters");
        physica::neural::ParameterBuffer resumed{stream, initial.size()};
        resumed.upload(first_step);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{gradients.data(), gradients.size()}, direct.gradients);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{gradients.data(), gradients.size()}, resumed.gradients);
        direct.step({}, 2u, 256u, 256u);
        resumed.step({}, 2u, 256u, 256u);
        const physica::neural::ParameterState direct_state  = direct.download();
        const physica::neural::ParameterState resumed_state = resumed.download();
        expect(std::memcmp(direct_state.parameters.data(), resumed_state.parameters.data(), initial.size() * sizeof(float)) == 0, "AdamW resume parameters");
        expect(std::memcmp(direct_state.ema.data(), resumed_state.ema.data(), initial.size() * sizeof(float)) == 0, "EMA resume");

        physica::neural::ParameterBuffer long_run{stream, initial.size()};
        long_run.initialize(initial);
        ::cuda::device_buffer<std::uint64_t> device_step{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init};
        ::cuda::device_buffer<std::uint64_t> device_processed_samples{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init};
        std::uint64_t first_device_step = 1u;
        std::uint64_t initial_processed_samples{};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&first_device_step, 1uz}, device_step);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint64_t>{&initial_processed_samples, 1uz}, device_processed_samples);
        std::array<float, 2u> reference_parameters = initial;
        std::array<float, 2u> reference_first_moments{};
        std::array<float, 2u> reference_second_moments{};
        std::array<float, 2u> reference_ema = initial;
        std::array<float, 2u> step_gradients{};
        constexpr physica::neural::TrainingConfiguration optimization{};
        for (std::uint64_t step = 1u; step <= 20'000u; ++step) {
            step_gradients[0] = std::sin(static_cast<float>(step) * 0.013F) * 0.25F;
            step_gradients[1] = std::cos(static_cast<float>(step) * 0.017F) * -0.5F;
            ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{step_gradients.data(), step_gradients.size()}, long_run.gradients);
            long_run.step(optimization, device_step.data(), device_processed_samples.data(), 256u);
            physica::generative::flow_matching::kernels::advance_training_state(stream, device_step.data(), device_processed_samples.data(), 256u);
            const float first_correction          = 1.0F - std::pow(optimization.first_decay, static_cast<float>(step));
            const float second_correction         = 1.0F - std::pow(optimization.second_decay, static_cast<float>(step));
            const std::uint64_t processed_samples = (step - 1u) * 256u;
            const float half_life                 = std::min(static_cast<float>(optimization.exponential_average.half_life_samples), static_cast<float>(processed_samples) * optimization.exponential_average.ramp_up_ratio);
            const float exponential_average_decay = processed_samples == 0u ? 0.0F : std::exp2(-256.0F / half_life);
            for (std::size_t index = 0uz; index < initial.size(); ++index) {
                reference_first_moments[index]  = std::fma(1.0F - optimization.first_decay, step_gradients[index], optimization.first_decay * reference_first_moments[index]);
                reference_second_moments[index] = std::fma(1.0F - optimization.second_decay, step_gradients[index] * step_gradients[index], optimization.second_decay * reference_second_moments[index]);
                const float corrected           = reference_first_moments[index] / first_correction / (std::sqrt(reference_second_moments[index] / second_correction) + optimization.epsilon);
                reference_parameters[index] -= optimization.learning_rate * std::fma(optimization.weight_decay, reference_parameters[index], corrected);
                reference_ema[index] = std::fma(1.0F - exponential_average_decay, reference_parameters[index], exponential_average_decay * reference_ema[index]);
            }
        }
        const physica::neural::ParameterState long_run_state = long_run.download();
        for (std::size_t index = 0uz; index < initial.size(); ++index) {
            expect(std::abs(long_run_state.parameters[index] - reference_parameters[index]) < 2.0e-5F, "AdamW long-run parameter");
            expect(std::abs(long_run_state.first_moments[index] - reference_first_moments[index]) < 2.0e-6F, "AdamW long-run first moment");
            expect(std::abs(long_run_state.second_moments[index] - reference_second_moments[index]) < 2.0e-6F, "AdamW long-run second moment");
            expect(std::abs(long_run_state.ema[index] - reference_ema[index]) < 2.0e-5F, "EMA long-run state");
        }
    }

    void test_sdpa(const ::cuda::stream_ref stream) {
        constexpr std::uint32_t sequence   = 37u;
        constexpr std::uint32_t width      = 32u;
        constexpr std::size_t qkv_count    = static_cast<std::size_t>(sequence) * 3uz * width;
        constexpr std::size_t output_count = static_cast<std::size_t>(sequence) * width;
        std::vector<float> qkv_host(qkv_count);
        std::vector<float> gradient_host(output_count);
        for (std::size_t index = 0uz; index < qkv_count; ++index) qkv_host[index] = std::sin(static_cast<float>(index) * 0.013F) * 0.1F;
        for (std::size_t index = 0uz; index < output_count; ++index) gradient_host[index] = std::cos(static_cast<float>(index) * 0.017F) * 0.05F;
        ::cuda::device_buffer<float> qkv{stream, ::cuda::device_default_memory_pool(stream.device()), qkv_count, ::cuda::no_init};
        ::cuda::device_buffer<float> output{stream, ::cuda::device_default_memory_pool(stream.device()), output_count, ::cuda::no_init};
        ::cuda::device_buffer<float> output_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), output_count, ::cuda::no_init};
        ::cuda::device_buffer<float> qkv_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), qkv_count, ::cuda::no_init};
        ::cuda::device_buffer<float> log_sum_exp{stream, ::cuda::device_default_memory_pool(stream.device()), sequence, ::cuda::no_init};
        ::cuda::device_buffer<float> attention_delta{stream, ::cuda::device_default_memory_pool(stream.device()), sequence, ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{qkv_host.data(), qkv_host.size()}, qkv);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{gradient_host.data(), gradient_host.size()}, output_gradient);
        physica::neural::kernels::sdpa_forward(stream, qkv.data(), output.data(), log_sum_exp.data(), 1u, sequence, width, 1u);
        physica::neural::kernels::sdpa_backward(stream, qkv.data(), output.data(), output_gradient.data(), log_sum_exp.data(), attention_delta.data(), qkv_gradient.data(), 1u, sequence, width, 1u);
        std::vector<float> output_host(output_count);
        std::vector<float> qkv_gradient_host(qkv_count);
        ::cuda::copy_bytes(stream, output, ::cuda::std::span<float>{output_host.data(), output_host.size()});
        ::cuda::copy_bytes(stream, qkv_gradient, ::cuda::std::span<float>{qkv_gradient_host.data(), qkv_gradient_host.size()});
        stream.sync();
        constexpr double scale = 0.17677669529663688110;
        std::array<double, sequence> probabilities{};
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::uint32_t key = 0u; key < sequence; ++key) {
            double score{};
            for (std::uint32_t feature = 0u; feature < width; ++feature) score += static_cast<double>(qkv_host[feature]) * qkv_host[static_cast<std::size_t>(key) * 3uz * width + width + feature];
            probabilities[key] = score * scale;
            maximum            = std::max(maximum, probabilities[key]);
        }
        double denominator{};
        for (double& probability : probabilities) denominator += probability = std::exp(probability - maximum);
        for (double& probability : probabilities) probability /= denominator;
        for (std::uint32_t feature = 0u; feature < 4u; ++feature) {
            double reference{};
            for (std::uint32_t key = 0u; key < sequence; ++key) reference += probabilities[key] * qkv_host[static_cast<std::size_t>(key) * 3uz * width + 2uz * width + feature];
            expect(std::abs(static_cast<double>(output_host[feature]) - reference) < 2.0e-5, "fused SDPA forward");
        }
        double delta{};
        for (std::uint32_t feature = 0u; feature < width; ++feature) delta += static_cast<double>(gradient_host[feature]) * output_host[feature];
        std::array<double, width> query_gradient{};
        for (std::uint32_t key = 0u; key < sequence; ++key) {
            double probability_gradient{};
            for (std::uint32_t feature = 0u; feature < width; ++feature) probability_gradient += static_cast<double>(gradient_host[feature]) * qkv_host[static_cast<std::size_t>(key) * 3uz * width + 2uz * width + feature];
            const double score_gradient = probabilities[key] * (probability_gradient - delta) * scale;
            for (std::uint32_t feature = 0u; feature < width; ++feature) query_gradient[feature] += score_gradient * qkv_host[static_cast<std::size_t>(key) * 3uz * width + width + feature];
        }
        for (std::uint32_t feature = 0u; feature < 4u; ++feature) expect(std::abs(static_cast<double>(qkv_gradient_host[feature]) - query_gradient[feature]) < 2.0e-5, "fused SDPA backward dQ");

        std::array<double, width> key_gradient{};
        std::array<double, width> value_gradient{};
        for (std::uint32_t query = 0u; query < sequence; ++query) {
            std::array<double, sequence> query_probabilities{};
            double query_maximum = -std::numeric_limits<double>::infinity();
            for (std::uint32_t key = 0u; key < sequence; ++key) {
                double score{};
                for (std::uint32_t feature = 0u; feature < width; ++feature) score += static_cast<double>(qkv_host[static_cast<std::size_t>(query) * 3uz * width + feature]) * qkv_host[static_cast<std::size_t>(key) * 3uz * width + width + feature];
                query_probabilities[key] = score * scale;
                query_maximum            = std::max(query_maximum, query_probabilities[key]);
            }
            double query_denominator{};
            for (double& probability : query_probabilities) query_denominator += probability = std::exp(probability - query_maximum);
            for (double& probability : query_probabilities) probability /= query_denominator;
            double output_dot{};
            double value_dot{};
            for (std::uint32_t feature = 0u; feature < width; ++feature) {
                const double output_value          = output_host[static_cast<std::size_t>(query) * width + feature];
                const double output_gradient_value = gradient_host[static_cast<std::size_t>(query) * width + feature];
                output_dot += output_gradient_value * output_value;
                value_dot += output_gradient_value * qkv_host[2uz * width + feature];
            }
            const double score_gradient = query_probabilities[0] * (value_dot - output_dot) * scale;
            for (std::uint32_t feature = 0u; feature < width; ++feature) {
                key_gradient[feature] += score_gradient * qkv_host[static_cast<std::size_t>(query) * 3uz * width + feature];
                value_gradient[feature] += query_probabilities[0] * gradient_host[static_cast<std::size_t>(query) * width + feature];
            }
        }
        for (std::uint32_t feature = 0u; feature < 4u; ++feature) {
            expect(std::abs(static_cast<double>(qkv_gradient_host[width + feature]) - key_gradient[feature]) < 2.0e-5, "fused SDPA backward dK");
            expect(std::abs(static_cast<double>(qkv_gradient_host[2uz * width + feature]) - value_gradient[feature]) < 2.0e-5, "fused SDPA backward dV");
        }
    }

    void test_transformer_and_model(const ::cuda::stream_ref stream) {
        {
            constexpr std::array<float, 5u> input_values{-4.0F, -1.0F, 0.0F, 1.0F, 4.0F};
            constexpr std::array<float, 5u> upstream_values{0.25F, -0.5F, 1.0F, 0.75F, -0.25F};
            ::cuda::device_buffer<float> input{stream, ::cuda::device_default_memory_pool(stream.device()), input_values.size(), ::cuda::no_init};
            ::cuda::device_buffer<float> upstream{stream, ::cuda::device_default_memory_pool(stream.device()), upstream_values.size(), ::cuda::no_init};
            ::cuda::device_buffer<float> output{stream, ::cuda::device_default_memory_pool(stream.device()), input_values.size(), ::cuda::no_init};
            ::cuda::device_buffer<float> input_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), input_values.size(), ::cuda::no_init};
            ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{input_values.data(), input_values.size()}, input);
            ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{upstream_values.data(), upstream_values.size()}, upstream);
            physica::generative::flow_matching::kernels::silu_forward(stream, input.data(), output.data(), input_values.size());
            physica::generative::flow_matching::kernels::silu_backward(stream, input.data(), upstream.data(), input_gradient.data(), input_values.size());
            std::array<float, input_values.size()> output_host{};
            std::array<float, input_values.size()> input_gradient_host{};
            ::cuda::copy_bytes(stream, output, ::cuda::std::span<float>{output_host});
            ::cuda::copy_bytes(stream, input_gradient, ::cuda::std::span<float>{input_gradient_host});
            stream.sync();
            for (std::size_t index = 0uz; index < input_values.size(); ++index) {
                const float value   = input_values[index];
                const float sigmoid = 1.0F / (1.0F + std::exp(-value));
                expect(std::abs(output_host[index] - value * sigmoid) < 1.0e-6F, "SiLU forward");
                expect(std::abs(input_gradient_host[index] - upstream_values[index] * (sigmoid + value * sigmoid * (1.0F - sigmoid))) < 1.0e-6F, "SiLU backward");
            }
        }
        physica::neural::MatmulRuntime matmul{stream, {.workspace_byte_count = 64uz * 1024uz * 1024uz, .tuning_byte_count = 8uz * 1024uz * 1024uz, .tuning_bias_byte_count = 64uz * 1024uz}};
        constexpr std::uint32_t batch = 16u;
        constexpr physica::neural::TransformerConfiguration configuration{256u, 32u, 1u, 1u, 64u};
        physica::neural::Transformer transformer{matmul, configuration};
        physica::neural::TransformerWorkspaceLayout layout{configuration, batch};
        expect(physica::neural::StaticTransformerWorkspace<configuration, batch>::byte_count == layout.byte_count, "consteval Transformer workspace layout");
        std::vector<float> parameter_values                                                            = transformer.initialize_parameters(7u);
        parameter_values[transformer.parameters.blocks[0].modulation_bias + 2uz * configuration.width] = 1.0F;
        parameter_values[transformer.parameters.blocks[0].modulation_bias + 5uz * configuration.width] = 1.0F;
        physica::neural::ParameterBuffer parameters{stream, parameter_values.size()};
        parameters.initialize(parameter_values);
        constexpr std::size_t token_count = static_cast<std::size_t>(batch) * configuration.sequence * configuration.width;
        std::vector<float> token_values(token_count);
        std::vector<float> condition_values(static_cast<std::size_t>(batch) * configuration.width);
        for (std::size_t index = 0uz; index < token_values.size(); ++index) token_values[index] = std::sin(static_cast<float>(index) * 0.01F);
        for (std::size_t index = 0uz; index < condition_values.size(); ++index) condition_values[index] = std::cos(static_cast<float>(index) * 0.1F);
        ::cuda::device_buffer<float> tokens{stream, ::cuda::device_default_memory_pool(stream.device()), token_count, ::cuda::no_init};
        ::cuda::device_buffer<float> condition{stream, ::cuda::device_default_memory_pool(stream.device()), condition_values.size(), ::cuda::no_init};
        ::cuda::device_buffer<float> output{stream, ::cuda::device_default_memory_pool(stream.device()), token_count, ::cuda::no_init};
        ::cuda::device_buffer<float> output_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), token_count, ::cuda::no_init};
        ::cuda::device_buffer<float> token_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), token_count, ::cuda::no_init};
        ::cuda::device_buffer<float> condition_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), condition_values.size(), ::cuda::no_init};
        ::cuda::device_buffer<std::uint8_t> workspace{stream, ::cuda::device_default_memory_pool(stream.device()), layout.byte_count, ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{token_values.data(), token_values.size()}, tokens);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{condition_values.data(), condition_values.size()}, condition);
        ::cuda::fill_bytes(stream, output_gradient, 0x3fu);
        ::cuda::fill_bytes(stream, condition_gradient, 0u);
        transformer.forward(parameters.parameters.data(), tokens.data(), condition.data(), output.data(), workspace.data(), layout);
        transformer.backward(parameters.parameters.data(), parameters.gradients.data(), tokens.data(), condition.data(), output_gradient.data(), token_gradient.data(), condition_gradient.data(), workspace.data(), layout);
        const physica::neural::ParameterState transformer_state = parameters.download();
        expect(std::ranges::any_of(transformer_state.parameters, [](const float value) { return std::isfinite(value) && value != 0.0F; }), "Transformer forward state");
        std::vector<float> transformer_gradients(parameters.gradients.size());
        ::cuda::copy_bytes(stream, parameters.gradients, ::cuda::std::span<float>{transformer_gradients.data(), transformer_gradients.size()});
        stream.sync();
        expect(std::ranges::any_of(transformer_gradients, [](const float value) { return std::isfinite(value) && value != 0.0F; }), "Transformer parameter gradients");
        for (const physica::neural::TransformerBlockParameterLayout& block : transformer.parameters.blocks)
            for (std::size_t feature = 0uz; feature < configuration.width; ++feature) expect(transformer_gradients[block.qkv_bias + configuration.width + feature] == 0.0F, "Transformer key bias is fixed at zero");

        physica::generative::flow_matching::FlowDiT model{stream, matmul};
        physica::generative::flow_matching::FlowDiTWorkspaceLayout model_layout{batch};
        physica::neural::ParameterBuffer model_parameters{stream, model.parameters.parameter_count};
        model_parameters.initialize(model.initialize_parameters(11u));
        constexpr std::size_t flow_value_count = static_cast<std::size_t>(batch) * 256uz * 12uz;
        ::cuda::device_buffer<float> patches{stream, ::cuda::device_default_memory_pool(stream.device()), flow_value_count, ::cuda::no_init};
        ::cuda::device_buffer<float> targets{stream, ::cuda::device_default_memory_pool(stream.device()), flow_value_count, ::cuda::no_init};
        ::cuda::device_buffer<float> times{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init};
        ::cuda::device_buffer<std::uint8_t> labels{stream, ::cuda::device_default_memory_pool(stream.device()), batch, ::cuda::no_init};
        ::cuda::device_buffer<float> patch_gradient{stream, ::cuda::device_default_memory_pool(stream.device()), flow_value_count, ::cuda::no_init};
        ::cuda::device_buffer<float> loss{stream, ::cuda::device_default_memory_pool(stream.device()), 1uz, ::cuda::no_init};
        ::cuda::device_buffer<std::uint8_t> model_workspace{stream, ::cuda::device_default_memory_pool(stream.device()), model_layout.byte_count, ::cuda::no_init};
        ::cuda::fill_bytes(stream, patches, 0u);
        const std::vector<float> target_values(flow_value_count, 1.0F);
        std::array<float, batch> time{};
        time.fill(0.5F);
        std::array<std::uint8_t, batch> label{};
        for (std::uint32_t index = 0u; index < batch; ++index) label[index] = static_cast<std::uint8_t>(index % 10u);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{target_values.data(), target_values.size()}, targets);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{time.data(), time.size()}, times);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{label.data(), label.size()}, labels);
        model.forward(model_parameters.parameters.data(), patches.data(), times.data(), labels.data(), model_workspace.data(), model_layout);
        std::vector<float> condition_host(static_cast<std::size_t>(batch) * 256uz);
        std::vector<float> condition_activated_host(static_cast<std::size_t>(batch) * 256uz);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{reinterpret_cast<const float*>(model_workspace.data() + model_layout.condition), condition_host.size()}, ::cuda::std::span<float>{condition_host.data(), condition_host.size()});
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{reinterpret_cast<const float*>(model_workspace.data() + model_layout.condition_activated), condition_activated_host.size()}, ::cuda::std::span<float>{condition_activated_host.data(), condition_activated_host.size()});
        stream.sync();
        bool nonlinear_condition{};
        for (std::size_t index = 0uz; index < condition_host.size(); ++index) {
            const float condition_value = condition_host[index];
            const float activated_value = condition_activated_host[index];
            const float reference       = condition_value / (1.0F + std::exp(-condition_value));
            expect(std::abs(activated_value - reference) < 0.01F, "FlowDiT AdaLN condition SiLU");
            nonlinear_condition |= std::abs(activated_value - condition_value) > 0.01F;
        }
        expect(nonlinear_condition, "FlowDiT AdaLN condition is nonlinear");
        model.loss(targets.data(), loss.data(), model_workspace.data(), model_layout);
        model.backward(model_parameters.parameters.data(), model_parameters.gradients.data(), patches.data(), times.data(), labels.data(), patch_gradient.data(), model_workspace.data(), model_layout);
        std::vector<float> velocity_weight_gradient(256uz * 12uz);
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{model_parameters.gradients.data() + model.parameters.velocity_weight, velocity_weight_gradient.size()}, ::cuda::std::span<float>{velocity_weight_gradient.data(), velocity_weight_gradient.size()});
        float patch_weight_gradient{};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const float>{model_parameters.gradients.data(), 1uz}, ::cuda::std::span<float>{&patch_weight_gradient, 1uz});
        float loss_value{};
        ::cuda::copy_bytes(stream, loss, ::cuda::std::span<float>{&loss_value, 1uz});
        stream.sync();
        expect(std::abs(loss_value - 1.0F) < 1.0e-6F, "FlowDiT zero-output initial loss");
        expect(patch_weight_gradient == 0.0F, "FlowDiT zero-initialized backbone gradient");
        expect(std::ranges::any_of(velocity_weight_gradient, [](const float value) { return std::isfinite(value) && value != 0.0F; }), "FlowDiT velocity gradients");

        physica::generative::flow_matching::SamplingRuntime sampling{stream, model};
        for (const physica::generative::flow_matching::SamplingSolver solver : {physica::generative::flow_matching::SamplingSolver::euler, physica::generative::flow_matching::SamplingSolver::heun, physica::generative::flow_matching::SamplingSolver::rk4}) {
            const physica::generative::flow_matching::SamplingResult result = sampling.sample(model_parameters.parameters.data(), {.solver = solver, .step_count = 1u});
            const std::uint32_t expected_nfe                                = solver == physica::generative::flow_matching::SamplingSolver::euler ? 1u : (solver == physica::generative::flow_matching::SamplingSolver::heun ? 2u : 4u);
            expect(result.nfe == expected_nfe, "ODE sampler NFE");
            expect(result.width == 320u && result.height == 320u && result.rgba.size() == 320uz * 320uz * 4uz, "ODE sampler RGBA output");
        }
    }

    void test_training_resume() {
        const std::filesystem::path directory         = std::filesystem::temp_directory_path() / "physica-flow-matching-resume";
        const std::filesystem::path dataset_directory = directory / "cifar-10-batches-bin";
        std::filesystem::create_directories(dataset_directory);
        std::vector<std::uint8_t> batch_bytes(10'000uz * 3073uz);
        for (std::uint32_t image = 0u; image < 10'000u; ++image) {
            batch_bytes[static_cast<std::size_t>(image) * 3073uz] = static_cast<std::uint8_t>(image % 10u);
            for (std::uint32_t element = 0u; element < 3072u; ++element) batch_bytes[static_cast<std::size_t>(image) * 3073uz + 1uz + element] = static_cast<std::uint8_t>((image + element) % 256u);
        }
        for (std::uint32_t batch = 0u; batch < 5u; ++batch) {
            std::ofstream file{dataset_directory / std::format("data_batch_{}.bin", batch + 1u), std::ios::binary};
            file.write(reinterpret_cast<const char*>(batch_bytes.data()), static_cast<std::streamsize>(batch_bytes.size()));
        }
        physica::generative::Cifar10TrainingSet dataset{dataset_directory};
        const std::filesystem::path continuous_path      = directory / "continuous.safetensors";
        const std::filesystem::path midpoint_path        = directory / "midpoint.safetensors";
        const std::filesystem::path resumed_path         = directory / "resumed.safetensors";
        const std::filesystem::path repeated_resume_path = directory / "repeated-resume.safetensors";
        const auto optimize                              = [](physica::generative::flow_matching::Trainer& trainer, const std::uint64_t iterations) {
            const physica::generative::flow_matching::TrainingStatistics statistics = trainer.optimize(iterations);
            expect(std::isfinite(statistics.average_loss) && statistics.samples_per_second > 0.0, "CUDA Graph training statistics");
        };
        {
            physica::generative::flow_matching::Trainer trainer{dataset, 0, 123u};
            optimize(trainer, 1u);
            trainer.save(midpoint_path);
            optimize(trainer, 1u);
            trainer.save(continuous_path);
            trainer.load(midpoint_path);
            optimize(trainer, 1u);
            trainer.save(resumed_path);
            trainer.load(midpoint_path);
            optimize(trainer, 1u);
            trainer.save(repeated_resume_path);
        }
        const physica::serialization::safetensors::File continuous      = physica::serialization::safetensors::read(continuous_path);
        const physica::serialization::safetensors::File resumed         = physica::serialization::safetensors::read(resumed_path);
        const physica::serialization::safetensors::File repeated_resume = physica::serialization::safetensors::read(repeated_resume_path);
        for (std::size_t index = 0uz; index < resumed.tensors.size(); ++index)
            if (resumed.tensors[index].name != "training.state") expect(resumed.tensors[index].data == repeated_resume.tensors[index].data, "repeated resume determinism");
        expect(continuous.tensors.size() == resumed.tensors.size(), "resume tensor count");
        for (std::size_t index = 0uz; index < continuous.tensors.size(); ++index) {
            expect(continuous.tensors[index].name == resumed.tensors[index].name, "resume tensor name");
            if (continuous.tensors[index].name == "training.state") expect(std::memcmp(continuous.tensors[index].data.data(), resumed.tensors[index].data.data(), 3uz * sizeof(std::uint64_t)) == 0, "resume step, samples, and seed");
            else expect(continuous.tensors[index].data == resumed.tensors[index].data, "resume bitwise tensor state");
        }
        std::filesystem::remove_all(directory);
    }
} // namespace

int main() {
    try {
        test_storage();
        {
            ::cuda::stream stream{::cuda::devices[0]};
            std::println("rng and ODE");
            test_rng_and_ode(stream);
            std::println("matmul and optimizer");
            test_matmul_and_optimizer(stream);
            std::println("SDPA");
            test_sdpa(stream);
            std::println("Transformer and FlowDiT");
            test_transformer_and_model(stream);
            stream.sync();
        }
        std::println("CUDA Graph resume");
        test_training_resume();
        std::println("Physica Flow Matching tests passed.");
        return 0;
    } catch (const std::exception& exception) {
        std::println(std::cerr, "{}", exception.what());
        return 1;
    }
}
