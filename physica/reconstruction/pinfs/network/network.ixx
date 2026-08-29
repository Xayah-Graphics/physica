module;

#include <cublas_v2.h>
#include <physica/cuda.h>

export module physica.reconstruction.pinfs.network;

import std;

export namespace physica::reconstruction::pinfs {
    enum class Activation : std::uint8_t {
        linear,
        sine,
        relu,
        sigmoid,
        softplus,
        rgb_density,
    };

    enum class Initialization : std::uint8_t {
        uniform,
        normal,
        constant,
    };

    struct LayerConfiguration final {
        std::uint32_t input_width{};
        std::uint32_t output_width{};
        Activation activation{Activation::linear};
        float frequency{1.0F};
        bool append_network_input{};
        float existing_input_scale{1.0F};
        float appended_input_scale{1.0F};
        bool weight_normalization{};
        Initialization weight_initialization{Initialization::uniform};
        float weight_center{};
        float weight_scale{};
        Initialization bias_initialization{Initialization::uniform};
        float bias_center{};
        float bias_scale{};
        std::uint32_t zero_weight_columns_begin{(std::numeric_limits<std::uint32_t>::max)()};
        std::uint32_t blended_hidden_layer_count{};
    };

    struct NetworkConfiguration final {
        std::uint32_t input_width{};
        std::vector<LayerConfiguration> layers;
    };

    struct DeviceTensor final {
        float* values{};
        float* derivatives{};
        std::uint32_t width{};
        std::uint32_t sample_count{};
        std::uint32_t derivative_count{};
    };

    struct ConstDeviceTensor final {
        const float* values{};
        const float* derivatives{};
        std::uint32_t width{};
        std::uint32_t sample_count{};
        std::uint32_t derivative_count{};
    };

    struct NetworkState final {
        std::vector<float> parameters;
        std::vector<float> first_moments;
        std::vector<float> second_moments;
        std::vector<std::uint32_t> layer_steps;
    };

    struct DenseNetwork final {
        DenseNetwork(::cuda::stream_ref stream, NetworkConfiguration configuration, std::uint32_t capacity, std::uint32_t derivative_capacity, std::uint32_t maximum_derivative_count, std::uint32_t seed);
        ~DenseNetwork() noexcept;

        DenseNetwork(const DenseNetwork&)            = delete;
        DenseNetwork& operator=(const DenseNetwork&) = delete;
        DenseNetwork(DenseNetwork&&)                 = delete;
        DenseNetwork& operator=(DenseNetwork&&)      = delete;

        DeviceTensor forward(ConstDeviceTensor input);
        DeviceTensor backward(ConstDeviceTensor output_adjoints);
        void clear_gradients();
        void step(float learning_rate);
        void set_fading_step(std::uint32_t step, std::uint32_t final_step);
        NetworkState download() const;
        void upload(const NetworkState& state);

    private:
        struct Layer final {
            Layer(::cuda::stream_ref stream, LayerConfiguration configuration, std::uint32_t capacity, std::uint32_t derivative_capacity, std::uint32_t maximum_derivative_count);

            LayerConfiguration configuration;
            ::cuda::device_buffer<float> weights;
            ::cuda::device_buffer<float> normalized_weights;
            ::cuda::device_buffer<float> biases;
            ::cuda::device_buffer<float> scales;
            ::cuda::device_buffer<float> weight_gradients;
            ::cuda::device_buffer<float> bias_gradients;
            ::cuda::device_buffer<float> scale_gradients;
            ::cuda::device_buffer<float> normalized_weight_gradients;
            ::cuda::device_buffer<float> weight_first_moments;
            ::cuda::device_buffer<float> bias_first_moments;
            ::cuda::device_buffer<float> scale_first_moments;
            ::cuda::device_buffer<float> weight_second_moments;
            ::cuda::device_buffer<float> bias_second_moments;
            ::cuda::device_buffer<float> scale_second_moments;
            std::uint32_t step{};
            ::cuda::device_buffer<float> inputs;
            ::cuda::device_buffer<float> input_derivatives;
            ::cuda::device_buffer<float> linear;
            ::cuda::device_buffer<float> linear_derivatives;
            ::cuda::device_buffer<float> outputs;
            ::cuda::device_buffer<float> output_derivatives;
        };

        struct CublasDeleter final {
            void operator()(cublasContext* handle) const noexcept;
        };

        ::cuda::stream_ref stream;
        NetworkConfiguration configuration;
        std::uint32_t maximum_width{};
        std::unique_ptr<cublasContext, CublasDeleter> cublas;
        std::vector<Layer> layers;
        ::cuda::device_buffer<float> adjoints_a;
        ::cuda::device_buffer<float> adjoints_b;
        ::cuda::device_buffer<float> derivative_adjoints_a;
        ::cuda::device_buffer<float> derivative_adjoints_b;
        ::cuda::device_buffer<float> network_input_adjoints;
        ::cuda::device_buffer<float> network_input_derivative_adjoints;
        ::cuda::device_buffer<float> blended_inputs;
        ::cuda::device_buffer<float> blended_input_derivatives;
        ::cuda::device_buffer<float> blended_input_adjoints;
        ::cuda::device_buffer<float> blended_input_derivative_adjoints;
        std::vector<float> fading_weights;
        ConstDeviceTensor current_input{};
    };
} // namespace physica::reconstruction::pinfs
