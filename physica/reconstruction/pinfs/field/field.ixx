module;

#include <physica/cuda.h>

export module physica.reconstruction.pinfs.field;

import std;
import physica.reconstruction.pinfs.network;
export import physica.math;

export namespace physica::reconstruction::pinfs {
    struct DynamicField final {
        DynamicField(::cuda::stream_ref stream, std::uint32_t capacity, std::uint32_t derivative_capacity, std::uint32_t maximum_derivative_count, std::uint32_t seed, float first_frequency, bool unique_first_frequency);

        DeviceTensor forward(ConstDeviceTensor points);
        void copy_density(ConstDeviceTensor field, float* destination);
        DeviceTensor backward(ConstDeviceTensor adjoints);
        void clear_gradients();
        void step(float learning_rate);
        void set_fading_step(std::uint32_t step, std::uint32_t final_step);
        [[nodiscard]] NetworkState download() const;
        void upload(const NetworkState& state);

    private:
        ::cuda::stream_ref stream;
        DenseNetwork network;
    };

    struct VelocityField final {
        VelocityField(::cuda::stream_ref stream, std::uint32_t capacity, std::uint32_t derivative_capacity, std::uint32_t seed);

        DeviceTensor forward(ConstDeviceTensor points);
        DeviceTensor backward(ConstDeviceTensor adjoints);
        void clear_gradients();
        void step(float learning_rate);
        void set_fading_step(std::uint32_t step, std::uint32_t final_step);
        [[nodiscard]] NetworkState download() const;
        void upload(const NetworkState& state);

    private:
        DenseNetwork network;
    };

    struct StaticFieldOutput final {
        DeviceTensor sdf;
        DeviceTensor color;
        const float* inverse_deviation{};
    };

    struct StaticFieldState final {
        NetworkState sdf;
        NetworkState color;
        float deviation{};
        float deviation_first_moment{};
        float deviation_second_moment{};
        std::uint32_t deviation_step{};
    };

    struct StaticField final {
        StaticField(::cuda::stream_ref stream, std::uint32_t capacity, std::uint32_t derivative_capacity, std::uint32_t seed, std::uint32_t frequency_count);

        DeviceTensor infer_sdf(const Vector3<float>* positions, std::uint32_t sample_count);
        DeviceTensor forward_sdf(const Vector3<float>* positions, std::uint32_t sample_count);
        const float* evaluate_inverse_deviation();
        StaticFieldOutput forward(const Vector3<float>* positions, const Vector3<float>* directions, std::uint32_t sample_count);
        void backward(ConstDeviceTensor color_adjoints, const float* sdf_adjoints, const float* gradient_adjoints, const float* inverse_deviation_adjoint);
        void clear_gradients();
        void step(float learning_rate);
        void set_fading_step(std::uint32_t step, std::uint32_t final_step);
        [[nodiscard]] StaticFieldState download() const;
        void upload(const StaticFieldState& state);

    private:
        ::cuda::stream_ref stream;
        std::uint32_t frequency_count{};
        std::uint32_t encoded_width{};
        DenseNetwork sdf_network;
        DenseNetwork color_network;
        ::cuda::device_buffer<float> encoded_positions;
        ::cuda::device_buffer<float> encoded_position_derivatives;
        ::cuda::device_buffer<float> encoding_weights;
        ::cuda::device_buffer<float> color_inputs;
        ::cuda::device_buffer<float> sdf_adjoints;
        ::cuda::device_buffer<float> sdf_derivative_adjoints;
        ::cuda::device_buffer<float> deviation;
        ::cuda::device_buffer<float> inverse_deviation;
        ::cuda::device_buffer<float> deviation_gradient;
        ::cuda::device_buffer<float> deviation_first_moment;
        ::cuda::device_buffer<float> deviation_second_moment;
        std::uint32_t deviation_step{};
    };
} // namespace physica::reconstruction::pinfs
