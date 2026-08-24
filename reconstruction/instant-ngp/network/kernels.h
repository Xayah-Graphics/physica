#ifndef PHYSICA_RECONSTRUCTION_INSTANT_NGP_NETWORK_KERNELS_H
#define PHYSICA_RECONSTRUCTION_INSTANT_NGP_NETWORK_KERNELS_H

#include <cublasLt.h>
#include <physica/cuda_stream.h>
#include <cstdint>

namespace physica::reconstruction::instant_ngp::kernels {
struct NetworkKernelShape final {
    std::uint32_t grid_level_count;
    std::uint32_t grid_features_per_level;
    std::uint32_t grid_base_resolution;
    std::uint32_t grid_resolution_scale;
    std::uint32_t grid_log2_hashmap_size;
    std::uint32_t direction_degree;
    std::uint32_t mlp_width;
    std::uint32_t density_hidden_layer_count;
    std::uint32_t color_hidden_layer_count;
    std::uint32_t density_output_width;
    std::uint32_t network_output_width;
    std::uint32_t training_batch_size;

    constexpr bool operator==(const NetworkKernelShape&) const = default;
};

template <NetworkKernelShape Shape>
struct NetworkKernels final {
    static void initialize_cublaslt_matmul_plans(cublasLtHandle_t cublaslt_handle, cublasLtMatmulDesc_t* operation_descriptors, cublasLtMatrixLayout_t* a_descriptors, cublasLtMatrixLayout_t* b_descriptors, cublasLtMatrixLayout_t* output_descriptors, cublasLtMatmulAlgo_t* algorithms);
    static void initialize_mlp_parameters(::cuda::stream_ref stream, std::uint32_t seed, float* parameters_full_precision, std::uint16_t* parameters, std::uint16_t* parameter_gradients);
    static void initialize_grid_parameters(::cuda::stream_ref stream, std::uint32_t seed, float* parameters_full_precision, std::uint16_t* parameters, std::uint16_t* parameter_gradients);
    static void upload_trainable_parameters(::cuda::stream_ref stream, const float* parameters_full_precision, float* output_parameters_full_precision, std::uint16_t* output_parameters, std::uint16_t* output_parameter_gradients);
    static void evaluate_network(::cuda::stream_ref stream, std::uint32_t sample_count, const float* sample_coords, const std::uint16_t* parameters, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output);
    static void evaluate_density_network(::cuda::stream_ref stream, std::uint32_t sample_count, const float* sample_coords, const std::uint16_t* parameters, std::uint16_t* density_input, std::uint16_t* density_output);
    static void forward_network(::cuda::stream_ref stream, const float* sample_coords, const std::uint16_t* parameters, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* density_forward_hidden, std::uint16_t* rgb_forward_hidden, std::uint16_t* network_output);
    static void backward_network(::cuda::stream_ref stream, const float* sample_coords, const std::uint16_t* parameters, std::uint16_t* gradients, const std::uint16_t* density_input, const std::uint16_t* rgb_input, const std::uint16_t* density_forward_hidden, const std::uint16_t* rgb_forward_hidden, const std::uint16_t* network_output, const std::uint16_t* network_output_gradients, std::uint16_t* rgb_output_gradients, std::uint16_t* rgb_input_gradients, std::uint16_t* density_input_gradients, std::uint16_t* density_backward_hidden, std::uint16_t* rgb_backward_hidden, cublasLtHandle_t cublaslt_handle, const cublasLtMatmulDesc_t* operation_descriptors, const cublasLtMatrixLayout_t* a_descriptors, const cublasLtMatrixLayout_t* b_descriptors, const cublasLtMatrixLayout_t* output_descriptors, const cublasLtMatmulAlgo_t* algorithms, std::uint8_t* cublaslt_workspace);
    static void step_optimizer(::cuda::stream_ref stream, float* parameters_full_precision, std::uint16_t* parameters, const std::uint16_t* gradients, float* first_moments, float* second_moments, std::uint32_t* parameter_steps);
};
} // namespace physica::reconstruction::instant_ngp::kernels

#endif
