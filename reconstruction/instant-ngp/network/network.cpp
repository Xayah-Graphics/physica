module;

#include "kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/std/span>

module physica.reconstruction.instant_ngp.network;

import std;

namespace physica::reconstruction::instant_ngp {
template <NetworkShape Shape>
inline constexpr kernels::NetworkKernelShape network_kernel_shape{
    .grid_level_count = Shape.grid.level_count,
    .grid_features_per_level = Shape.grid.features_per_level,
    .grid_base_resolution = Shape.grid.base_resolution,
    .grid_resolution_scale = Shape.grid.resolution_scale,
    .grid_log2_hashmap_size = Shape.grid.log2_hashmap_size,
    .direction_degree = Shape.direction.degree,
    .mlp_width = Shape.density.width,
    .density_hidden_layer_count = Shape.density.hidden_layer_count,
    .color_hidden_layer_count = Shape.color.hidden_layer_count,
    .density_output_width = Shape.density.output_width,
    .network_output_width = Shape.color.output_width,
    .training_batch_size = Shape.training_batch_size,
};

template <NetworkShape Shape>
Network<Shape>::Network(const ::cuda::stream_ref source_stream, const std::uint32_t seed)
    : stream{source_stream},
      parameters_full_precision{stream, ::cuda::device_default_memory_pool(stream.device()), parameter_count, ::cuda::no_init},
      parameters{stream, ::cuda::device_default_memory_pool(stream.device()), parameter_count, ::cuda::no_init},
      gradients{stream, ::cuda::device_default_memory_pool(stream.device()), parameter_count, ::cuda::no_init},
      first_moments{stream, ::cuda::device_default_memory_pool(stream.device()), parameter_count, ::cuda::no_init},
      second_moments{stream, ::cuda::device_default_memory_pool(stream.device()), parameter_count, ::cuda::no_init},
      parameter_steps{stream, ::cuda::device_default_memory_pool(stream.device()), parameter_count, ::cuda::no_init},
      density_input{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(grid_output_width) * Shape.training_batch_size, ::cuda::no_init},
      field_features{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(color_input_width) * Shape.training_batch_size, ::cuda::no_init},
      output{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(Shape.color.output_width) * Shape.inference_capacity, ::cuda::no_init},
      color_output_gradients{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(Shape.color.output_width) * Shape.training_batch_size, ::cuda::no_init},
      color_input_gradients{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(color_input_width) * Shape.training_batch_size, ::cuda::no_init},
      density_input_gradients{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(grid_output_width) * Shape.training_batch_size, ::cuda::no_init},
      density_forward_hidden{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(Shape.density.hidden_layer_count) * Shape.density.width * Shape.training_batch_size, ::cuda::no_init},
      color_forward_hidden{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(Shape.color.hidden_layer_count) * Shape.color.width * Shape.training_batch_size, ::cuda::no_init},
      density_backward_hidden{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(Shape.density.hidden_layer_count) * Shape.density.width * Shape.training_batch_size, ::cuda::no_init},
      color_backward_hidden{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(Shape.color.hidden_layer_count) * Shape.color.width * Shape.training_batch_size, ::cuda::no_init},
      cublaslt_workspace{stream, ::cuda::device_default_memory_pool(stream.device()), static_cast<std::size_t>(64u) * 1024u * 1024u, ::cuda::no_init} {
    const cublasStatus_t status = cublasLtCreate(std::out_ptr(cublaslt_handle));
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cublasLtCreate failed: {}", cublasGetStatusString(status))};
    kernels::NetworkKernels<network_kernel_shape<Shape>>::initialize_cublaslt_matmul_plans(cublaslt_handle.get(), cublaslt_matmul_plans.operation_descriptors.data(), cublaslt_matmul_plans.a_descriptors.data(), cublaslt_matmul_plans.b_descriptors.data(), cublaslt_matmul_plans.output_descriptors.data(), cublaslt_matmul_plans.algorithms.data());
    kernels::NetworkKernels<network_kernel_shape<Shape>>::initialize_mlp_parameters(stream, seed, parameters_full_precision.data(), parameters.data(), gradients.data());
    kernels::NetworkKernels<network_kernel_shape<Shape>>::initialize_grid_parameters(stream, seed, parameters_full_precision.data(), parameters.data(), gradients.data());
    ::cuda::fill_bytes(stream, first_moments, 0u);
    ::cuda::fill_bytes(stream, second_moments, 0u);
    ::cuda::fill_bytes(stream, parameter_steps, 0u);
    stream.sync();
}

template <NetworkShape Shape>
Network<Shape>::~Network() noexcept = default;

template <NetworkShape Shape>
NetworkOutput Network<Shape>::infer(const DeviceSamples samples) {
    kernels::NetworkKernels<network_kernel_shape<Shape>>::evaluate_network(stream, samples.count, reinterpret_cast<const float*>(samples.data), parameters.data(), density_input.data(), field_features.data(), output.data());
    return {.data = output.data(), .count = samples.count};
}

template <NetworkShape Shape>
NetworkOutput Network<Shape>::infer_density(const DeviceSamples samples) {
    kernels::NetworkKernels<network_kernel_shape<Shape>>::evaluate_density_network(stream, samples.count, reinterpret_cast<const float*>(samples.data), parameters.data(), density_input.data(), field_features.data());
    return {.data = field_features.data(), .count = samples.count};
}

template <NetworkShape Shape>
void Network<Shape>::forward(const DeviceSamples samples) {
    kernels::NetworkKernels<network_kernel_shape<Shape>>::forward_network(stream, reinterpret_cast<const float*>(samples.data), parameters.data(), density_input.data(), field_features.data(), density_forward_hidden.data(), color_forward_hidden.data(), output.data());
}

template <NetworkShape Shape>
void Network<Shape>::backward(const DeviceSamples samples, const NetworkGradients source_gradients) {
    kernels::NetworkKernels<network_kernel_shape<Shape>>::backward_network(stream, reinterpret_cast<const float*>(samples.data), parameters.data(), gradients.data(), density_input.data(), field_features.data(), density_forward_hidden.data(), color_forward_hidden.data(), output.data(), source_gradients.data, color_output_gradients.data(), color_input_gradients.data(), density_input_gradients.data(), density_backward_hidden.data(), color_backward_hidden.data(), cublaslt_handle.get(), cublaslt_matmul_plans.operation_descriptors.data(), cublaslt_matmul_plans.a_descriptors.data(), cublaslt_matmul_plans.b_descriptors.data(), cublaslt_matmul_plans.output_descriptors.data(), cublaslt_matmul_plans.algorithms.data(), cublaslt_workspace.data());
}

template <NetworkShape Shape>
void Network<Shape>::step() {
    kernels::NetworkKernels<network_kernel_shape<Shape>>::step_optimizer(stream, parameters_full_precision.data(), parameters.data(), gradients.data(), first_moments.data(), second_moments.data(), parameter_steps.data());
}

template <NetworkShape Shape>
NetworkState Network<Shape>::download() const {
    NetworkState state{
        .parameters = std::vector<float>(parameter_count),
        .first_moments = std::vector<float>(parameter_count),
        .second_moments = std::vector<float>(parameter_count),
        .parameter_steps = std::vector<std::uint32_t>(parameter_count),
    };
    ::cuda::copy_bytes(stream, parameters_full_precision, ::cuda::std::span{state.parameters.data(), state.parameters.size()});
    ::cuda::copy_bytes(stream, first_moments, ::cuda::std::span{state.first_moments.data(), state.first_moments.size()});
    ::cuda::copy_bytes(stream, second_moments, ::cuda::std::span{state.second_moments.data(), state.second_moments.size()});
    ::cuda::copy_bytes(stream, parameter_steps, ::cuda::std::span{state.parameter_steps.data(), state.parameter_steps.size()});
    stream.sync();
    return state;
}

template <NetworkShape Shape>
void Network<Shape>::upload(const NetworkState& state) {
    kernels::NetworkKernels<network_kernel_shape<Shape>>::upload_trainable_parameters(stream, state.parameters.data(), parameters_full_precision.data(), parameters.data(), gradients.data());
    ::cuda::copy_bytes(stream, ::cuda::std::span{state.first_moments.data(), state.first_moments.size()}, first_moments);
    ::cuda::copy_bytes(stream, ::cuda::std::span{state.second_moments.data(), state.second_moments.size()}, second_moments);
    ::cuda::copy_bytes(stream, ::cuda::std::span{state.parameter_steps.data(), state.parameter_steps.size()}, parameter_steps);
    stream.sync();
}

template <NetworkShape Shape>
Network<Shape>::CublasLtMatmulPlans::~CublasLtMatmulPlans() noexcept {
    for (std::size_t index = 0uz; index < cublaslt_matmul_plan_count; ++index) {
        if (output_descriptors[index] != nullptr) cublasLtMatrixLayoutDestroy(output_descriptors[index]);
        if (b_descriptors[index] != nullptr) cublasLtMatrixLayoutDestroy(b_descriptors[index]);
        if (a_descriptors[index] != nullptr) cublasLtMatrixLayoutDestroy(a_descriptors[index]);
        if (operation_descriptors[index] != nullptr) cublasLtMatmulDescDestroy(operation_descriptors[index]);
    }
}

template <NetworkShape Shape>
void Network<Shape>::CublasLtDeleter::operator()(cublasLtContext* const handle) const noexcept {
    cublasLtDestroy(handle);
}

template struct Network<nerf_synthetic_network_shape>;
} // namespace physica::reconstruction::instant_ngp
