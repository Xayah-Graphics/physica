#include "lbfgsb-kernels.h"
#include <cub/device/device_radix_sort.cuh>
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/limits>

namespace physica::optimization::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __device__ std::uint32_t physical_correction(const std::uint32_t logical, const std::uint32_t memory, const std::uint32_t correction_count, const std::uint32_t correction_head) {
            return correction_count < memory ? logical : (correction_head + logical) % memory;
        }

        __device__ double block_sum(const double value, double* reduction) {
            reduction[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
                __syncthreads();
            }
            const double result = reduction[0];
            __syncthreads();
            return result;
        }

        __device__ double block_min(const double value, double* reduction) {
            reduction[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) reduction[threadIdx.x] = ::cuda::std::min(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
                __syncthreads();
            }
            const double result = reduction[0];
            __syncthreads();
            return result;
        }

        __device__ double block_dot(const double* first, const double* second, const std::uint32_t count, double* reduction) {
            double value = 0.0;
            for (std::uint32_t index = threadIdx.x; index < count; index += blockDim.x) value += first[index] * second[index];
            return block_sum(value, reduction);
        }

        __device__ void hessian_product(const Storage storage, const std::uint32_t parameter_count, const std::uint32_t memory, const std::uint32_t correction_count, const std::uint32_t correction_head, const double scale, const double* input, double* output, double* reduction) {
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) output[parameter] = scale * input[parameter];
            __syncthreads();
            for (std::uint32_t logical = 0u; logical < correction_count; ++logical) {
                const std::uint32_t physical      = physical_correction(logical, memory, correction_count, correction_head);
                const double* hessian_step        = storage.hessian_steps + static_cast<std::size_t>(logical) * parameter_count;
                const double* gradient_change     = storage.correction_gradient_differences + static_cast<std::size_t>(physical) * parameter_count;
                const double hessian_coefficient  = block_dot(hessian_step, input, parameter_count, reduction) / storage.hessian_step_curvatures[logical];
                const double gradient_coefficient = block_dot(gradient_change, input, parameter_count, reduction) / storage.gradient_curvatures[logical];
                for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) output[parameter] += -hessian_coefficient * hessian_step[parameter] + gradient_coefficient * gradient_change[parameter];
                __syncthreads();
            }
        }

        __device__ void inverse_hessian_product(const Storage storage, const std::uint32_t parameter_count, const std::uint32_t memory, const std::uint32_t correction_count, const std::uint32_t correction_head, const double* input, double* output, double* coefficients, double* reduction) {
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) output[parameter] = input[parameter];
            __syncthreads();
            for (std::uint32_t reverse = 0u; reverse < correction_count; ++reverse) {
                const std::uint32_t logical   = correction_count - 1u - reverse;
                const std::uint32_t physical  = physical_correction(logical, memory, correction_count, correction_head);
                const double* step            = storage.correction_steps + static_cast<std::size_t>(physical) * parameter_count;
                const double* gradient_change = storage.correction_gradient_differences + static_cast<std::size_t>(physical) * parameter_count;
                const double coefficient      = storage.inverse_curvatures[physical] * block_dot(step, output, parameter_count, reduction);
                if (threadIdx.x == 0u) coefficients[logical] = coefficient;
                __syncthreads();
                for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) output[parameter] -= coefficient * gradient_change[parameter];
                __syncthreads();
            }

            double scale = 1.0;
            if (correction_count != 0u) {
                const std::uint32_t latest    = physical_correction(correction_count - 1u, memory, correction_count, correction_head);
                const double* gradient_change = storage.correction_gradient_differences + static_cast<std::size_t>(latest) * parameter_count;
                scale                         = 1.0 / (block_dot(gradient_change, gradient_change, parameter_count, reduction) * storage.inverse_curvatures[latest]);
            }
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) output[parameter] *= scale;
            __syncthreads();

            for (std::uint32_t logical = 0u; logical < correction_count; ++logical) {
                const std::uint32_t physical  = physical_correction(logical, memory, correction_count, correction_head);
                const double* step            = storage.correction_steps + static_cast<std::size_t>(physical) * parameter_count;
                const double* gradient_change = storage.correction_gradient_differences + static_cast<std::size_t>(physical) * parameter_count;
                const double coefficient      = coefficients[logical] - storage.inverse_curvatures[physical] * block_dot(gradient_change, output, parameter_count, reduction);
                for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) output[parameter] += coefficient * step[parameter];
                __syncthreads();
            }
        }

        __global__ void initialize_kernel(const std::uint32_t parameter_count, const Storage storage) {
            const std::uint32_t parameter = blockIdx.x * blockDim.x + threadIdx.x;
            if (parameter < parameter_count) storage.parameters[parameter] = ::cuda::std::clamp(storage.parameters[parameter], storage.lower_bounds[parameter], storage.upper_bounds[parameter]);
        }

        __global__ void gradient_metrics_kernel(const std::uint32_t parameter_count, const double* parameters, const double* gradient, const Storage storage) {
            __shared__ double gradient_reduction[block_size];
            __shared__ double projected_reduction[block_size];
            double gradient_squared  = 0.0;
            double projected_squared = 0.0;
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) {
                gradient_squared += gradient[parameter] * gradient[parameter];
                const double projected = parameters[parameter] - ::cuda::std::clamp(parameters[parameter] - gradient[parameter], storage.lower_bounds[parameter], storage.upper_bounds[parameter]);
                projected_squared += projected * projected;
            }
            const double gradient_norm  = ::cuda::std::sqrt(block_sum(gradient_squared, gradient_reduction));
            const double projected_norm = ::cuda::std::sqrt(block_sum(projected_squared, projected_reduction));
            if (threadIdx.x == 0u) *static_cast<GradientMetrics*>(storage.status) = {.gradient_norm = gradient_norm, .projected_gradient_norm = projected_norm};
        }

        __global__ void breakpoint_kernel(const std::uint32_t parameter_count, const Storage storage) {
            const std::uint32_t parameter = blockIdx.x * blockDim.x + threadIdx.x;
            if (parameter >= parameter_count) return;
            double value = -storage.gradient[parameter];
            if (storage.parameters[parameter] <= storage.lower_bounds[parameter] && value < 0.0) value = 0.0;
            if (storage.parameters[parameter] >= storage.upper_bounds[parameter] && value > 0.0) value = 0.0;
            storage.path[parameter]               = value;
            storage.free_direction[parameter]     = value;
            storage.breakpoints[parameter]        = value > 0.0 ? (storage.upper_bounds[parameter] - storage.parameters[parameter]) / value : value < 0.0 ? (storage.lower_bounds[parameter] - storage.parameters[parameter]) / value : ::cuda::std::numeric_limits<double>::infinity();
            storage.breakpoint_indices[parameter] = parameter;
        }

        __global__ void search_direction_kernel(const std::uint32_t parameter_count, const std::uint32_t memory, const std::uint32_t correction_count, const std::uint32_t correction_head, const Storage storage) {
            __shared__ double reduction[block_size];
            __shared__ double scale;
            __shared__ double cauchy_time;
            __shared__ std::uint32_t free_count;
            __shared__ std::uint32_t error;
            extern __shared__ double shared_values[];
            double* hessian_path          = shared_values;
            double* gradient_path         = hessian_path + memory;
            double* hessian_displacement  = gradient_path + memory;
            double* gradient_displacement = hessian_displacement + memory;

            if (threadIdx.x == 0u) {
                scale       = 1.0;
                cauchy_time = 0.0;
                error       = 0u;
            }
            __syncthreads();

            if (correction_count != 0u) {
                const std::uint32_t latest    = physical_correction(correction_count - 1u, memory, correction_count, correction_head);
                const double* gradient_change = storage.correction_gradient_differences + static_cast<std::size_t>(latest) * parameter_count;
                const double next_scale       = block_dot(gradient_change, gradient_change, parameter_count, reduction) * storage.inverse_curvatures[latest];
                if (threadIdx.x == 0u) scale = next_scale;
                __syncthreads();
            }

            for (std::uint32_t logical = 0u; logical < correction_count; ++logical) {
                const std::uint32_t physical = physical_correction(logical, memory, correction_count, correction_head);
                const double* step           = storage.correction_steps + static_cast<std::size_t>(physical) * parameter_count;
                double* hessian_step         = storage.hessian_steps + static_cast<std::size_t>(logical) * parameter_count;
                for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) hessian_step[parameter] = scale * step[parameter];
                __syncthreads();
                for (std::uint32_t previous = 0u; previous < logical; ++previous) {
                    const std::uint32_t previous_physical  = physical_correction(previous, memory, correction_count, correction_head);
                    const double* previous_hessian_step    = storage.hessian_steps + static_cast<std::size_t>(previous) * parameter_count;
                    const double* previous_gradient_change = storage.correction_gradient_differences + static_cast<std::size_t>(previous_physical) * parameter_count;
                    const double hessian_coefficient       = block_dot(previous_hessian_step, step, parameter_count, reduction) / storage.hessian_step_curvatures[previous];
                    const double gradient_coefficient      = block_dot(previous_gradient_change, step, parameter_count, reduction) / storage.gradient_curvatures[previous];
                    for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) hessian_step[parameter] += -hessian_coefficient * previous_hessian_step[parameter] + gradient_coefficient * previous_gradient_change[parameter];
                    __syncthreads();
                }
                const double hessian_curvature = block_dot(step, hessian_step, parameter_count, reduction);
                if (threadIdx.x == 0u) {
                    storage.hessian_step_curvatures[logical] = hessian_curvature;
                    storage.gradient_curvatures[logical]     = 1.0 / storage.inverse_curvatures[physical];
                }
                __syncthreads();
            }

            const double gradient_direction = block_dot(storage.gradient, storage.free_direction, parameter_count, reduction);
            double direction_squared        = block_dot(storage.free_direction, storage.free_direction, parameter_count, reduction);
            for (std::uint32_t logical = threadIdx.x; logical < correction_count; logical += blockDim.x) {
                hessian_displacement[logical]  = 0.0;
                gradient_displacement[logical] = 0.0;
            }
            __syncthreads();
            for (std::uint32_t logical = 0u; logical < correction_count; ++logical) {
                const std::uint32_t physical    = physical_correction(logical, memory, correction_count, correction_head);
                const double next_hessian_path  = block_dot(storage.hessian_steps + static_cast<std::size_t>(logical) * parameter_count, storage.free_direction, parameter_count, reduction);
                const double next_gradient_path = block_dot(storage.correction_gradient_differences + static_cast<std::size_t>(physical) * parameter_count, storage.free_direction, parameter_count, reduction);
                if (threadIdx.x == 0u) {
                    hessian_path[logical]  = next_hessian_path;
                    gradient_path[logical] = next_gradient_path;
                }
                __syncthreads();
            }

            if (threadIdx.x == 0u) {
                double next_gradient_direction = gradient_direction;
                double direction_displacement  = 0.0;
                double current_time            = 0.0;
                std::uint32_t next_breakpoint  = 0u;
                for (;;) {
                    double slope     = next_gradient_direction + scale * direction_displacement;
                    double curvature = scale * direction_squared;
                    for (std::uint32_t logical = 0u; logical < correction_count; ++logical) {
                        slope += -hessian_path[logical] * hessian_displacement[logical] / storage.hessian_step_curvatures[logical] + gradient_path[logical] * gradient_displacement[logical] / storage.gradient_curvatures[logical];
                        curvature += -hessian_path[logical] * hessian_path[logical] / storage.hessian_step_curvatures[logical] + gradient_path[logical] * gradient_path[logical] / storage.gradient_curvatures[logical];
                    }
                    const double interval        = next_breakpoint == parameter_count ? ::cuda::std::numeric_limits<double>::infinity() : storage.sorted_breakpoints[next_breakpoint] - current_time;
                    const double stationary_time = -slope / curvature;
                    if (stationary_time >= 0.0 && stationary_time <= interval) {
                        current_time += stationary_time;
                        break;
                    }
                    if (!::cuda::std::isfinite(interval)) {
                        error = 1u;
                        break;
                    }
                    direction_displacement += interval * direction_squared;
                    for (std::uint32_t logical = 0u; logical < correction_count; ++logical) {
                        hessian_displacement[logical] += interval * hessian_path[logical];
                        gradient_displacement[logical] += interval * gradient_path[logical];
                    }
                    current_time = storage.sorted_breakpoints[next_breakpoint];
                    while (next_breakpoint < parameter_count && storage.sorted_breakpoints[next_breakpoint] == current_time) {
                        const std::uint32_t parameter       = storage.sorted_breakpoint_indices[next_breakpoint];
                        const double value                  = storage.free_direction[parameter];
                        const double parameter_displacement = value > 0.0 ? storage.upper_bounds[parameter] - storage.parameters[parameter] : storage.lower_bounds[parameter] - storage.parameters[parameter];
                        next_gradient_direction -= storage.gradient[parameter] * value;
                        direction_displacement -= value * parameter_displacement;
                        direction_squared -= value * value;
                        for (std::uint32_t logical = 0u; logical < correction_count; ++logical) {
                            const std::uint32_t physical = physical_correction(logical, memory, correction_count, correction_head);
                            hessian_path[logical] -= storage.hessian_steps[static_cast<std::size_t>(logical) * parameter_count + parameter] * value;
                            gradient_path[logical] -= storage.correction_gradient_differences[static_cast<std::size_t>(physical) * parameter_count + parameter] * value;
                        }
                        storage.free_direction[parameter] = 0.0;
                        ++next_breakpoint;
                    }
                    if (direction_squared == 0.0) break;
                }
                cauchy_time = current_time;
            }
            __syncthreads();
            if (error != 0u) {
                if (threadIdx.x == 0u) *static_cast<SearchResult*>(storage.status) = {.base_directional_derivative = 0.0, .error = error};
                return;
            }

            double local_free_count = 0.0;
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) {
                storage.cauchy[parameter]       = ::cuda::std::clamp(storage.parameters[parameter] + cauchy_time * storage.path[parameter], storage.lower_bounds[parameter], storage.upper_bounds[parameter]);
                storage.displacement[parameter] = storage.cauchy[parameter] - storage.parameters[parameter];
                storage.free_mask[parameter]    = storage.cauchy[parameter] > storage.lower_bounds[parameter] && storage.cauchy[parameter] < storage.upper_bounds[parameter];
                local_free_count += storage.free_mask[parameter] != 0u ? 1.0 : 0.0;
                storage.subspace_direction[parameter] = 0.0;
            }
            const double free_count_value = block_sum(local_free_count, reduction);
            if (threadIdx.x == 0u) free_count = static_cast<std::uint32_t>(free_count_value);
            __syncthreads();

            hessian_product(storage, parameter_count, memory, correction_count, correction_head, scale, storage.displacement, storage.hessian_displacement, reduction);
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) storage.model_gradient[parameter] = storage.gradient[parameter] + storage.hessian_displacement[parameter];
            __syncthreads();

            if (free_count == parameter_count) {
                inverse_hessian_product(storage, parameter_count, memory, correction_count, correction_head, storage.model_gradient, storage.subspace_direction, hessian_path, reduction);
                for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) storage.subspace_direction[parameter] = -storage.subspace_direction[parameter];
                __syncthreads();
            } else {
                double residual_squared_local = 0.0;
                for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) {
                    storage.residual[parameter]  = storage.free_mask[parameter] == 0u ? 0.0 : -storage.model_gradient[parameter];
                    storage.conjugate[parameter] = storage.residual[parameter];
                    residual_squared_local += storage.residual[parameter] * storage.residual[parameter];
                }
                double residual_squared       = block_sum(residual_squared_local, reduction);
                const double initial_residual = residual_squared;
                for (std::uint32_t conjugate_iteration = 0u; conjugate_iteration <= 2u * correction_count && residual_squared > 1.0e-20 * initial_residual; ++conjugate_iteration) {
                    hessian_product(storage, parameter_count, memory, correction_count, correction_head, scale, storage.conjugate, storage.hessian_conjugate, reduction);
                    for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x)
                        if (storage.free_mask[parameter] == 0u) storage.hessian_conjugate[parameter] = 0.0;
                    __syncthreads();
                    const double conjugate_curvature = block_dot(storage.conjugate, storage.hessian_conjugate, parameter_count, reduction);
                    const double conjugate_step      = residual_squared / conjugate_curvature;
                    for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) {
                        storage.subspace_direction[parameter] += conjugate_step * storage.conjugate[parameter];
                        storage.residual[parameter] -= conjugate_step * storage.hessian_conjugate[parameter];
                    }
                    __syncthreads();
                    const double next_residual_squared = block_dot(storage.residual, storage.residual, parameter_count, reduction);
                    const double coefficient           = next_residual_squared / residual_squared;
                    for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) storage.conjugate[parameter] = storage.residual[parameter] + coefficient * storage.conjugate[parameter];
                    __syncthreads();
                    residual_squared = next_residual_squared;
                }
            }

            double local_step = 1.0;
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) {
                if (storage.subspace_direction[parameter] > 0.0) local_step = ::cuda::std::min(local_step, (storage.upper_bounds[parameter] - storage.cauchy[parameter]) / storage.subspace_direction[parameter]);
                if (storage.subspace_direction[parameter] < 0.0) local_step = ::cuda::std::min(local_step, (storage.lower_bounds[parameter] - storage.cauchy[parameter]) / storage.subspace_direction[parameter]);
            }
            const double subspace_step = block_min(local_step, reduction);
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) {
                const double candidate       = ::cuda::std::clamp(storage.cauchy[parameter] + subspace_step * storage.subspace_direction[parameter], storage.lower_bounds[parameter], storage.upper_bounds[parameter]);
                storage.direction[parameter] = candidate - storage.parameters[parameter];
            }
            __syncthreads();
            double directional_derivative = block_dot(storage.gradient, storage.direction, parameter_count, reduction);
            if (!(directional_derivative < 0.0)) {
                for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) storage.direction[parameter] = storage.cauchy[parameter] - storage.parameters[parameter];
                __syncthreads();
                directional_derivative = block_dot(storage.gradient, storage.direction, parameter_count, reduction);
            }
            if (threadIdx.x == 0u) *static_cast<SearchResult*>(storage.status) = {.base_directional_derivative = directional_derivative, .error = directional_derivative < 0.0 ? 0u : 2u};
        }

        __global__ void set_trial_kernel(const std::uint32_t parameter_count, const double step_length, const Storage storage) {
            const std::uint32_t parameter = blockIdx.x * blockDim.x + threadIdx.x;
            if (parameter < parameter_count) storage.trial_parameters[parameter] = ::cuda::std::clamp(storage.parameters[parameter] + step_length * storage.direction[parameter], storage.lower_bounds[parameter], storage.upper_bounds[parameter]);
        }

        __global__ void accept_trial_kernel(const std::uint32_t parameter_count, const std::uint32_t correction_slot, const double* next_gradient, const Storage storage) {
            __shared__ double reduction[block_size];
            double* step            = storage.correction_steps + static_cast<std::size_t>(correction_slot) * parameter_count;
            double* gradient_change = storage.correction_gradient_differences + static_cast<std::size_t>(correction_slot) * parameter_count;
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) {
                step[parameter]            = storage.trial_parameters[parameter] - storage.parameters[parameter];
                gradient_change[parameter] = next_gradient[parameter] - storage.gradient[parameter];
            }
            __syncthreads();
            const double curvature    = block_dot(step, gradient_change, parameter_count, reduction);
            const double step_squared = block_dot(step, step, parameter_count, reduction);
            const bool accepted       = curvature > 1.0e-12 * step_squared;
            if (threadIdx.x == 0u && accepted) storage.inverse_curvatures[correction_slot] = 1.0 / curvature;
            for (std::uint32_t parameter = threadIdx.x; parameter < parameter_count; parameter += blockDim.x) {
                storage.parameters[parameter] = storage.trial_parameters[parameter];
                storage.gradient[parameter]   = next_gradient[parameter];
            }
            if (threadIdx.x == 0u) *static_cast<AcceptanceResult*>(storage.status) = {.correction_accepted = accepted ? 1u : 0u};
        }
    } // namespace

    std::size_t sort_scratch_size(const std::uint32_t parameter_count) {
        std::size_t bytes{};
        cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<double*>(nullptr), static_cast<double*>(nullptr), static_cast<std::uint32_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), parameter_count);
        return bytes;
    }

    void initialize(const ::cuda::stream_ref stream, const std::uint32_t parameter_count, const Storage storage) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(parameter_count), initialize_kernel, parameter_count, storage);
    }

    GradientMetrics gradient_metrics(const ::cuda::stream_ref stream, const std::uint32_t parameter_count, const double* parameters, const double* gradient, const Storage storage) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1u), ::cuda::block_dims(block_size))), gradient_metrics_kernel, parameter_count, parameters, gradient, storage);
        GradientMetrics result;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const GradientMetrics>{static_cast<const GradientMetrics*>(storage.status), 1u}, ::cuda::std::span<GradientMetrics>{&result, 1u});
        stream.sync();
        return result;
    }

    SearchResult prepare_direction(const ::cuda::stream_ref stream, const std::uint32_t parameter_count, const std::uint32_t memory, const std::uint32_t correction_count, const std::uint32_t correction_head, const Storage storage) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(parameter_count), breakpoint_kernel, parameter_count, storage);
        std::size_t sort_scratch_bytes = storage.sort_scratch_bytes;
        cub::DeviceRadixSort::SortPairs(storage.sort_scratch, sort_scratch_bytes, storage.breakpoints, storage.sorted_breakpoints, storage.breakpoint_indices, storage.sorted_breakpoint_indices, parameter_count, 0, static_cast<int>(8u * sizeof(double)), stream.get());
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1u), ::cuda::block_dims(block_size)), ::cuda::dynamic_shared_memory<std::uint8_t[]>(static_cast<std::size_t>(4u) * memory * sizeof(double))), search_direction_kernel, parameter_count, memory, correction_count, correction_head, storage);
        SearchResult result;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const SearchResult>{static_cast<const SearchResult*>(storage.status), 1u}, ::cuda::std::span<SearchResult>{&result, 1u});
        stream.sync();
        return result;
    }

    void set_trial(const ::cuda::stream_ref stream, const std::uint32_t parameter_count, const double step_length, const Storage storage) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(parameter_count), set_trial_kernel, parameter_count, step_length, storage);
    }

    AcceptanceResult accept_trial(const ::cuda::stream_ref stream, const std::uint32_t parameter_count, const std::uint32_t correction_slot, const double* next_gradient, const Storage storage) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::make_hierarchy(::cuda::grid_dims(1u), ::cuda::block_dims(block_size))), accept_trial_kernel, parameter_count, correction_slot, next_gradient, storage);
        AcceptanceResult result;
        ::cuda::copy_bytes(stream, ::cuda::std::span<const AcceptanceResult>{static_cast<const AcceptanceResult*>(storage.status), 1u}, ::cuda::std::span<AcceptanceResult>{&result, 1u});
        stream.sync();
        return result;
    }
} // namespace physica::optimization::kernels
