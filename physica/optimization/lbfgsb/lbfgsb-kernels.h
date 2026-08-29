#ifndef PHYSICA_OPTIMIZATION_LBFGSB_KERNELS_H
#define PHYSICA_OPTIMIZATION_LBFGSB_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::optimization::kernels {
    struct Storage final {
        double* parameters;
        double* lower_bounds;
        double* upper_bounds;
        double* gradient;
        double* trial_parameters;
        double* correction_steps;
        double* correction_gradient_differences;
        double* inverse_curvatures;
        double* hessian_steps;
        double* hessian_step_curvatures;
        double* gradient_curvatures;
        double* direction;
        double* path;
        double* free_direction;
        double* cauchy;
        double* displacement;
        double* hessian_displacement;
        double* model_gradient;
        double* subspace_direction;
        double* residual;
        double* conjugate;
        double* hessian_conjugate;
        double* breakpoints;
        double* sorted_breakpoints;
        std::uint32_t* breakpoint_indices;
        std::uint32_t* sorted_breakpoint_indices;
        std::uint8_t* free_mask;
        void* sort_scratch;
        std::size_t sort_scratch_bytes;
        void* status;
    };

    struct SearchResult final {
        double base_directional_derivative;
        std::uint32_t error;
    };

    struct GradientMetrics final {
        double gradient_norm;
        double projected_gradient_norm;
    };

    struct AcceptanceResult final {
        std::uint32_t correction_accepted;
    };

    [[nodiscard]] std::size_t sort_scratch_size(std::uint32_t parameter_count);
    void initialize(::cuda::stream_ref stream, std::uint32_t parameter_count, Storage storage);
    [[nodiscard]] GradientMetrics gradient_metrics(::cuda::stream_ref stream, std::uint32_t parameter_count, const double* parameters, const double* gradient, Storage storage);
    [[nodiscard]] SearchResult prepare_direction(::cuda::stream_ref stream, std::uint32_t parameter_count, std::uint32_t memory, std::uint32_t correction_count, std::uint32_t correction_head, Storage storage);
    void set_trial(::cuda::stream_ref stream, std::uint32_t parameter_count, double step_length, Storage storage);
    [[nodiscard]] AcceptanceResult accept_trial(::cuda::stream_ref stream, std::uint32_t parameter_count, std::uint32_t correction_slot, const double* next_gradient, Storage storage);
} // namespace physica::optimization::kernels

#endif
