module;

#include <physica/cuda.h>

export module physica.fluids.gas.adjoint_control.control;

import std;
import physica.fluids.gas.domain;
import physica.fluids.gas.adjoint_control;

export namespace physica::fluids::gas::adjoint_control {
    struct ControlConfiguration final {
        std::array<std::uint32_t, 3> lattice{12u, 12u, 12u};
        std::uint32_t step_count{20u};
        float gaussian_sigma{0.075F};
        double lower_bound{-18.0};
        double upper_bound{18.0};
    };

    struct ControlSystem final {
        const ControlConfiguration configuration;
        const std::uint32_t center_count;
        std::vector<double> parameter_values;
        std::vector<double> lower_bounds;
        std::vector<double> upper_bounds;

        explicit ControlSystem(ControlConfiguration configuration);

        void forward(const Domain& domain, std::uint32_t step, ::cuda::std::span<const double> parameters, DenseControl& output) const;
        void jvp(const Domain& domain, std::uint32_t step, ::cuda::std::span<const double> direction, DenseControlTangent& output_tangent) const;
        void vjp(const Domain& domain, std::uint32_t step, const DenseControlAdjoint& output_adjoint, ::cuda::std::span<double> gradient) const;
        [[nodiscard]] std::vector<std::uint8_t> active_parameters(std::uint32_t begin_step, std::uint32_t end_step) const;
    };
} // namespace physica::fluids::gas::adjoint_control
