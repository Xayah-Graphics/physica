module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

export module physica.fluids.gas.adjoint_control.control;

import std;
export import physica.fluids.gas.adjoint_control.solver;

export namespace physica::fluids::gas::adjoint_control {
    struct ControlConfiguration final {
        std::array<std::uint32_t, 3> lattice{12u, 12u, 12u};
        std::uint32_t step_count{20u};
        float gaussian_sigma{0.075F};
        double lower_bound{-18.0};
        double upper_bound{18.0};
    };

    struct ParameterSet final {
        std::vector<double> values;
        std::vector<double> lower_bounds;
        std::vector<double> upper_bounds;
    };

    struct ControlSystem final {
        const ControlConfiguration configuration;
        ParameterSet parameters;
        const std::uint32_t center_count;

        ControlSystem(const Domain& domain, ControlConfiguration configuration);

        void upload_parameters(const Domain& domain, std::span<const double> values);
        void upload_direction(const Domain& domain, std::span<const double> direction);
        void clear_gradient(const Domain& domain);
        void forward(const Domain& domain, std::uint32_t step, DenseControl& output) const;
        void jvp(const Domain& domain, std::uint32_t step, DenseControlTangent& output_tangent) const;
        void vjp(const Domain& domain, std::uint32_t step, const DenseControlAdjoint& output_adjoint);
        void download_gradient(const Domain& domain, std::span<double> gradient) const;
        [[nodiscard]] std::vector<std::uint8_t> active_parameters(std::uint32_t begin_step, std::uint32_t end_step) const;

    private:
        ::cuda::device_buffer<double> device_parameters;
        ::cuda::device_buffer<double> device_direction;
        ::cuda::device_buffer<double> device_gradient;
    };
} // namespace physica::fluids::gas::adjoint_control
