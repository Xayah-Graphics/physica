module;

#include <physica/cuda.h>

export module physica.fluids.gas.keyframe_smoke.control;

import std;
import physica.fluids.gas.domain;
import physica.fluids.gas.keyframe_smoke;

export namespace physica::fluids::gas::keyframe_smoke {
    struct BoundedValue final {
        double initial{};
        double lower{};
        double upper{};
    };

    struct WindDefinition final {
        std::uint32_t begin_step{};
        std::uint32_t end_step{};
        float width{0.1F};
        std::array<BoundedValue, 3> center{};
        std::array<BoundedValue, 3> vector{};
    };

    struct VortexDefinition final {
        std::uint32_t begin_step{};
        std::uint32_t end_step{};
        float width{0.1F};
        Vector3 axis{0.0F, 1.0F, 0.0F};
        std::array<BoundedValue, 3> center{};
        BoundedValue strength{};
    };

    struct ControlConfiguration final {
        std::vector<WindDefinition> winds;
        std::vector<VortexDefinition> vortices;
    };

    struct ControlSystem final {
        const ControlConfiguration configuration;
        std::vector<double> parameter_values;
        std::vector<double> lower_bounds;
        std::vector<double> upper_bounds;

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
        ::cuda::device_buffer<std::byte> device_winds;
        ::cuda::device_buffer<std::byte> device_vortices;
    };
} // namespace physica::fluids::gas::keyframe_smoke
