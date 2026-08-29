module;

#include <physica/cuda.h>

export module physica.fluids.gas.operators.conservation;

import std;
import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    struct MassConservation final {
        struct Configuration final {
            float dissipation{};
        };

        struct Cache final {
            ::cuda::device_buffer<double> input_mass;
            ::cuda::device_buffer<double> advected_mass;
        };

        struct TangentWorkspace final {
            ::cuda::device_buffer<double> input_mass;
            ::cuda::device_buffer<double> advected_mass;
        };

        struct AdjointWorkspace final {
            ::cuda::device_buffer<double> density_dot;
        };

        explicit MassConservation(Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const ScalarField<float>& input, const ScalarField<float>& advected, ScalarField<float>& output, Cache& cache) const;
        void jvp(const Domain& domain, const ScalarField<float>& input, const ScalarField<float>& advected, const ScalarField<float>& input_tangent, const ScalarField<float>& advected_tangent, const Cache& cache, ScalarField<float>& output_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const ScalarField<float>& advected, const Cache& cache, const ScalarField<double>& output_adjoint, ScalarField<double>& input_adjoint, ScalarField<double>& advected_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };
} // namespace physica::fluids::gas::operators
