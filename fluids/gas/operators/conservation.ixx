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

        void forward(const Domain& domain, const CellField<float>& input, const CellField<float>& advected, CellField<float>& output, Cache& cache) const;
        void jvp(const Domain& domain, const CellField<float>& input, const CellField<float>& advected, const CellField<float>& input_tangent, const CellField<float>& advected_tangent, const Cache& cache, CellField<float>& output_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const CellField<float>& advected, const Cache& cache, const CellField<double>& output_adjoint, CellField<double>& input_adjoint, CellField<double>& advected_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };
} // namespace physica::fluids::gas::operators
