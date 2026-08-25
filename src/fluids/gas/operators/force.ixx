module;

#include <physica/cuda.h>

export module physica.fluids.gas.operators.force;

import physica.fluids.gas.domain;

export namespace physica::fluids::gas::operators {
    struct DensityBuoyancy final {
        struct Configuration final {
            float buoyancy{1.5F};
        };

        struct Cache final {
            CenteredVectorField<float> force;
        };

        struct TangentWorkspace final {
            CenteredVectorField<float> force;
        };

        struct AdjointWorkspace final {};

        explicit DensityBuoyancy(Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const CellField<float>& density, Cache& cache) const;
        void jvp(const Domain& domain, const CellField<float>& density_tangent, const Cache& cache, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const Cache& cache, const CenteredVectorField<double>& force_adjoint, CellField<double>& density_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };

    struct VorticityConfinement final {
        struct Cache final {
            CenteredVectorField<float> centered_velocity;
            CenteredVectorField<float> vorticity;
            CellField<float> magnitude;
            CenteredVectorField<float> normal;
            CellField<float> normalizer;
        };

        struct TangentWorkspace final {
            CenteredVectorField<float> centered_velocity;
            CenteredVectorField<float> vorticity;
            CellField<float> magnitude;
            CenteredVectorField<float> normal;
        };

        struct AdjointWorkspace final {
            CenteredVectorField<double> centered_velocity;
            CenteredVectorField<double> vorticity;
            CellField<double> magnitude;
            CenteredVectorField<double> normal;
        };

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const StaggeredVectorField<float>& velocity, const float* confinement, CenteredVectorField<float>& force, Cache& cache) const;
        void forward(const Domain& domain, const StaggeredVectorField<float>& velocity, float confinement, CenteredVectorField<float>& force, Cache& cache) const;
        void jvp(const Domain& domain, const StaggeredVectorField<float>& velocity_tangent, const float* confinement, const float* confinement_tangent, const Cache& cache, CenteredVectorField<float>& force_tangent, TangentWorkspace& workspace) const;
        void jvp(const Domain& domain, const StaggeredVectorField<float>& velocity_tangent, float confinement, const Cache& cache, CenteredVectorField<float>& force_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const float* confinement, const Cache& cache, const CenteredVectorField<double>& force_adjoint, StaggeredVectorField<double>& velocity_adjoint, double* confinement_adjoint, AdjointWorkspace& workspace) const;
        void vjp(const Domain& domain, float confinement, const Cache& cache, const CenteredVectorField<double>& force_adjoint, StaggeredVectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const;
    };

    struct ThermalBuoyancyVorticity final {
        struct Configuration final {
            bool vorticity_confinement_enabled{true};
        };

        struct Cache final {
            CenteredVectorField<float> force;
            VorticityConfinement::Cache vorticity;
        };

        struct TangentWorkspace final {
            CenteredVectorField<float> force;
            VorticityConfinement::TangentWorkspace vorticity;
        };

        struct AdjointWorkspace final {
            VorticityConfinement::AdjointWorkspace vorticity;
        };

        struct Parameters final {
            ::cuda::device_buffer<float> values;
        };

        struct ParameterTangent final {
            ::cuda::device_buffer<float> values;
        };

        struct ParameterAdjoint final {
            ::cuda::device_buffer<double> values;
        };

        inline static constexpr std::size_t ambient_temperature   = 0u;
        inline static constexpr std::size_t density_buoyancy      = 1u;
        inline static constexpr std::size_t temperature_buoyancy  = 2u;
        inline static constexpr std::size_t vorticity_confinement = 3u;

        explicit ThermalBuoyancyVorticity(Configuration configuration);

        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const;
        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const CellField<float>& density, const CellField<float>& temperature, const StaggeredVectorField<float>& velocity, const CenteredVectorField<float>& external_acceleration, const Parameters& parameters, Cache& cache) const;
        void jvp(const Domain& domain, const CellField<float>& density, const CellField<float>& temperature, const CellField<float>& density_tangent, const CellField<float>& temperature_tangent, const StaggeredVectorField<float>& velocity_tangent, const CenteredVectorField<float>& external_acceleration_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Cache& cache, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const CellField<float>& density, const CellField<float>& temperature, const Parameters& parameters, const Cache& cache, const CenteredVectorField<double>& force_adjoint, StaggeredVectorField<double>& velocity_adjoint, CellField<double>& density_adjoint, CellField<double>& temperature_adjoint, CenteredVectorField<double>& external_acceleration_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
        VorticityConfinement vorticity;
    };

    struct ControlledDensityBuoyancyVorticity final {
        struct Configuration final {
            float density_buoyancy{2.0F};
            float vorticity_confinement{2.0F};
        };

        struct Cache final {
            DensityBuoyancy::Cache buoyancy;
            CenteredVectorField<float> total;
            VorticityConfinement::Cache vorticity;
        };

        struct TangentWorkspace final {
            DensityBuoyancy::TangentWorkspace buoyancy;
            CenteredVectorField<float> total;
            VorticityConfinement::TangentWorkspace vorticity;
        };

        struct AdjointWorkspace final {
            CenteredVectorField<double> physical;
            DensityBuoyancy::AdjointWorkspace buoyancy;
            VorticityConfinement::AdjointWorkspace vorticity;
        };

        explicit ControlledDensityBuoyancyVorticity(Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const CellField<float>& density, const StaggeredVectorField<float>& velocity, const CenteredVectorField<float>& control, Cache& cache) const;
        void jvp(const Domain& domain, const CellField<float>& density_tangent, const StaggeredVectorField<float>& velocity_tangent, const CenteredVectorField<float>& control_tangent, const Cache& cache, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const Cache& cache, const CenteredVectorField<double>& total_adjoint, CellField<double>& density_adjoint, StaggeredVectorField<double>& velocity_adjoint, CenteredVectorField<double>& control_adjoint, AdjointWorkspace& workspace) const;

    private:
        const float vorticity_confinement;
        DensityBuoyancy buoyancy;
        VorticityConfinement vorticity;
    };
} // namespace physica::fluids::gas::operators
