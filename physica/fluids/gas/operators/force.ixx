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
            VectorField<float> force;
        };

        struct TangentWorkspace final {
            VectorField<float> force;
        };

        struct AdjointWorkspace final {};

        explicit DensityBuoyancy(Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const ScalarField<float>& density, Cache& cache) const;
        void jvp(const Domain& domain, const ScalarField<float>& density_tangent, const Cache& cache, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const Cache& cache, const VectorField<double>& force_adjoint, ScalarField<double>& density_adjoint, AdjointWorkspace& workspace) const;

    private:
        const Configuration configuration;
    };

    struct VorticityConfinement final {
        struct Cache final {
            VectorField<float> centered_velocity;
            VectorField<float> vorticity;
            ScalarField<float> magnitude;
            VectorField<float> normal;
            ScalarField<float> normalizer;
        };

        struct TangentWorkspace final {
            VectorField<float> centered_velocity;
            VectorField<float> vorticity;
            ScalarField<float> magnitude;
            VectorField<float> normal;
        };

        struct AdjointWorkspace final {
            VectorField<double> centered_velocity;
            VectorField<double> vorticity;
            ScalarField<double> magnitude;
            VectorField<double> normal;
        };

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const VectorField<float>& velocity, const float* confinement, VectorField<float>& force, Cache& cache) const;
        void forward(const Domain& domain, const VectorField<float>& velocity, float confinement, VectorField<float>& force, Cache& cache) const;
        void jvp(const Domain& domain, const VectorField<float>& velocity_tangent, const float* confinement, const float* confinement_tangent, const Cache& cache, VectorField<float>& force_tangent, TangentWorkspace& workspace) const;
        void jvp(const Domain& domain, const VectorField<float>& velocity_tangent, float confinement, const Cache& cache, VectorField<float>& force_tangent, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const float* confinement, const Cache& cache, const VectorField<double>& force_adjoint, VectorField<double>& velocity_adjoint, double* confinement_adjoint, AdjointWorkspace& workspace) const;
        void vjp(const Domain& domain, float confinement, const Cache& cache, const VectorField<double>& force_adjoint, VectorField<double>& velocity_adjoint, AdjointWorkspace& workspace) const;
    };

    struct ThermalBuoyancyVorticity final {
        struct Configuration final {
            bool vorticity_confinement_enabled{true};
        };

        struct Cache final {
            VectorField<float> force;
            VorticityConfinement::Cache vorticity;
        };

        struct TangentWorkspace final {
            VectorField<float> force;
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

        void forward(const Domain& domain, const ScalarField<float>& density, const ScalarField<float>& temperature, const VectorField<float>& velocity, const VectorField<float>& external_acceleration, const Parameters& parameters, Cache& cache) const;
        void jvp(const Domain& domain, const ScalarField<float>& density, const ScalarField<float>& temperature, const ScalarField<float>& density_tangent, const ScalarField<float>& temperature_tangent, const VectorField<float>& velocity_tangent, const VectorField<float>& external_acceleration_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, const Cache& cache, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const ScalarField<float>& density, const ScalarField<float>& temperature, const Parameters& parameters, const Cache& cache, const VectorField<double>& force_adjoint, VectorField<double>& velocity_adjoint, ScalarField<double>& density_adjoint, ScalarField<double>& temperature_adjoint, VectorField<double>& external_acceleration_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const;

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
            VectorField<float> total;
            VorticityConfinement::Cache vorticity;
        };

        struct TangentWorkspace final {
            DensityBuoyancy::TangentWorkspace buoyancy;
            VectorField<float> total;
            VorticityConfinement::TangentWorkspace vorticity;
        };

        struct AdjointWorkspace final {
            VectorField<double> physical;
            DensityBuoyancy::AdjointWorkspace buoyancy;
            VorticityConfinement::AdjointWorkspace vorticity;
        };

        explicit ControlledDensityBuoyancyVorticity(Configuration configuration);

        [[nodiscard]] Cache allocate_cache(const Domain& domain) const;
        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const;
        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const;

        void forward(const Domain& domain, const ScalarField<float>& density, const VectorField<float>& velocity, const VectorField<float>& control, Cache& cache) const;
        void jvp(const Domain& domain, const ScalarField<float>& density_tangent, const VectorField<float>& velocity_tangent, const VectorField<float>& control_tangent, const Cache& cache, TangentWorkspace& workspace) const;
        void vjp(const Domain& domain, const Cache& cache, const VectorField<double>& total_adjoint, ScalarField<double>& density_adjoint, VectorField<double>& velocity_adjoint, VectorField<double>& control_adjoint, AdjointWorkspace& workspace) const;

    private:
        const float vorticity_confinement;
        DensityBuoyancy buoyancy;
        VorticityConfinement vorticity;
    };
} // namespace physica::fluids::gas::operators
