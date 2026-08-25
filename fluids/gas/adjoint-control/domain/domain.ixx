module;

#include <physica/cuda.h>

export module physica.fluids.gas.adjoint_control.domain;

import std;

export namespace physica::fluids::gas::adjoint_control {
    enum class SpatialDimension : std::uint32_t {
        planar = 2u,
        volumetric = 3u,
    };

    struct Vector3 final {
        float x{};
        float y{};
        float z{};
    };

    enum class ScalarBoundaryMode : std::uint32_t {
        fixed_value,
        zero_gradient,
        periodic,
    };

    enum class VelocityBoundaryMode : std::uint32_t {
        fixed_value,
        zero_gradient,
        normal_fixed_tangent_zero_gradient,
        periodic,
    };

    struct ScalarBoundaryFace final {
        ScalarBoundaryMode mode{ScalarBoundaryMode::zero_gradient};
        float value{};
    };

    struct VelocityBoundaryFace final {
        VelocityBoundaryMode mode{VelocityBoundaryMode::normal_fixed_tangent_zero_gradient};
        Vector3 value{};
    };

    struct ScalarBoundary final {
        ScalarBoundaryFace x_min{};
        ScalarBoundaryFace x_max{};
        ScalarBoundaryFace y_min{};
        ScalarBoundaryFace y_max{};
        ScalarBoundaryFace z_min{};
        ScalarBoundaryFace z_max{};
    };

    struct VelocityBoundary final {
        VelocityBoundaryFace x_min{};
        VelocityBoundaryFace x_max{};
        VelocityBoundaryFace y_min{};
        VelocityBoundaryFace y_max{};
        VelocityBoundaryFace z_min{};
        VelocityBoundaryFace z_max{};
    };

    struct DomainConfiguration final {
        SpatialDimension dimension{SpatialDimension::volumetric};
        std::array<std::uint32_t, 3> resolution{30u, 30u, 30u};
        float cell_size{1.0F / 30.0F};
        float time_step{1.0F / 30.0F};
        VelocityBoundary velocity_boundary{};
        ScalarBoundary pressure_boundary{};
        ScalarBoundary density_boundary{};
    };

    struct ScalarField final {
        ::cuda::device_buffer<float> values;
    };

    struct CenteredVectorField final {
        ScalarField x;
        ScalarField y;
        ScalarField z;
    };

    struct StaggeredVectorField final {
        ::cuda::device_buffer<float> x;
        ::cuda::device_buffer<float> y;
        ::cuda::device_buffer<float> z;
    };

    struct ScalarAdjointField final {
        ::cuda::device_buffer<double> values;
    };

    struct CenteredVectorAdjointField final {
        ScalarAdjointField x;
        ScalarAdjointField y;
        ScalarAdjointField z;
    };

    struct StaggeredVectorAdjointField final {
        ::cuda::device_buffer<double> x;
        ::cuda::device_buffer<double> y;
        ::cuda::device_buffer<double> z;
    };

    struct Domain final {
        const DomainConfiguration configuration;
        ::cuda::stream_ref stream;
        const std::size_t cell_count;
        const std::array<std::size_t, 3> face_counts;
        const std::uint32_t pressure_anchor;

        Domain(DomainConfiguration configuration, ::cuda::stream_ref stream);

        Domain(const Domain&) = delete;
        Domain& operator=(const Domain&) = delete;
        Domain(Domain&&) = delete;
        Domain& operator=(Domain&&) = delete;

        [[nodiscard]] ScalarField allocate_scalar_field() const;
        [[nodiscard]] CenteredVectorField allocate_centered_vector_field() const;
        [[nodiscard]] StaggeredVectorField allocate_staggered_vector_field() const;
        [[nodiscard]] ScalarAdjointField allocate_scalar_adjoint_field() const;
        [[nodiscard]] CenteredVectorAdjointField allocate_centered_vector_adjoint_field() const;
        [[nodiscard]] StaggeredVectorAdjointField allocate_staggered_vector_adjoint_field() const;

        void clear(ScalarField& field) const;
        void clear(CenteredVectorField& field) const;
        void clear(StaggeredVectorField& field) const;
        void clear(ScalarAdjointField& field) const;
        void clear(CenteredVectorAdjointField& field) const;
        void clear(StaggeredVectorAdjointField& field) const;

        void copy(const ScalarField& source, ScalarField& destination) const;
        void copy(const CenteredVectorField& source, CenteredVectorField& destination) const;
        void copy(const StaggeredVectorField& source, StaggeredVectorField& destination) const;
        void copy(const ScalarAdjointField& source, ScalarAdjointField& destination) const;
        void copy(const CenteredVectorAdjointField& source, CenteredVectorAdjointField& destination) const;
        void copy(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) const;

        void accumulate(const ScalarAdjointField& source, ScalarAdjointField& destination) const;
        void accumulate(const CenteredVectorAdjointField& source, CenteredVectorAdjointField& destination) const;
        void accumulate(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) const;
    };
} // namespace physica::fluids::gas::adjoint_control
