module;

#include <physica/cuda.h>

export module physica.fluids.gas.smoke.domain;

import std;

export namespace physica::fluids::gas::smoke {
    struct Vector3 final {
        float x;
        float y;
        float z;
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
        ScalarBoundaryMode mode = ScalarBoundaryMode::zero_gradient;
        float value             = 0.0F;
    };

    struct VelocityBoundaryFace final {
        VelocityBoundaryMode mode = VelocityBoundaryMode::zero_gradient;
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

    struct Ellipsoid final {
        Vector3 center{};
        Vector3 radius{1.0F, 1.0F, 1.0F};
    };

    struct Box final {
        Vector3 center{};
        Vector3 half_extent{1.0F, 1.0F, 1.0F};
    };

    struct Collider final {
        std::variant<Ellipsoid, Box> shape{Ellipsoid{}};
        Vector3 velocity{};
        float density     = 0.0F;
        float temperature = 0.0F;
    };

    struct DomainConfiguration final {
        std::array<std::uint32_t, 3> resolution{32u, 48u, 32u};
        float cell_size = 1.0F / 32.0F;
        float time_step = 1.0F / 60.0F;
        VelocityBoundary velocity_boundary{};
        ScalarBoundary pressure_boundary{};
        ScalarBoundary density_boundary{};
        ScalarBoundary temperature_boundary{};
        std::vector<Collider> colliders{};
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

    enum class ExecutionMode : std::uint32_t {
        forward,
        differentiable,
    };

    struct Domain final {
        const DomainConfiguration configuration;
        ::cuda::stream_ref stream;
        std::size_t cell_count;
        std::array<std::size_t, 3> face_counts;
        ::cuda::device_buffer<std::uint32_t> cell_mask;
        StaggeredVectorField collider_velocity;
        ScalarField collider_density;
        ScalarField collider_temperature;
        std::uint32_t pressure_anchor = 0u;

        Domain(DomainConfiguration configuration, ::cuda::stream_ref stream);

        Domain(const Domain&)            = delete;
        Domain& operator=(const Domain&) = delete;
        Domain(Domain&&)                 = delete;
        Domain& operator=(Domain&&)      = delete;

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
        void copy(const StaggeredVectorField& source, StaggeredVectorField& destination) const;
        void copy(const ScalarAdjointField& source, ScalarAdjointField& destination) const;
        void copy(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) const;
        void accumulate(const ScalarAdjointField& source, ScalarAdjointField& destination) const;
        void accumulate(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) const;
    };
} // namespace physica::fluids::gas::smoke
