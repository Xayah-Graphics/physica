module;

#include <physica/cuda.h>

export module physica.fluids.gas.domain;

import std;
export import physica.math;
export import physica.fluids.grid;

export namespace physica::fluids::gas {
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
        VelocityBoundaryMode mode{VelocityBoundaryMode::zero_gradient};
        Vector3<float> value{};
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

    [[nodiscard]] inline ScalarBoundary homogeneous(ScalarBoundary boundary) {
        boundary.x_min.value = 0.0F;
        boundary.x_max.value = 0.0F;
        boundary.y_min.value = 0.0F;
        boundary.y_max.value = 0.0F;
        boundary.z_min.value = 0.0F;
        boundary.z_max.value = 0.0F;
        return boundary;
    }

    [[nodiscard]] inline VelocityBoundary homogeneous(VelocityBoundary boundary) {
        boundary.x_min.value = {};
        boundary.x_max.value = {};
        boundary.y_min.value = {};
        boundary.y_max.value = {};
        boundary.z_min.value = {};
        boundary.z_max.value = {};
        return boundary;
    }

    struct Ellipsoid final {
        Vector3<float> center{};
        Vector3<float> radius{1.0F, 1.0F, 1.0F};
    };

    struct Collider final {
        std::variant<Ellipsoid, AxisAlignedBox3<float>> shape{Ellipsoid{}};
        Vector3<float> velocity{};
    };

    struct DomainConfiguration final {
        grid::Configuration grid{.resolution = {32u, 48u, 32u}, .cell_size = 1.0F / 32.0F};
        float time_step{1.0F / 60.0F};
        VelocityBoundary velocity_boundary{};
        std::vector<Collider> colliders{};
    };

    struct Domain final {
        const DomainConfiguration configuration;
        grid::Grid grid;
        simulation::ScalarField<std::uint32_t> collider_ids;
        simulation::VectorField<float> collider_velocity;
        std::uint32_t first_fluid_cell{};

        Domain(DomainConfiguration configuration, ::cuda::stream_ref stream);

        Domain(const Domain&)            = delete;
        Domain& operator=(const Domain&) = delete;
        Domain(Domain&&)                 = delete;
        Domain& operator=(Domain&&)      = delete;

        [[nodiscard]] simulation::ScalarField<float> allocate_collider_field(std::span<const float> values) const;

    private:
        std::vector<std::uint32_t> collider_indices;
    };
} // namespace physica::fluids::gas
