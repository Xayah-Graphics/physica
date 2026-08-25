module;

#include <physica/cuda.h>

export module physica.fluids.gas.domain;

import std;

export namespace physica::fluids::gas {
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
        VelocityBoundaryMode mode{VelocityBoundaryMode::zero_gradient};
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
    };

    struct DomainConfiguration final {
        std::array<std::uint32_t, 3> resolution{32u, 48u, 32u};
        float cell_size{1.0F / 32.0F};
        float time_step{1.0F / 60.0F};
        VelocityBoundary velocity_boundary{};
        std::vector<Collider> colliders{};
    };

    template <class Value>
    struct CellField final {
        ::cuda::device_buffer<Value> values;
    };

    template <class Value>
    struct CenteredVectorField final {
        CellField<Value> x;
        CellField<Value> y;
        CellField<Value> z;
    };

    template <class Value>
    struct StaggeredVectorField final {
        ::cuda::device_buffer<Value> x;
        ::cuda::device_buffer<Value> y;
        ::cuda::device_buffer<Value> z;
    };

    struct Domain final {
        const DomainConfiguration configuration;
        ::cuda::stream_ref stream;
        const std::size_t cell_count;
        const std::array<std::size_t, 3> face_counts;
        CellField<std::uint32_t> collider_ids;
        StaggeredVectorField<float> collider_velocity;
        std::uint32_t first_fluid_cell{};

        Domain(DomainConfiguration configuration, ::cuda::stream_ref stream);

        Domain(const Domain&)            = delete;
        Domain& operator=(const Domain&) = delete;
        Domain(Domain&&)                 = delete;
        Domain& operator=(Domain&&)      = delete;

        template <class Value>
        [[nodiscard]] CellField<Value> allocate_cell_field() const {
            return {.values = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), cell_count, ::cuda::no_init}};
        }

        template <class Value>
        [[nodiscard]] CenteredVectorField<Value> allocate_centered_vector_field() const {
            return {.x = allocate_cell_field<Value>(), .y = allocate_cell_field<Value>(), .z = allocate_cell_field<Value>()};
        }

        template <class Value>
        [[nodiscard]] StaggeredVectorField<Value> allocate_staggered_vector_field() const {
            return {
                .x = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[0], ::cuda::no_init},
                .y = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[1], ::cuda::no_init},
                .z = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[2], ::cuda::no_init},
            };
        }

        [[nodiscard]] CellField<float> allocate_collider_field(std::span<const float> values) const;

        template <class Value>
        void clear(CellField<Value>& field) const {
            ::cuda::fill_bytes(stream, field.values, 0u);
        }

        template <class Value>
        void clear(CenteredVectorField<Value>& field) const {
            clear(field.x);
            clear(field.y);
            clear(field.z);
        }

        template <class Value>
        void clear(StaggeredVectorField<Value>& field) const {
            ::cuda::fill_bytes(stream, field.x, 0u);
            ::cuda::fill_bytes(stream, field.y, 0u);
            ::cuda::fill_bytes(stream, field.z, 0u);
        }

        template <class Value>
        void copy(const CellField<Value>& source, CellField<Value>& destination) const {
            ::cuda::copy_bytes(stream, source.values, destination.values);
        }

        template <class Value>
        void copy(const CenteredVectorField<Value>& source, CenteredVectorField<Value>& destination) const {
            copy(source.x, destination.x);
            copy(source.y, destination.y);
            copy(source.z, destination.z);
        }

        template <class Value>
        void copy(const StaggeredVectorField<Value>& source, StaggeredVectorField<Value>& destination) const {
            ::cuda::copy_bytes(stream, source.x, destination.x);
            ::cuda::copy_bytes(stream, source.y, destination.y);
            ::cuda::copy_bytes(stream, source.z, destination.z);
        }

    private:
        std::vector<std::uint32_t> collider_indices;
    };
} // namespace physica::fluids::gas
