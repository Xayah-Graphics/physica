module;

#include <physica/cuda.h>

export module physica.deformables.cloth.domain;

import std;

export namespace physica::deformables::cloth {
    struct Vector3 final {
        float x;
        float y;
        float z;
    };

    struct Triangle final {
        std::uint32_t first;
        std::uint32_t second;
        std::uint32_t third;
    };

    struct DomainConfiguration final {
        std::vector<Vector3> rest_positions;
        std::vector<Triangle> triangles;
        std::vector<std::optional<Vector3>> anchors;
    };

    struct IndexField final {
        ::cuda::device_buffer<std::uint32_t> values;
    };

    struct ScalarField final {
        ::cuda::device_buffer<float> values;
    };

    struct ScalarAdjointField final {
        ::cuda::device_buffer<double> values;
    };

    struct VectorField final {
        ScalarField x;
        ScalarField y;
        ScalarField z;
    };

    struct VectorAdjointField final {
        ScalarAdjointField x;
        ScalarAdjointField y;
        ScalarAdjointField z;
    };

    struct State final {
        VectorField positions;
        VectorField velocities;
    };

    struct Control final {
        VectorField external_forces;
    };

    struct StateTangent final {
        VectorField positions;
        VectorField velocities;
    };

    struct ControlTangent final {
        VectorField external_forces;
    };

    struct StateAdjoint final {
        VectorAdjointField positions;
        VectorAdjointField velocities;
    };

    struct ControlAdjoint final {
        VectorAdjointField external_forces;
    };

    enum class ExecutionMode : std::uint32_t {
        forward,
        differentiable,
    };

    struct Domain final {
        const DomainConfiguration configuration;
        ::cuda::stream_ref stream;
        std::size_t particle_count;

        Domain(DomainConfiguration configuration, ::cuda::stream_ref stream);

        Domain(const Domain&) = delete;
        Domain& operator=(const Domain&) = delete;
        Domain(Domain&&) = delete;
        Domain& operator=(Domain&&) = delete;

        [[nodiscard]] IndexField allocate_index_field(std::size_t size) const;
        [[nodiscard]] ScalarField allocate_scalar_field(std::size_t size) const;
        [[nodiscard]] ScalarAdjointField allocate_scalar_adjoint_field(std::size_t size) const;
        [[nodiscard]] VectorField allocate_vector_field() const;
        [[nodiscard]] VectorAdjointField allocate_vector_adjoint_field() const;

        void clear(ScalarField& field) const;
        void clear(ScalarAdjointField& field) const;
        void clear(VectorField& field) const;
        void clear(VectorAdjointField& field) const;
        void copy(const VectorField& source, VectorField& destination) const;
        void copy(const VectorAdjointField& source, VectorAdjointField& destination) const;
        void accumulate(const VectorAdjointField& source, VectorAdjointField& destination) const;
        void upload(std::span<const Vector3> source, VectorField& destination) const;
    };
} // namespace physica::deformables::cloth
