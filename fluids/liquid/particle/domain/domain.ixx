module;

#include <physica/cuda.h>

export module physica.fluids.liquid.particle.domain;

import std;

export namespace physica::fluids::liquid::particle {
    struct Vector3 final {
        float x;
        float y;
        float z;
    };

    struct BoxBoundary final {
        Vector3 minimum;
        Vector3 maximum;
        Vector3 velocity;
        bool no_slip{true};
    };

    struct BoundaryParticle final {
        Vector3 position;
        Vector3 velocity;
        float volume;
    };

    struct DomainConfiguration final {
        std::uint32_t particle_count;
        float time_step;
        float support_radius;
        float particle_radius;
        Vector3 gravity;
        BoxBoundary boundary;
        std::vector<BoundaryParticle> boundary_particles;
    };

    enum class ExecutionMode : std::uint32_t {
        forward,
        differentiable,
    };

    struct ScalarField final {
        ::cuda::device_buffer<float> values;
    };

    struct ScalarAdjointField final {
        ::cuda::device_buffer<double> values;
    };

    struct VectorField final {
        ::cuda::device_buffer<float> x;
        ::cuda::device_buffer<float> y;
        ::cuda::device_buffer<float> z;
    };

    struct VectorAdjointField final {
        ::cuda::device_buffer<double> x;
        ::cuda::device_buffer<double> y;
        ::cuda::device_buffer<double> z;
    };

    struct ParticleState final {
        VectorField positions;
        VectorField velocities;
        std::uint64_t step_index{};
    };

    struct Control final {
        VectorField external_accelerations;
    };

    struct ParticleStateTangent final {
        VectorField positions;
        VectorField velocities;
    };

    struct ControlTangent final {
        VectorField external_accelerations;
    };

    struct ParticleStateAdjoint final {
        VectorAdjointField positions;
        VectorAdjointField velocities;
    };

    struct ControlAdjoint final {
        VectorAdjointField external_accelerations;
    };

    struct ParticleParameters final {
        ::cuda::device_buffer<float> masses;
        ::cuda::device_buffer<float> rest_densities;
        ::cuda::device_buffer<float> viscosities;
        ::cuda::device_buffer<float> surface_tensions;
    };

    struct ParticleParameterTangent final {
        ::cuda::device_buffer<float> masses;
        ::cuda::device_buffer<float> rest_densities;
        ::cuda::device_buffer<float> viscosities;
        ::cuda::device_buffer<float> surface_tensions;
    };

    struct ParticleParameterAdjoint final {
        ::cuda::device_buffer<double> masses;
        ::cuda::device_buffer<double> rest_densities;
        ::cuda::device_buffer<double> viscosities;
        ::cuda::device_buffer<double> surface_tensions;
    };

    struct DeviceBoundary final {
        VectorField positions;
        VectorField velocities;
        ScalarField volumes;
    };

    struct Domain final {
        const DomainConfiguration configuration;
        ::cuda::stream_ref stream;
        DeviceBoundary boundary;

        Domain(DomainConfiguration configuration, ::cuda::stream_ref stream);

        Domain(const Domain&) = delete;
        Domain& operator=(const Domain&) = delete;
        Domain(Domain&&) = delete;
        Domain& operator=(Domain&&) = delete;

        [[nodiscard]] ScalarField allocate_scalar_field(std::size_t count) const;
        [[nodiscard]] ScalarAdjointField allocate_scalar_adjoint_field(std::size_t count) const;
        [[nodiscard]] VectorField allocate_vector_field(std::size_t count) const;
        [[nodiscard]] VectorAdjointField allocate_vector_adjoint_field(std::size_t count) const;
        [[nodiscard]] ParticleState allocate_particle_state() const;
        [[nodiscard]] Control allocate_control() const;
        [[nodiscard]] ParticleStateTangent allocate_particle_state_tangent() const;
        [[nodiscard]] ControlTangent allocate_control_tangent() const;
        [[nodiscard]] ParticleStateAdjoint allocate_particle_state_adjoint() const;
        [[nodiscard]] ControlAdjoint allocate_control_adjoint() const;
        [[nodiscard]] ParticleParameters allocate_particle_parameters() const;
        [[nodiscard]] ParticleParameterTangent allocate_particle_parameter_tangent() const;
        [[nodiscard]] ParticleParameterAdjoint allocate_particle_parameter_adjoint() const;

        void clear(ScalarField& field) const;
        void clear(ScalarAdjointField& field) const;
        void clear(VectorField& field) const;
        void clear(VectorAdjointField& field) const;
        void clear(ParticleParameterTangent& tangent) const;
        void clear(ParticleParameterAdjoint& adjoint) const;
        void copy(const ScalarField& source, ScalarField& destination) const;
        void copy(const ScalarAdjointField& source, ScalarAdjointField& destination) const;
        void copy(const VectorField& source, VectorField& destination) const;
        void copy(const VectorAdjointField& source, VectorAdjointField& destination) const;
        void copy(const ParticleState& source, ParticleState& destination) const;
        void copy(const ParticleStateTangent& source, ParticleStateTangent& destination) const;
        void copy(const ParticleStateAdjoint& source, ParticleStateAdjoint& destination) const;
        void accumulate(const ScalarAdjointField& source, ScalarAdjointField& destination) const;
        void accumulate(const VectorAdjointField& source, VectorAdjointField& destination) const;
        void accumulate(const ParticleStateAdjoint& source, ParticleStateAdjoint& destination) const;
        void upload(std::span<const float> source, ScalarField& destination) const;
        void upload(std::span<const Vector3> source, VectorField& destination) const;
    };
} // namespace physica::fluids::liquid::particle
