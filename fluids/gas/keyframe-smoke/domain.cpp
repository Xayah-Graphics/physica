module;

#include "domain_kernels.h"
#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>

module physica.fluids.gas.keyframe_smoke.domain;

import std;

namespace physica::fluids::gas::keyframe_smoke {
    Domain::Domain(DomainConfiguration next_configuration, const ::cuda::stream_ref source_stream)
        : configuration(std::move(next_configuration)),
          stream(source_stream),
          cell_count(static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * configuration.resolution[2]),
          face_counts{
              static_cast<std::size_t>(configuration.resolution[0] + 1u) * configuration.resolution[1] * configuration.resolution[2],
              static_cast<std::size_t>(configuration.resolution[0]) * (configuration.resolution[1] + 1u) * configuration.resolution[2],
              static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * (configuration.resolution[2] + 1u),
          },
          pressure_anchor([&] {
              const std::array faces{configuration.pressure_boundary.x_min, configuration.pressure_boundary.x_max, configuration.pressure_boundary.y_min, configuration.pressure_boundary.y_max, configuration.pressure_boundary.z_min, configuration.pressure_boundary.z_max};
              return std::ranges::any_of(faces, [](const ScalarBoundaryFace& face) { return face.mode == ScalarBoundaryMode::fixed_value; }) ? static_cast<std::uint32_t>(cell_count) : 0u;
          }()) {}

    ScalarField Domain::allocate_scalar_field() const {
        return {.values = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), cell_count, ::cuda::no_init}};
    }

    CenteredVectorField Domain::allocate_centered_vector_field() const {
        return {.x = allocate_scalar_field(), .y = allocate_scalar_field(), .z = allocate_scalar_field()};
    }

    StaggeredVectorField Domain::allocate_staggered_vector_field() const {
        return {
            .x = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[0], ::cuda::no_init},
            .y = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[1], ::cuda::no_init},
            .z = ::cuda::device_buffer<float>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[2], ::cuda::no_init},
        };
    }

    ScalarAdjointField Domain::allocate_scalar_adjoint_field() const {
        return {.values = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), cell_count, ::cuda::no_init}};
    }

    CenteredVectorAdjointField Domain::allocate_centered_vector_adjoint_field() const {
        return {.x = allocate_scalar_adjoint_field(), .y = allocate_scalar_adjoint_field(), .z = allocate_scalar_adjoint_field()};
    }

    StaggeredVectorAdjointField Domain::allocate_staggered_vector_adjoint_field() const {
        return {
            .x = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[0], ::cuda::no_init},
            .y = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[1], ::cuda::no_init},
            .z = ::cuda::device_buffer<double>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[2], ::cuda::no_init},
        };
    }

    void Domain::clear(ScalarField& field) const { ::cuda::fill_bytes(stream, field.values, 0u); }
    void Domain::clear(CenteredVectorField& field) const {
        clear(field.x);
        clear(field.y);
        clear(field.z);
    }
    void Domain::clear(StaggeredVectorField& field) const {
        ::cuda::fill_bytes(stream, field.x, 0u);
        ::cuda::fill_bytes(stream, field.y, 0u);
        ::cuda::fill_bytes(stream, field.z, 0u);
    }
    void Domain::clear(ScalarAdjointField& field) const { ::cuda::fill_bytes(stream, field.values, 0u); }
    void Domain::clear(CenteredVectorAdjointField& field) const {
        clear(field.x);
        clear(field.y);
        clear(field.z);
    }
    void Domain::clear(StaggeredVectorAdjointField& field) const {
        ::cuda::fill_bytes(stream, field.x, 0u);
        ::cuda::fill_bytes(stream, field.y, 0u);
        ::cuda::fill_bytes(stream, field.z, 0u);
    }

    void Domain::copy(const ScalarField& source, ScalarField& destination) const { ::cuda::copy_bytes(stream, source.values, destination.values); }
    void Domain::copy(const CenteredVectorField& source, CenteredVectorField& destination) const {
        copy(source.x, destination.x);
        copy(source.y, destination.y);
        copy(source.z, destination.z);
    }
    void Domain::copy(const StaggeredVectorField& source, StaggeredVectorField& destination) const {
        ::cuda::copy_bytes(stream, source.x, destination.x);
        ::cuda::copy_bytes(stream, source.y, destination.y);
        ::cuda::copy_bytes(stream, source.z, destination.z);
    }
    void Domain::copy(const ScalarAdjointField& source, ScalarAdjointField& destination) const { ::cuda::copy_bytes(stream, source.values, destination.values); }
    void Domain::copy(const CenteredVectorAdjointField& source, CenteredVectorAdjointField& destination) const {
        copy(source.x, destination.x);
        copy(source.y, destination.y);
        copy(source.z, destination.z);
    }
    void Domain::copy(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) const {
        ::cuda::copy_bytes(stream, source.x, destination.x);
        ::cuda::copy_bytes(stream, source.y, destination.y);
        ::cuda::copy_bytes(stream, source.z, destination.z);
    }

    void Domain::accumulate(const ScalarAdjointField& source, ScalarAdjointField& destination) const { cuda_detail::accumulate(stream, source.values.data(), destination.values.data(), source.values.size()); }
    void Domain::accumulate(const CenteredVectorAdjointField& source, CenteredVectorAdjointField& destination) const {
        accumulate(source.x, destination.x);
        accumulate(source.y, destination.y);
        accumulate(source.z, destination.z);
    }
    void Domain::accumulate(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) const {
        cuda_detail::accumulate(stream, source.x.data(), destination.x.data(), source.x.size());
        cuda_detail::accumulate(stream, source.y.data(), destination.y.data(), source.y.size());
        cuda_detail::accumulate(stream, source.z.data(), destination.z.data(), source.z.size());
    }
} // namespace physica::fluids::gas::keyframe_smoke
