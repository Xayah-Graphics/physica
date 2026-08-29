module;

#include <physica/cuda.h>

export module physica.fluids.liquid.operators.neighborhood;

import std;
import physica.fluids.liquid.meshfree;

export namespace physica::fluids::liquid::operators {
    struct Neighborhood final {
        ::cuda::device_buffer<std::uint64_t> sorted_keys;
        ::cuda::device_buffer<std::uint32_t> sorted_particle_indices;
        ::cuda::device_buffer<std::uint32_t> cell_offsets;
        ::cuda::device_buffer<std::uint64_t> sorted_boundary_keys;
        ::cuda::device_buffer<std::uint32_t> sorted_boundary_indices;
        ::cuda::device_buffer<std::uint32_t> boundary_cell_offsets;
        std::array<std::uint32_t, 3u> cell_resolution;
        Vector3<float> cell_origin;
        float cell_size;
        float boundary_time;
    };

    struct UniformGridNeighborhood final {
        struct Workspace final {
            ::cuda::device_buffer<std::uint64_t> unsorted_keys;
            ::cuda::device_buffer<std::uint32_t> unsorted_particle_indices;
            ::cuda::device_buffer<std::uint64_t> unsorted_boundary_keys;
            ::cuda::device_buffer<std::uint32_t> unsorted_boundary_indices;
            ::cuda::device_buffer<std::byte> sort_scratch;
            ::cuda::device_buffer<std::byte> boundary_sort_scratch;
        };

        [[nodiscard]] Neighborhood allocate_cache(const meshfree::Model& model) const;
        [[nodiscard]] Workspace allocate_workspace(const meshfree::Model& model) const;
        void build(const meshfree::Model& model, std::uint64_t step_index, const VectorField<float>& positions, Neighborhood& neighborhood, Workspace& workspace) const;
    };
} // namespace physica::fluids::liquid::operators
