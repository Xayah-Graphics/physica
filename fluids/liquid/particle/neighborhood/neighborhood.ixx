module;

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>

export module physica.fluids.liquid.particle.neighborhood;

import std;
import physica.fluids.liquid.particle.domain;

export namespace physica::fluids::liquid::particle {
    struct Neighborhood final {
        ::cuda::device_buffer<std::uint64_t> sorted_keys;
        ::cuda::device_buffer<std::uint32_t> sorted_particle_indices;
        ::cuda::device_buffer<std::uint32_t> cell_offsets;
        ::cuda::device_buffer<std::uint64_t> sorted_boundary_keys;
        ::cuda::device_buffer<std::uint32_t> sorted_boundary_indices;
        ::cuda::device_buffer<std::uint32_t> boundary_cell_offsets;
        std::array<std::uint32_t, 3u> cell_resolution;
        Vector3 cell_origin;
        float cell_size;
        float boundary_time;
    };

    struct NeighborhoodSearch final {
        explicit NeighborhoodSearch(const Domain& domain);

        NeighborhoodSearch(const NeighborhoodSearch&) = delete;
        NeighborhoodSearch& operator=(const NeighborhoodSearch&) = delete;
        NeighborhoodSearch(NeighborhoodSearch&&) = delete;
        NeighborhoodSearch& operator=(NeighborhoodSearch&&) = delete;

        [[nodiscard]] Neighborhood allocate() const;
        void build(std::uint64_t step_index, const VectorField& positions, Neighborhood& neighborhood);

    private:
        const Domain& domain;
        ::cuda::device_buffer<std::uint64_t> unsorted_keys;
        ::cuda::device_buffer<std::uint32_t> unsorted_particle_indices;
        ::cuda::device_buffer<std::uint64_t> unsorted_boundary_keys;
        ::cuda::device_buffer<std::uint32_t> unsorted_boundary_indices;
        ::cuda::device_buffer<std::byte> sort_scratch;
        ::cuda::device_buffer<std::byte> boundary_sort_scratch;
    };
} // namespace physica::fluids::liquid::particle
