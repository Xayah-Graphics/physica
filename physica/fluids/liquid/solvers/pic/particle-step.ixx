module;

#include <physica/cuda.h>

export module physica.fluids.liquid.pic:particle_step;

import std;
import :model;

export namespace physica::fluids::liquid::pic {
    struct ParticleStep final {
        struct Configuration final {
            std::uint32_t minimum_per_cell{3u};
            std::uint32_t target_per_cell{8u};
            std::uint32_t maximum_per_cell{12u};
        };

        struct Diagnostics final {
            float kinetic_energy{};
            Vector3<float> linear_momentum{};
            Vector3<float> angular_momentum{};
        };

        struct Maintenance final {
            std::uint32_t survivor_count{};
            std::uint32_t seed_count{};

            [[nodiscard]] std::uint32_t particle_count() const {
                return survivor_count + seed_count;
            }
        };

        struct Workspace final {
            ScalarField<std::uint32_t> raw_counts;
            ScalarField<std::uint32_t> survivor_counts;
            ScalarField<std::uint32_t> keep_flags;
            ScalarField<std::uint32_t> destinations;
            ScalarField<std::uint32_t> seed_counts;
            ScalarField<std::uint32_t> seed_offsets;
            VectorField<float> compacted_positions;
            VectorField<float> compacted_velocities;
            ScalarField<float> speeds;
            ::cuda::device_buffer<float> diagnostic_values;
            ::cuda::device_buffer<float> diagnostic_output;
            ::cuda::device_buffer<float> reduction_output;
            ::cuda::device_buffer<std::uint32_t> maintenance_totals;
            ::cuda::device_buffer<std::byte> reduction_scratch;
            ::cuda::device_buffer<std::byte> scan_scratch;
        };

        explicit ParticleStep(Configuration configuration);

        [[nodiscard]] Workspace allocate_workspace(const Model& model) const;
        [[nodiscard]] float maximum_speed(const Model& model, std::uint32_t particle_count, const VectorField<float>& velocities, Workspace& workspace) const;
        [[nodiscard]] Diagnostics diagnostics(const Model& model, float time, std::uint32_t particle_count, float particle_mass, const VectorField<float>& positions, const VectorField<float>& velocities, Workspace& workspace) const;
        void advect(const Model& model, float time, float time_step, std::uint32_t particle_count, const VectorField<float>& grid_velocity, VectorField<float>& positions, VectorField<float>& velocities) const;
        [[nodiscard]] Maintenance plan_maintenance(const Model& model, float time, std::uint32_t particle_count, const VectorField<float>& positions, const ScalarField<std::uint32_t>& cell_types, const ScalarField<float>& level_set, Workspace& workspace) const;
        void compact_and_seed(const Model& model, float time, std::uint64_t seed, std::uint32_t particle_count, const Maintenance& maintenance, const VectorField<float>& positions, const VectorField<float>& velocities, const VectorField<float>& grid_velocity, Workspace& workspace) const;

        const Configuration configuration;
    };
} // namespace physica::fluids::liquid::pic
