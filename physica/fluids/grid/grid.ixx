module;

#include <physica/cuda.h>

export module physica.fluids.grid;

import std;
export import physica.simulation.field;

export namespace physica::fluids::grid {
    struct Configuration final {
        std::array<std::uint32_t, 3u> resolution;
        float cell_size;
        Vector3<float> origin{};
        Vector3<float> velocity{};
    };

    struct Grid final {
        const Configuration configuration;
        const ::cuda::stream_ref stream;
        const std::size_t cell_count;
        const std::array<std::size_t, 3u> face_counts;

        Grid(Configuration next_configuration, const ::cuda::stream_ref stream)
            : configuration(std::move(next_configuration)), stream(stream), cell_count(static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * configuration.resolution[2]),
              face_counts{
                  static_cast<std::size_t>(configuration.resolution[0] + 1u) * configuration.resolution[1] * configuration.resolution[2],
                  static_cast<std::size_t>(configuration.resolution[0]) * (configuration.resolution[1] + 1u) * configuration.resolution[2],
                  static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * (configuration.resolution[2] + 1u),
              } {}

        Grid(const Grid&)            = delete;
        Grid& operator=(const Grid&) = delete;
        Grid(Grid&&)                 = delete;
        Grid& operator=(Grid&&)      = delete;

        template <class Value>
        [[nodiscard]] simulation::ScalarField<Value> allocate_cell_field() const {
            return simulation::ScalarField<Value>{stream, cell_count};
        }

        template <class Value>
        [[nodiscard]] simulation::VectorField<Value> allocate_cell_vector_field() const {
            return simulation::VectorField<Value>{stream, cell_count};
        }

        template <class Value>
        [[nodiscard]] simulation::VectorField<Value> allocate_mac_field() const {
            return simulation::VectorField<Value>{stream, face_counts};
        }

        template <class Field>
        void clear(Field& field) const {
            simulation::clear(stream, field);
        }

        template <class Field>
        void copy(const Field& source, Field& destination) const {
            simulation::copy(stream, source, destination);
        }
    };
} // namespace physica::fluids::grid
