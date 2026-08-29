module;

#include <physica/cuda.h>

export module physica.fluids.grid;

import std;
export import physica.field;

export namespace physica::fluids::grid {
    struct Configuration final {
        std::array<std::uint32_t, 3u> resolution;
        float cell_size;
        Vector3<float> origin{};
        Vector3<float> velocity{};
    };

    struct Grid final {
        const Configuration configuration;
        FieldContext fields;
        const std::size_t cell_count;
        const std::array<std::size_t, 3u> face_counts;

        Grid(Configuration next_configuration, const ::cuda::stream_ref stream)
            : configuration(std::move(next_configuration)), fields(stream), cell_count(static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * configuration.resolution[2]),
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
        [[nodiscard]] ScalarField<Value> allocate_cell_field() const {
            return fields.allocate_scalar_field<Value>(cell_count);
        }

        template <class Value>
        [[nodiscard]] VectorField<Value> allocate_cell_vector_field() const {
            return fields.allocate_vector_field<Value>(cell_count);
        }

        template <class Value>
        [[nodiscard]] VectorField<Value> allocate_mac_field() const {
            const ::cuda::stream_ref stream = fields.stream;
            return {
                .x = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[0], ::cuda::no_init},
                .y = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[1], ::cuda::no_init},
                .z = ::cuda::device_buffer<Value>{stream, ::cuda::device_default_memory_pool(stream.device()), face_counts[2], ::cuda::no_init},
            };
        }

        template <class Field>
        void clear(Field& field) const {
            fields.clear(field);
        }

        template <class Field>
        void copy(const Field& source, Field& destination) const {
            fields.copy(source, destination);
        }
    };
} // namespace physica::fluids::grid
