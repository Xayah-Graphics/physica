module;

#include <physica/cuda.h>

export module physica.fluids.liquid.pic:projection;

import std;
import :model;

export namespace physica::fluids::liquid::pic {
    struct Projection final {
        struct Configuration final {
            float density{1000.0F};
            std::uint32_t maximum_iterations{120u};
            float tolerance{1.0e-5F};
        };

        struct Diagnostics final {
            std::uint32_t iterations{};
            float relative_residual{};
        };

        struct Workspace final {
            ScalarField<float> diagonal;
            ScalarField<float> rhs;
            ScalarField<float> residual;
            ScalarField<float> preconditioned_residual;
            ScalarField<float> direction;
            ScalarField<float> matrix_direction;
            ScalarField<float> products;
            ::cuda::device_buffer<float> scalars;
            ::cuda::device_buffer<std::uint32_t> state;
            ::cuda::device_buffer<std::byte> reduction_scratch;
        };

        explicit Projection(Configuration configuration);

        [[nodiscard]] Workspace allocate_workspace(const Model& model) const;
        [[nodiscard]] Diagnostics forward(const Model& model, float time, float time_step, const ScalarField<std::uint32_t>& cell_types, const ScalarField<float>& level_set, const VectorField<float>& input_velocity, VectorField<float>& output_velocity, ScalarField<float>& pressure, Workspace& workspace) const;

        const Configuration configuration;
    };
} // namespace physica::fluids::liquid::pic
