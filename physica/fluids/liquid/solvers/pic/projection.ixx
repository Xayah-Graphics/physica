module;

#include <physica/cuda.h>

export module physica.fluids.liquid.solvers.pic.projection;

import std;
import physica.fluids.liquid.solvers.pic.model;

export namespace physica::fluids::liquid::solvers::pic {
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
            simulation::ScalarField<float> diagonal;
            simulation::ScalarField<float> rhs;
            simulation::ScalarField<float> residual;
            simulation::ScalarField<float> preconditioned_residual;
            simulation::ScalarField<float> direction;
            simulation::ScalarField<float> matrix_direction;
            simulation::ScalarField<float> products;
            ::cuda::device_buffer<float> scalars;
            ::cuda::device_buffer<std::uint32_t> state;
            ::cuda::device_buffer<std::byte> reduction_scratch;
        };

        explicit Projection(Configuration configuration);

        [[nodiscard]] Workspace allocate_workspace(const Model& model) const;
        [[nodiscard]] Diagnostics forward(const Model& model, float time, float time_step, const simulation::ScalarField<std::uint32_t>& cell_types, const simulation::ScalarField<float>& level_set, const simulation::VectorField<float>& input_velocity, simulation::VectorField<float>& output_velocity, simulation::ScalarField<float>& pressure, Workspace& workspace) const;

        const Configuration configuration;
    };
} // namespace physica::fluids::liquid::solvers::pic
