module;

#include <physica/fluids/grid/interop.h>
#include "projection-kernels.h"
#include <physica/cuda.h>

module physica.fluids.liquid.pic;

import std;
import :projection;

namespace physica::fluids::liquid::pic {
    Projection::Projection(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    Projection::Workspace Projection::allocate_workspace(const Model& model) const {
        return {
            .diagonal                = model.grid.allocate_cell_field<float>(),
            .rhs                     = model.grid.allocate_cell_field<float>(),
            .residual                = model.grid.allocate_cell_field<float>(),
            .preconditioned_residual = model.grid.allocate_cell_field<float>(),
            .direction               = model.grid.allocate_cell_field<float>(),
            .matrix_direction        = model.grid.allocate_cell_field<float>(),
            .products                = model.grid.allocate_cell_field<float>(),
            .scalars                 = ::cuda::device_buffer<float>{model.grid.fields.stream, ::cuda::device_default_memory_pool(model.grid.fields.stream.device()), 7u, ::cuda::no_init},
            .state                   = ::cuda::device_buffer<std::uint32_t>{model.grid.fields.stream, ::cuda::device_default_memory_pool(model.grid.fields.stream.device()), 2u, ::cuda::no_init},
            .reduction_scratch       = ::cuda::device_buffer<std::byte>{model.grid.fields.stream, ::cuda::device_default_memory_pool(model.grid.fields.stream.device()), kernels::projection::reduction_storage_size(model.grid.cell_count), ::cuda::no_init},
        };
    }

    Projection::Diagnostics Projection::forward(const Model& model, const float time, const float time_step, const ScalarField<std::uint32_t>& cell_types, const ScalarField<float>& level_set, const VectorField<float>& input_velocity, VectorField<float>& output_velocity, ScalarField<float>& pressure, Workspace& workspace) const {
        kernels::projection::project(model.grid.fields.stream, grid::device::grid(model.grid.configuration, time), time_step, configuration.density, configuration.maximum_iterations, configuration.tolerance, cell_types.values.data(), level_set.values.data(), field::view(input_velocity), field::view(output_velocity), workspace.rhs.values.data(), pressure.values.data(), workspace.diagonal.values.data(), workspace.residual.values.data(), workspace.preconditioned_residual.values.data(), workspace.direction.values.data(), workspace.matrix_direction.values.data(), workspace.products.values.data(), workspace.scalars.data(), workspace.state.data(), workspace.reduction_scratch.data(), workspace.reduction_scratch.size());
        Diagnostics diagnostics;
        ::cuda::copy_bytes(model.grid.fields.stream, ::cuda::std::span{workspace.state.data() + 1u, 1u}, ::cuda::std::span{&diagnostics.iterations, 1u});
        ::cuda::copy_bytes(model.grid.fields.stream, ::cuda::std::span{workspace.scalars.data() + 6u, 1u}, ::cuda::std::span{&diagnostics.relative_residual, 1u});
        model.grid.fields.stream.sync();
        return diagnostics;
    }
} // namespace physica::fluids::liquid::pic
