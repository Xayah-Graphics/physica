module;

#include "control-kernels.h"
#include <fluids/gas/interop.h>
#include <physica/cuda.h>

module physica.fluids.gas.solvers.adjoint_control.control;

import std;

namespace physica::fluids::gas::solvers::adjoint_control {
    ControlSystem::ControlSystem(ControlConfiguration next_configuration) : configuration(std::move(next_configuration)), center_count(configuration.lattice[0] * configuration.lattice[1] * configuration.lattice[2]), parameter_values(static_cast<std::size_t>(configuration.step_count) * center_count * 3u), lower_bounds(parameter_values.size(), configuration.lower_bound), upper_bounds(parameter_values.size(), configuration.upper_bound) {}

    void ControlSystem::forward(const Domain& domain, const std::uint32_t step, const ::cuda::std::span<const double> parameters, DenseControl& output) const {
        kernels::control_forward(domain.grid.stream, device::discretization(domain.configuration), step, {configuration.lattice[0], configuration.lattice[1], configuration.lattice[2]}, configuration.gaussian_sigma, configuration.step_count, parameters.data(), simulation::view(output.force));
    }

    void ControlSystem::jvp(const Domain& domain, const std::uint32_t step, const ::cuda::std::span<const double> direction, DenseControlTangent& output_tangent) const {
        kernels::control_jvp(domain.grid.stream, device::discretization(domain.configuration), step, {configuration.lattice[0], configuration.lattice[1], configuration.lattice[2]}, configuration.gaussian_sigma, configuration.step_count, direction.data(), simulation::view(output_tangent.force));
    }

    void ControlSystem::vjp(const Domain& domain, const std::uint32_t step, const DenseControlAdjoint& output_adjoint, const ::cuda::std::span<double> gradient) const {
        kernels::control_vjp(domain.grid.stream, device::discretization(domain.configuration), step, {configuration.lattice[0], configuration.lattice[1], configuration.lattice[2]}, configuration.gaussian_sigma, configuration.step_count, simulation::view(output_adjoint.force), gradient.data());
    }

    std::vector<std::uint8_t> ControlSystem::active_parameters(const std::uint32_t begin_step, const std::uint32_t end_step) const {
        std::vector<std::uint8_t> result(parameter_values.size());
        const std::size_t parameters_per_step = static_cast<std::size_t>(center_count) * 3u;
        const std::uint32_t active_end        = std::min(end_step, configuration.step_count);
        for (std::uint32_t step = begin_step; step < active_end; ++step) std::ranges::fill(std::span{result}.subspan(static_cast<std::size_t>(step) * parameters_per_step, parameters_per_step), 1u);
        return result;
    }
} // namespace physica::fluids::gas::solvers::adjoint_control
