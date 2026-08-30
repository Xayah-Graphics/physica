module;

#include "control-kernels.h"
#include <fluids/gas/interop.h>
#include <physica/cuda.h>

module physica.fluids.gas.solvers.keyframe_smoke.control;

import std;

namespace physica::fluids::gas::solvers::keyframe_smoke {
    ControlSystem::ControlSystem(const Domain& domain, ControlConfiguration next_configuration) : configuration(std::move(next_configuration)), parameter_values(configuration.winds.size() * 6u + configuration.vortices.size() * 4u), lower_bounds(parameter_values.size()), upper_bounds(parameter_values.size()), device_winds(domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), configuration.winds.size() * sizeof(kernels::WindData), ::cuda::no_init), device_vortices(domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), configuration.vortices.size() * sizeof(kernels::VortexData), ::cuda::no_init) {
        std::vector<kernels::WindData> winds;
        std::vector<kernels::VortexData> vortices;
        winds.reserve(configuration.winds.size());
        vortices.reserve(configuration.vortices.size());
        std::uint32_t parameter_offset = 0u;
        for (const WindDefinition& wind : configuration.winds) {
            winds.push_back({.begin_step = wind.begin_step, .end_step = wind.end_step, .width = wind.width, .parameter_offset = parameter_offset});
            for (const BoundedValue value : wind.center) {
                parameter_values[parameter_offset] = value.initial;
                lower_bounds[parameter_offset]     = value.lower;
                upper_bounds[parameter_offset]     = value.upper;
                ++parameter_offset;
            }
            for (const BoundedValue value : wind.vector) {
                parameter_values[parameter_offset] = value.initial;
                lower_bounds[parameter_offset]     = value.lower;
                upper_bounds[parameter_offset]     = value.upper;
                ++parameter_offset;
            }
        }
        for (const VortexDefinition& vortex : configuration.vortices) {
            vortices.push_back({.begin_step = vortex.begin_step, .end_step = vortex.end_step, .width = vortex.width, .axis = {.x = vortex.axis.x, .y = vortex.axis.y, .z = vortex.axis.z}, .parameter_offset = parameter_offset});
            for (const BoundedValue value : vortex.center) {
                parameter_values[parameter_offset] = value.initial;
                lower_bounds[parameter_offset]     = value.lower;
                upper_bounds[parameter_offset]     = value.upper;
                ++parameter_offset;
            }
            parameter_values[parameter_offset] = vortex.strength.initial;
            lower_bounds[parameter_offset]     = vortex.strength.lower;
            upper_bounds[parameter_offset]     = vortex.strength.upper;
            ++parameter_offset;
        }
        ::cuda::copy_bytes(domain.grid.stream, std::as_bytes(std::span{winds}), device_winds);
        ::cuda::copy_bytes(domain.grid.stream, std::as_bytes(std::span{vortices}), device_vortices);
    }

    void ControlSystem::forward(const Domain& domain, const std::uint32_t step, const ::cuda::std::span<const double> parameters, DenseControl& output) const {
        kernels::control_forward(domain.grid.stream, device::discretization(domain.configuration), step, reinterpret_cast<const kernels::WindData*>(device_winds.data()), static_cast<std::uint32_t>(configuration.winds.size()), reinterpret_cast<const kernels::VortexData*>(device_vortices.data()), static_cast<std::uint32_t>(configuration.vortices.size()), parameters.data(), simulation::view(output.force));
    }

    void ControlSystem::jvp(const Domain& domain, const std::uint32_t step, const ::cuda::std::span<const double> parameters, const ::cuda::std::span<const double> direction, DenseControlTangent& output_tangent) const {
        kernels::control_jvp(domain.grid.stream, device::discretization(domain.configuration), step, reinterpret_cast<const kernels::WindData*>(device_winds.data()), static_cast<std::uint32_t>(configuration.winds.size()), reinterpret_cast<const kernels::VortexData*>(device_vortices.data()), static_cast<std::uint32_t>(configuration.vortices.size()), parameters.data(), direction.data(), simulation::view(output_tangent.force));
    }

    void ControlSystem::vjp(const Domain& domain, const std::uint32_t step, const ::cuda::std::span<const double> parameters, const DenseControlAdjoint& output_adjoint, const ::cuda::std::span<double> gradient) const {
        kernels::control_vjp(domain.grid.stream, device::discretization(domain.configuration), step, reinterpret_cast<const kernels::WindData*>(device_winds.data()), static_cast<std::uint32_t>(configuration.winds.size()), reinterpret_cast<const kernels::VortexData*>(device_vortices.data()), static_cast<std::uint32_t>(configuration.vortices.size()), parameters.data(), simulation::view(output_adjoint.force), gradient.data());
    }

    std::vector<std::uint8_t> ControlSystem::active_parameters(const std::uint32_t begin_step, const std::uint32_t end_step) const {
        std::vector<std::uint8_t> result(parameter_values.size(), 0u);
        std::size_t parameter_offset = 0u;
        for (const WindDefinition& wind : configuration.winds) {
            if (wind.begin_step < end_step && wind.end_step > begin_step)
                for (std::size_t parameter = 0u; parameter < 6u; ++parameter) result[parameter_offset + parameter] = 1u;
            parameter_offset += 6u;
        }
        for (const VortexDefinition& vortex : configuration.vortices) {
            if (vortex.begin_step < end_step && vortex.end_step > begin_step)
                for (std::size_t parameter = 0u; parameter < 4u; ++parameter) result[parameter_offset + parameter] = 1u;
            parameter_offset += 4u;
        }
        return result;
    }
} // namespace physica::fluids::gas::solvers::keyframe_smoke
