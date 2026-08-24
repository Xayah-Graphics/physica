module;

#include "control_kernels.h"
#include "interop.h"
#include <physica/cuda.h>

module physica.fluids.gas.keyframe_smoke.control;

import std;

namespace physica::fluids::gas::keyframe_smoke {
    ControlSystem::ControlSystem(const Domain& domain, ControlConfiguration next_configuration)
        : configuration(std::move(next_configuration)),
          parameters([&] {
              ParameterSet result;
              for (const WindDefinition& wind : configuration.winds) {
                  for (const BoundedValue value : wind.center) {
                      result.values.push_back(value.initial);
                      result.lower_bounds.push_back(value.lower);
                      result.upper_bounds.push_back(value.upper);
                  }
                  for (const BoundedValue value : wind.vector) {
                      result.values.push_back(value.initial);
                      result.lower_bounds.push_back(value.lower);
                      result.upper_bounds.push_back(value.upper);
                  }
              }
              for (const VortexDefinition& vortex : configuration.vortices) {
                  for (const BoundedValue value : vortex.center) {
                      result.values.push_back(value.initial);
                      result.lower_bounds.push_back(value.lower);
                      result.upper_bounds.push_back(value.upper);
                  }
                  result.values.push_back(vortex.strength.initial);
                  result.lower_bounds.push_back(vortex.strength.lower);
                  result.upper_bounds.push_back(vortex.strength.upper);
              }
              return result;
          }()),
          device_parameters(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), parameters.values.size(), ::cuda::no_init),
          device_direction(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), parameters.values.size(), ::cuda::no_init),
          device_gradient(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), parameters.values.size(), ::cuda::no_init),
          device_winds(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), configuration.winds.size() * sizeof(cuda_detail::WindData), ::cuda::no_init),
          device_vortices(domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), configuration.vortices.size() * sizeof(cuda_detail::VortexData), ::cuda::no_init) {
        std::vector<cuda_detail::WindData> winds;
        std::vector<cuda_detail::VortexData> vortices;
        winds.reserve(configuration.winds.size());
        vortices.reserve(configuration.vortices.size());
        std::uint32_t parameter_offset = 0u;
        for (const WindDefinition& wind : configuration.winds) {
            winds.push_back({.begin_step = wind.begin_step, .end_step = wind.end_step, .width = wind.width, .parameter_offset = parameter_offset});
            parameter_offset += 6u;
        }
        for (const VortexDefinition& vortex : configuration.vortices) {
            vortices.push_back({.begin_step = vortex.begin_step, .end_step = vortex.end_step, .width = vortex.width, .axis = {.x = vortex.axis.x, .y = vortex.axis.y, .z = vortex.axis.z}, .parameter_offset = parameter_offset});
            parameter_offset += 4u;
        }
        ::cuda::copy_bytes(domain.stream, std::as_bytes(std::span{winds}), device_winds);
        ::cuda::copy_bytes(domain.stream, std::as_bytes(std::span{vortices}), device_vortices);
        upload_parameters(domain, parameters.values);
        ::cuda::fill_bytes(domain.stream, device_direction, 0u);
        ::cuda::fill_bytes(domain.stream, device_gradient, 0u);
    }

    void ControlSystem::upload_parameters(const Domain& domain, const std::span<const double> values) {
        ::cuda::copy_bytes(domain.stream, values, device_parameters);
    }

    void ControlSystem::upload_direction(const Domain& domain, const std::span<const double> direction) {
        ::cuda::copy_bytes(domain.stream, direction, device_direction);
    }

    void ControlSystem::clear_gradient(const Domain& domain) { ::cuda::fill_bytes(domain.stream, device_gradient, 0u); }

    void ControlSystem::forward(const Domain& domain, const std::uint32_t step, DenseControl& output) const {
        cuda_detail::control_forward(domain.stream, cuda_detail::grid(domain.configuration), step, reinterpret_cast<const cuda_detail::WindData*>(device_winds.data()), static_cast<std::uint32_t>(configuration.winds.size()), reinterpret_cast<const cuda_detail::VortexData*>(device_vortices.data()), static_cast<std::uint32_t>(configuration.vortices.size()), device_parameters.data(), cuda_detail::centered(output.force));
    }

    void ControlSystem::jvp(const Domain& domain, const std::uint32_t step, DenseControlTangent& output_tangent) const {
        cuda_detail::control_jvp(domain.stream, cuda_detail::grid(domain.configuration), step, reinterpret_cast<const cuda_detail::WindData*>(device_winds.data()), static_cast<std::uint32_t>(configuration.winds.size()), reinterpret_cast<const cuda_detail::VortexData*>(device_vortices.data()), static_cast<std::uint32_t>(configuration.vortices.size()), device_parameters.data(), device_direction.data(), cuda_detail::centered(output_tangent.force));
    }

    void ControlSystem::vjp(const Domain& domain, const std::uint32_t step, const DenseControlAdjoint& output_adjoint) {
        cuda_detail::control_vjp(domain.stream, cuda_detail::grid(domain.configuration), step, reinterpret_cast<const cuda_detail::WindData*>(device_winds.data()), static_cast<std::uint32_t>(configuration.winds.size()), reinterpret_cast<const cuda_detail::VortexData*>(device_vortices.data()), static_cast<std::uint32_t>(configuration.vortices.size()), device_parameters.data(), cuda_detail::centered_adjoint(output_adjoint.force), device_gradient.data());
    }

    void ControlSystem::download_gradient(const Domain& domain, const std::span<double> gradient) const {
        ::cuda::copy_bytes(domain.stream, device_gradient, gradient);
        domain.stream.sync();
    }

    std::vector<ControlGlyph> ControlSystem::glyphs(const std::uint32_t step, const std::span<const double> values) const {
        std::vector<ControlGlyph> result;
        std::size_t parameter_offset = 0u;
        for (const WindDefinition& wind : configuration.winds) {
            if (step >= wind.begin_step && step < wind.end_step) result.push_back({
                .kind = ControlKind::wind,
                .begin_step = wind.begin_step,
                .end_step = wind.end_step,
                .center = {static_cast<float>(values[parameter_offset]), static_cast<float>(values[parameter_offset + 1u]), static_cast<float>(values[parameter_offset + 2u])},
                .direction = {static_cast<float>(values[parameter_offset + 3u]), static_cast<float>(values[parameter_offset + 4u]), static_cast<float>(values[parameter_offset + 5u])},
                .width = wind.width,
                .magnitude = static_cast<float>(std::sqrt(values[parameter_offset + 3u] * values[parameter_offset + 3u] + values[parameter_offset + 4u] * values[parameter_offset + 4u] + values[parameter_offset + 5u] * values[parameter_offset + 5u])),
            });
            parameter_offset += 6u;
        }
        for (const VortexDefinition& vortex : configuration.vortices) {
            if (step >= vortex.begin_step && step < vortex.end_step) result.push_back({
                .kind = ControlKind::vortex,
                .begin_step = vortex.begin_step,
                .end_step = vortex.end_step,
                .center = {static_cast<float>(values[parameter_offset]), static_cast<float>(values[parameter_offset + 1u]), static_cast<float>(values[parameter_offset + 2u])},
                .direction = vortex.axis,
                .width = vortex.width,
                .magnitude = static_cast<float>(values[parameter_offset + 3u]),
            });
            parameter_offset += 4u;
        }
        return result;
    }

    std::vector<std::uint8_t> ControlSystem::active_parameters(const std::uint32_t begin_step, const std::uint32_t end_step) const {
        std::vector<std::uint8_t> result(parameters.values.size(), 0u);
        std::size_t parameter_offset = 0u;
        for (const WindDefinition& wind : configuration.winds) {
            if (wind.begin_step >= begin_step && wind.begin_step < end_step) for (std::size_t parameter = 0u; parameter < 6u; ++parameter) result[parameter_offset + parameter] = 1u;
            parameter_offset += 6u;
        }
        for (const VortexDefinition& vortex : configuration.vortices) {
            if (vortex.begin_step >= begin_step && vortex.begin_step < end_step) for (std::size_t parameter = 0u; parameter < 4u; ++parameter) result[parameter_offset + parameter] = 1u;
            parameter_offset += 4u;
        }
        return result;
    }
} // namespace physica::fluids::gas::keyframe_smoke
