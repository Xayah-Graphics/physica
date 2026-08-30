module;

#include "objective-kernels.h"
#include <fluids/gas/interop.h>
#include <physica/cuda.h>

module physica.fluids.gas.operators.objective;

import std;

namespace physica::fluids::gas::operators {
    namespace {
        ::cuda::device_buffer<double> allocate_scalar(const Domain& domain) {
            return ::cuda::device_buffer<double>{domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), 1u, ::cuda::no_init};
        }

        std::vector<float> gaussian_weights(const float sigma, const std::uint32_t radius) {
            std::vector<float> result(radius * 2u + 1u);
            if (radius == 0u) {
                result[0] = 1.0F;
                return result;
            }
            float sum{};
            for (int offset = -static_cast<int>(radius); offset <= static_cast<int>(radius); ++offset) {
                const float value       = std::exp(-0.5F * offset * offset / (sigma * sigma));
                result[offset + radius] = value;
                sum += value;
            }
            for (float& value : result) value /= sum;
            return result;
        }
    } // namespace

    Quadratic::Quadratic(const Domain& domain, Configuration next_configuration) : configuration(std::move(next_configuration)), blur_radius(static_cast<std::uint32_t>(std::ceil(3.0F * configuration.blur_sigma_cells))), blur_weights(domain.grid.stream, ::cuda::device_default_memory_pool(domain.grid.stream.device()), blur_radius * 2u + 1u, ::cuda::no_init) {
        const std::vector<float> weights = gaussian_weights(configuration.blur_sigma_cells, blur_radius);
        ::cuda::copy_bytes(domain.grid.stream, weights, blur_weights);
    }

    KeyframeCache Quadratic::allocate_keyframe_cache(const Domain& domain) const {
        return {
            .density_residual          = domain.grid.allocate_cell_field<float>(),
            .blurred_density_residual  = domain.grid.allocate_cell_field<float>(),
            .velocity_residual         = domain.grid.allocate_mac_field<float>(),
            .blurred_velocity_residual = domain.grid.allocate_mac_field<float>(),
            .density_loss              = allocate_scalar(domain),
            .velocity_loss             = allocate_scalar(domain),
            .directional_derivative    = allocate_scalar(domain),
        };
    }

    StepCache Quadratic::allocate_step_cache(const Domain& domain) const {
        return {.control_effort = allocate_scalar(domain), .directional_derivative = allocate_scalar(domain)};
    }

    Workspace Quadratic::allocate_workspace(const Domain& domain) const {
        return {
            .density_blur_first   = domain.grid.allocate_cell_field<float>(),
            .density_blur_second  = domain.grid.allocate_cell_field<float>(),
            .density_blur_output  = domain.grid.allocate_cell_field<float>(),
            .velocity_blur_first  = domain.grid.allocate_mac_field<float>(),
            .velocity_blur_second = domain.grid.allocate_mac_field<float>(),
            .velocity_blur_output = domain.grid.allocate_mac_field<float>(),
        };
    }

    Quadratic Quadratic::with_blur_sigma(const Domain& domain, const float sigma_cells) const {
        return Quadratic{domain, {.control_effort_weight = configuration.control_effort_weight, .blur_sigma_cells = sigma_cells}};
    }

    void Quadratic::evaluate_keyframe(const Domain& domain, const simulation::ScalarField<float>& density, const simulation::VectorField<float>& velocity, const simulation::ScalarField<float>& target_density, const simulation::VectorField<float>& target_velocity, const double density_weight, const double velocity_weight, KeyframeCache& cache, Workspace& workspace) const {
        const device::Discretization grid = device::discretization(domain.configuration);
        kernels::residual_forward(domain.grid.stream, grid, simulation::scalar_view(density), simulation::scalar_view(target_density), simulation::scalar_view(cache.density_residual));
        kernels::residual_forward(domain.grid.stream, grid, simulation::view(velocity), simulation::view(target_velocity), simulation::view(cache.velocity_residual));
        kernels::blur_forward(domain.grid.stream, grid, blur_radius, blur_weights.data(), simulation::scalar_view(cache.density_residual), simulation::scalar_view(workspace.density_blur_first), simulation::scalar_view(workspace.density_blur_second), simulation::scalar_view(cache.blurred_density_residual));
        kernels::blur_forward(domain.grid.stream, grid, blur_radius, blur_weights.data(), simulation::view(cache.velocity_residual), simulation::view(workspace.velocity_blur_first), simulation::view(workspace.velocity_blur_second), simulation::view(cache.blurred_velocity_residual));
        kernels::squared_loss(domain.grid.stream, grid, density_weight, simulation::scalar_view(cache.blurred_density_residual), cache.density_loss.data());
        kernels::squared_loss(domain.grid.stream, grid, velocity_weight, simulation::view(cache.blurred_velocity_residual), cache.velocity_loss.data());
    }

    void Quadratic::keyframe_jvp(const Domain& domain, const simulation::ScalarField<float>& density_tangent, const simulation::VectorField<float>& velocity_tangent, const double density_weight, const double velocity_weight, KeyframeCache& cache, Workspace& workspace) const {
        kernels::directional_loss(domain.grid.stream, device::discretization(domain.configuration), blur_radius, blur_weights.data(), density_weight, velocity_weight, simulation::scalar_view(cache.blurred_density_residual), simulation::view(cache.blurred_velocity_residual), simulation::scalar_view(density_tangent), simulation::view(velocity_tangent), simulation::scalar_view(workspace.density_blur_first), simulation::scalar_view(workspace.density_blur_second), simulation::scalar_view(workspace.density_blur_output), simulation::view(workspace.velocity_blur_first), simulation::view(workspace.velocity_blur_second), simulation::view(workspace.velocity_blur_output), cache.directional_derivative.data());
    }

    void Quadratic::keyframe_vjp(const Domain& domain, const double density_weight, const double velocity_weight, const KeyframeCache& cache, simulation::ScalarField<double>& density_adjoint, simulation::VectorField<double>& velocity_adjoint, Workspace& workspace) const {
        kernels::inject_keyframe_adjoint(domain.grid.stream, device::discretization(domain.configuration), blur_radius, blur_weights.data(), density_weight, velocity_weight, simulation::scalar_view(cache.blurred_density_residual), simulation::view(cache.blurred_velocity_residual), simulation::scalar_view(workspace.density_blur_first), simulation::scalar_view(workspace.density_blur_second), simulation::view(workspace.velocity_blur_first), simulation::view(workspace.velocity_blur_second), simulation::scalar_view(density_adjoint), simulation::view(velocity_adjoint));
    }

    void Quadratic::evaluate_control_effort(const Domain& domain, const simulation::VectorField<float>& control, StepCache& cache) const {
        kernels::control_effort(domain.grid.stream, device::discretization(domain.configuration), configuration.control_effort_weight, simulation::view(control), cache.control_effort.data());
    }

    void Quadratic::accumulate(const Domain& domain, const ::cuda::device_buffer<double>& contribution, ::cuda::device_buffer<double>& objective) const {
        kernels::accumulate(domain.grid.stream, contribution.data(), objective.data());
    }

    void Quadratic::control_effort_jvp(const Domain& domain, const simulation::VectorField<float>& control, const simulation::VectorField<float>& control_tangent, StepCache& cache) const {
        kernels::control_effort_jvp(domain.grid.stream, device::discretization(domain.configuration), configuration.control_effort_weight, simulation::view(control), simulation::view(control_tangent), cache.directional_derivative.data());
    }

    void Quadratic::control_effort_vjp(const Domain& domain, const simulation::VectorField<float>& control, simulation::VectorField<double>& control_adjoint) const {
        kernels::control_effort_vjp(domain.grid.stream, device::discretization(domain.configuration), configuration.control_effort_weight, simulation::view(control), simulation::view(control_adjoint));
    }
} // namespace physica::fluids::gas::operators
