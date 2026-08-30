#include "dynamics-kernels.h"
#include <cstdint>
#include <cuda/cmath>
#include <cuda/std/cmath>
#include <cuda_runtime.h>

namespace physica::fluids::liquid::solvers::sph::kernels::pcisph {

    namespace {

        constexpr std::uint32_t block_size = 256u;


        __global__ void pressure_update_forward_kernel(const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const device::ParticleParameterView particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, float* pressures) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float volume  = particles.masses[particle] / particles.rest_densities[particle];
            const float delta   = 1.0F / (2.0F * time_step * time_step * volume * volume * reference_gradient_norm);
            pressures[particle] = ::cuda::std::max(0.0F, previous_pressures[particle] + pressure_relaxation[particle] * delta * (predicted_densities[particle] - particles.rest_densities[particle]));
        }

        __global__ void pressure_update_jvp_kernel(const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const device::ParticleParameterView particles, const device::ParticleParameterTangentView particle_tangent, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const float* previous_pressure_tangent, const float* predicted_density_tangent, const float* pressure_relaxation_tangent, float* pressure_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float error          = predicted_densities[particle] - particles.rest_densities[particle];
            const float volume         = particles.masses[particle] / particles.rest_densities[particle];
            const float delta          = 1.0F / (2.0F * time_step * time_step * volume * volume * reference_gradient_norm);
            const float delta_dot      = -2.0F * delta * (particle_tangent.masses[particle] / particles.masses[particle] - particle_tangent.rest_densities[particle] / particles.rest_densities[particle]);
            const float candidate      = previous_pressures[particle] + pressure_relaxation[particle] * delta * error;
            pressure_tangent[particle] = candidate > 0.0F ? previous_pressure_tangent[particle] + pressure_relaxation_tangent[particle] * delta * error + pressure_relaxation[particle] * delta_dot * error + pressure_relaxation[particle] * delta * (predicted_density_tangent[particle] - particle_tangent.rest_densities[particle]) : 0.0F;
        }

        __global__ void pressure_update_vjp_kernel(const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const device::ParticleParameterView particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const double* pressure_adjoint, const device::ParticleParameterAdjointView particle_adjoint, double* previous_pressure_adjoint, double* predicted_density_adjoint, double* pressure_relaxation_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const double error     = static_cast<double>(predicted_densities[particle]) - particles.rest_densities[particle];
            const double volume    = static_cast<double>(particles.masses[particle]) / particles.rest_densities[particle];
            const double delta     = 1.0 / (2.0 * time_step * time_step * volume * volume * reference_gradient_norm);
            const double candidate = previous_pressures[particle] + pressure_relaxation[particle] * delta * error;
            if (candidate > 0.0) {
                const double adjoint = pressure_adjoint[particle];
                previous_pressure_adjoint[particle] += adjoint;
                predicted_density_adjoint[particle] += pressure_relaxation[particle] * delta * adjoint;
                particle_adjoint.rest_densities[particle] += pressure_relaxation[particle] * delta * adjoint * (-1.0 + 2.0 * error / particles.rest_densities[particle]);
                particle_adjoint.masses[particle] -= 2.0 * pressure_relaxation[particle] * delta * error * adjoint / particles.masses[particle];
                pressure_relaxation_adjoint[particle] += delta * error * adjoint;
            }
        }

    } // namespace

    void launch_pressure_update_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const device::ParticleParameterView particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, float* pressures) {
        pressure_update_forward_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, reference_gradient_norm, particles, previous_pressures, predicted_densities, pressure_relaxation, pressures);
    }

    void launch_pressure_update_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const device::ParticleParameterView particles, const device::ParticleParameterTangentView particle_tangent, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const float* previous_pressure_tangent, const float* predicted_density_tangent, const float* pressure_relaxation_tangent, float* pressure_tangent) {
        pressure_update_jvp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, reference_gradient_norm, particles, particle_tangent, previous_pressures, predicted_densities, pressure_relaxation, previous_pressure_tangent, predicted_density_tangent, pressure_relaxation_tangent, pressure_tangent);
    }

    void launch_pressure_update_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const device::ParticleParameterView particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const double* pressure_adjoint, const device::ParticleParameterAdjointView particle_adjoint, double* previous_pressure_adjoint, double* predicted_density_adjoint, double* pressure_relaxation_adjoint) {
        pressure_update_vjp_kernel<<<::cuda::ceil_div(particle_count, block_size), block_size, 0, stream.get()>>>(particle_count, time_step, reference_gradient_norm, particles, previous_pressures, predicted_densities, pressure_relaxation, pressure_adjoint, particle_adjoint, previous_pressure_adjoint, predicted_density_adjoint, pressure_relaxation_adjoint);
    }

} // namespace physica::fluids::liquid::solvers::sph::kernels::pcisph
