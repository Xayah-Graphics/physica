#include "density-kernels.h"
#include <cuda/launch>
#include <fluids/liquid/device.cuh>

namespace physica::fluids::liquid::operators::kernels::density {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        template <bool Poly6>
        __device__ float density_kernel(const Vector3<float> displacement, const float support_radius) {
            if constexpr (Poly6) return device::poly6(displacement, support_radius);
            else return device::cubic(displacement, support_radius);
        }

        template <bool Poly6>
        __device__ Vector3<float> density_kernel_gradient(const Vector3<float> displacement, const float support_radius) {
            if constexpr (Poly6) return device::poly6_gradient(displacement, support_radius);
            else return device::cubic_gradient(displacement, support_radius);
        }

        template <bool Poly6>
        __global__ void density_forward_kernel(const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, float* densities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = load(positions, particle);
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            float result = 0.0F;
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            result += parameters.masses[neighbor] * density_kernel<Poly6>((position - load(positions, neighbor)), support_radius);
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            result += parameters.rest_densities[particle] * boundary.volumes[neighbor] * density_kernel<Poly6>((position - device::boundary_position(boundary, neighbor)), support_radius);
                        }
                    }
            densities[particle] = result;
        }

        template <bool Poly6>
        __global__ void density_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> position_tangent, const device::ParticleParameterView parameters, const device::ParticleParameterTangentView parameter_tangent, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, float* density_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = load(positions, particle);
            const Vector3<float> tangent  = load(position_tangent, particle);
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            float result = 0.0F;
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor      = neighborhood.sorted_particle_indices[sorted];
                            const Vector3<float> displacement = (position - load(positions, neighbor));
                            result += parameter_tangent.masses[neighbor] * density_kernel<Poly6>(displacement, support_radius) + parameters.masses[neighbor] * dot(density_kernel_gradient<Poly6>(displacement, support_radius), (tangent - load(position_tangent, neighbor)));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor      = neighborhood.sorted_boundary_indices[sorted];
                            const Vector3<float> displacement = (position - device::boundary_position(boundary, neighbor));
                            result += boundary.volumes[neighbor] * (parameter_tangent.rest_densities[particle] * density_kernel<Poly6>(displacement, support_radius) + parameters.rest_densities[particle] * dot(density_kernel_gradient<Poly6>(displacement, support_radius), tangent));
                        }
                    }
            density_tangent[particle] = result;
        }

        template <bool Poly6>
        __global__ void density_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const double* density_adjoint, const simulation::VectorView<double> position_adjoint, const device::ParticleParameterAdjointView parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector3<float> position = load(positions, particle);
            const double output_adjoint   = density_adjoint[particle];
            Vector3<double> position_contribution{};
            double mass_contribution         = 0.0;
            double rest_density_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            device::particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const device::CellRange range = device::cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t other         = neighborhood.sorted_particle_indices[sorted];
                            const Vector3<float> displacement = (position - load(positions, other));
                            const Vector3<float> gradient     = density_kernel_gradient<Poly6>(displacement, support_radius);
                            position_contribution             = position_contribution + gradient * (output_adjoint * parameters.masses[other] + density_adjoint[other] * parameters.masses[particle]);
                            mass_contribution += density_adjoint[other] * density_kernel<Poly6>(displacement, support_radius);
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor      = neighborhood.sorted_boundary_indices[sorted];
                            const Vector3<float> displacement = (position - device::boundary_position(boundary, neighbor));
                            const float volume                = boundary.volumes[neighbor];
                            position_contribution             = (position_contribution + (density_kernel_gradient<Poly6>(displacement, support_radius) * output_adjoint * parameters.rest_densities[particle] * volume));
                            rest_density_contribution += output_adjoint * volume * density_kernel<Poly6>(displacement, support_radius);
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.rest_densities[particle] += rest_density_contribution;
        }
    } // namespace

    void cubic_density_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, float* densities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_forward_kernel<false>, particle_count, support_radius, topology_positions, positions, parameters, neighborhood, boundary, densities);
    }

    void cubic_density_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> position_tangent, const device::ParticleParameterView parameters, const device::ParticleParameterTangentView parameter_tangent, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, float* density_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_jvp_kernel<false>, particle_count, support_radius, topology_positions, positions, position_tangent, parameters, parameter_tangent, neighborhood, boundary, density_tangent);
    }

    void cubic_density_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const double* density_adjoint, const simulation::VectorView<double> position_adjoint, const device::ParticleParameterAdjointView parameter_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_vjp_kernel<false>, particle_count, support_radius, topology_positions, positions, parameters, neighborhood, boundary, density_adjoint, position_adjoint, parameter_adjoint);
    }

    void poly6_density_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, float* densities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_forward_kernel<true>, particle_count, support_radius, topology_positions, positions, parameters, neighborhood, boundary, densities);
    }

    void poly6_density_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const simulation::VectorView<const float> position_tangent, const device::ParticleParameterView parameters, const device::ParticleParameterTangentView parameter_tangent, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, float* density_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_jvp_kernel<true>, particle_count, support_radius, topology_positions, positions, position_tangent, parameters, parameter_tangent, neighborhood, boundary, density_tangent);
    }

    void poly6_density_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const simulation::VectorView<const float> topology_positions, const simulation::VectorView<const float> positions, const device::ParticleParameterView parameters, const device::NeighborhoodView neighborhood, const device::BoundaryView boundary, const double* density_adjoint, const simulation::VectorView<double> position_adjoint, const device::ParticleParameterAdjointView parameter_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_vjp_kernel<true>, particle_count, support_radius, topology_positions, positions, parameters, neighborhood, boundary, density_adjoint, position_adjoint, parameter_adjoint);
    }
} // namespace physica::fluids::liquid::operators::kernels::density
