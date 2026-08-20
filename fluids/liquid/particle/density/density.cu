#include "device.cuh"
#include "kernels.h"
#include "../neighborhood/device.cuh"
#include <cuda/launch>

namespace physica::fluids::liquid::particle::cuda_detail {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __device__ float density_kernel(const Float3 displacement, const float support_radius, const bool pbf_kernel) {
            return pbf_kernel ? poly6(displacement, support_radius) : cubic(displacement, support_radius);
        }

        __device__ Float3 density_kernel_gradient(const Float3 displacement, const float support_radius, const bool pbf_kernel) {
            return pbf_kernel ? poly6_gradient(displacement, support_radius) : cubic_gradient(displacement, support_radius);
        }

        __global__ void density_forward_kernel(const std::uint32_t particle_count, const float support_radius, const bool pbf_kernel, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, float* densities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            float result = 0.0F;
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            result += parameters.masses[neighbor] * density_kernel(subtract(position, load(positions, neighbor)), support_radius, pbf_kernel);
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            result += parameters.rest_densities[particle] * boundary.volumes[neighbor] * density_kernel(subtract(position, boundary_position(boundary, neighbor)), support_radius, pbf_kernel);
                        }
                    }
            densities[particle] = result;
        }

        __global__ void density_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const bool pbf_kernel, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView parameters, const ParticleParameterTangentView parameter_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, float* density_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 tangent = load(position_tangent, particle);
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            float result = 0.0F;
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            result += parameter_tangent.masses[neighbor] * density_kernel(displacement, support_radius, pbf_kernel) + parameters.masses[neighbor] * dot(density_kernel_gradient(displacement, support_radius, pbf_kernel), subtract(tangent, load(position_tangent, neighbor)));
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, boundary_position(boundary, neighbor));
                            result += boundary.volumes[neighbor] * (parameter_tangent.rest_densities[particle] * density_kernel(displacement, support_radius, pbf_kernel) + parameters.rest_densities[particle] * dot(density_kernel_gradient(displacement, support_radius, pbf_kernel), tangent));
                        }
                    }
            density_tangent[particle] = result;
        }

        __global__ void density_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const bool pbf_kernel, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const double* density_adjoint, const VectorView<double> position_adjoint, const ParticleParameterAdjointView parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const double output_adjoint = density_adjoint[particle];
            Double3 position_contribution{};
            double mass_contribution = 0.0;
            double rest_density_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        for (std::uint32_t sorted = range.first; sorted < range.last; ++sorted) {
                            const std::uint32_t other = neighborhood.sorted_particle_indices[sorted];
                            const Float3 displacement = subtract(position, load(positions, other));
                            const Float3 gradient = density_kernel_gradient(displacement, support_radius, pbf_kernel);
                            position_contribution = add(position_contribution, scale(gradient, output_adjoint * parameters.masses[other] + density_adjoint[other] * parameters.masses[particle]));
                            mass_contribution += density_adjoint[other] * density_kernel(displacement, support_radius, pbf_kernel);
                        }
                        for (std::uint32_t sorted = range.boundary_first; sorted < range.boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, boundary_position(boundary, neighbor));
                            const float volume = boundary.volumes[neighbor];
                            position_contribution = add(position_contribution, scale(density_kernel_gradient(displacement, support_radius, pbf_kernel), output_adjoint * parameters.rest_densities[particle] * volume));
                            rest_density_contribution += output_adjoint * volume * density_kernel(displacement, support_radius, pbf_kernel);
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.rest_densities[particle] += rest_density_contribution;
        }
    } // namespace

    void density_forward(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const bool pbf_kernel, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, float* densities) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_forward_kernel, particle_count, support_radius, pbf_kernel, topology_positions, positions, parameters, neighborhood, boundary, densities);
    }

    void density_jvp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const bool pbf_kernel, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ConstVectorView<float> position_tangent, const ParticleParameterView parameters, const ParticleParameterTangentView parameter_tangent, const NeighborhoodView neighborhood, const BoundaryView boundary, float* density_tangent) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_jvp_kernel, particle_count, support_radius, pbf_kernel, topology_positions, positions, position_tangent, parameters, parameter_tangent, neighborhood, boundary, density_tangent);
    }

    void density_vjp(const ::cuda::stream_ref stream, const std::uint32_t particle_count, const float support_radius, const bool pbf_kernel, const ConstVectorView<float> topology_positions, const ConstVectorView<float> positions, const ParticleParameterView parameters, const NeighborhoodView neighborhood, const BoundaryView boundary, const double* density_adjoint, const VectorView<double> position_adjoint, const ParticleParameterAdjointView parameter_adjoint) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(particle_count), density_vjp_kernel, particle_count, support_radius, pbf_kernel, topology_positions, positions, parameters, neighborhood, boundary, density_adjoint, position_adjoint, parameter_adjoint);
    }
} // namespace physica::fluids::liquid::particle::cuda_detail
