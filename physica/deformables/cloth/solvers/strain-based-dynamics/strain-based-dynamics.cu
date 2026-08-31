#include "strain-based-dynamics-kernels.h"
#include <cuda/launch>
#include <simulation/field/device.cuh>

namespace physica::deformables::cloth::solvers::strain_based_dynamics::kernels {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __device__ void project_constraint(const float constraint, const float stiffness, const float first_inverse_mass, const float second_inverse_mass, const float third_inverse_mass, const Vector3<float> first_gradient, const Vector3<float> second_gradient, const Vector3<float> third_gradient, Vector3<float>& first_position, Vector3<float>& second_position, Vector3<float>& third_position) {
            const float denominator = first_inverse_mass * dot(first_gradient, first_gradient) + second_inverse_mass * dot(second_gradient, second_gradient) + third_inverse_mass * dot(third_gradient, third_gradient);
            if (denominator == 0.0F) return;
            const float multiplier = -stiffness * constraint / denominator;
            first_position         = first_position + (first_inverse_mass * multiplier) * first_gradient;
            second_position        = second_position + (second_inverse_mass * multiplier) * second_gradient;
            third_position         = third_position + (third_inverse_mass * multiplier) * third_gradient;
        }

        __global__ void project_strain_kernel(const std::uint32_t triangle_count, const std::uint32_t color_offset, const std::uint32_t* colored_triangles, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const float* inverse_rest_00, const float* inverse_rest_01, const float* inverse_rest_10, const float* inverse_rest_11, const std::uint32_t* fixed_vertex_mask, const float* masses, const float stretch_stiffness_u, const float stretch_stiffness_v, const float shear_stiffness, const simulation::VectorView<float> positions) {
            const std::uint32_t local_triangle = blockIdx.x * blockDim.x + threadIdx.x;
            if (local_triangle >= triangle_count) return;
            const std::uint32_t triangle = colored_triangles[color_offset + local_triangle];
            const std::uint32_t first    = triangle_first[triangle];
            const std::uint32_t second   = triangle_second[triangle];
            const std::uint32_t third    = triangle_third[triangle];

            const float inverse_00          = inverse_rest_00[triangle];
            const float inverse_01          = inverse_rest_01[triangle];
            const float inverse_10          = inverse_rest_10[triangle];
            const float inverse_11          = inverse_rest_11[triangle];
            const float first_inverse_mass  = fixed_vertex_mask[first] == 0u ? 1.0F / masses[first] : 0.0F;
            const float second_inverse_mass = fixed_vertex_mask[second] == 0u ? 1.0F / masses[second] : 0.0F;
            const float third_inverse_mass  = fixed_vertex_mask[third] == 0u ? 1.0F / masses[third] : 0.0F;
            Vector3<float> first_position   = load(positions, first);
            Vector3<float> second_position  = load(positions, second);
            Vector3<float> third_position   = load(positions, third);

            const float u_first_coefficient  = -inverse_00 - inverse_10;
            const float u_second_coefficient = inverse_00;
            const float u_third_coefficient  = inverse_10;
            const float v_first_coefficient  = -inverse_01 - inverse_11;
            const float v_second_coefficient = inverse_01;
            const float v_third_coefficient  = inverse_11;

            Vector3<float> first_displacement  = second_position - first_position;
            Vector3<float> second_displacement = third_position - first_position;
            Vector3<float> deformation_u       = inverse_00 * first_displacement + inverse_10 * second_displacement;
            Vector3<float> first_gradient      = (2.0F * u_first_coefficient) * deformation_u;
            Vector3<float> second_gradient     = (2.0F * u_second_coefficient) * deformation_u;
            Vector3<float> third_gradient      = (2.0F * u_third_coefficient) * deformation_u;
            project_constraint(dot(deformation_u, deformation_u) - 1.0F, stretch_stiffness_u, first_inverse_mass, second_inverse_mass, third_inverse_mass, first_gradient, second_gradient, third_gradient, first_position, second_position, third_position);

            first_displacement  = second_position - first_position;
            second_displacement = third_position - first_position;
            Vector3<float> deformation_v = inverse_01 * first_displacement + inverse_11 * second_displacement;
            first_gradient              = (2.0F * v_first_coefficient) * deformation_v;
            second_gradient             = (2.0F * v_second_coefficient) * deformation_v;
            third_gradient              = (2.0F * v_third_coefficient) * deformation_v;
            project_constraint(dot(deformation_v, deformation_v) - 1.0F, stretch_stiffness_v, first_inverse_mass, second_inverse_mass, third_inverse_mass, first_gradient, second_gradient, third_gradient, first_position, second_position, third_position);

            first_displacement  = second_position - first_position;
            second_displacement = third_position - first_position;
            deformation_u = inverse_00 * first_displacement + inverse_10 * second_displacement;
            deformation_v = inverse_01 * first_displacement + inverse_11 * second_displacement;
            first_gradient  = u_first_coefficient * deformation_v + v_first_coefficient * deformation_u;
            second_gradient = u_second_coefficient * deformation_v + v_second_coefficient * deformation_u;
            third_gradient  = u_third_coefficient * deformation_v + v_third_coefficient * deformation_u;
            project_constraint(dot(deformation_u, deformation_v), shear_stiffness, first_inverse_mass, second_inverse_mass, third_inverse_mass, first_gradient, second_gradient, third_gradient, first_position, second_position, third_position);

            store(positions, first, first_position);
            store(positions, second, second_position);
            store(positions, third, third_position);
        }

    } // namespace

    void project_strain(const ::cuda::stream_ref stream, const std::uint32_t triangle_count, const std::uint32_t color_offset, const std::uint32_t* colored_triangles, const std::uint32_t* triangle_first, const std::uint32_t* triangle_second, const std::uint32_t* triangle_third, const float* inverse_rest_00, const float* inverse_rest_01, const float* inverse_rest_10, const float* inverse_rest_11, const std::uint32_t* fixed_vertex_mask, const float* masses, const float stretch_stiffness_u, const float stretch_stiffness_v, const float shear_stiffness, const simulation::VectorView<float> positions) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(triangle_count), project_strain_kernel, triangle_count, color_offset, colored_triangles, triangle_first, triangle_second, triangle_third, inverse_rest_00, inverse_rest_01, inverse_rest_10, inverse_rest_11, fixed_vertex_mask, masses, stretch_stiffness_u, stretch_stiffness_v, shear_stiffness, positions);
    }
} // namespace physica::deformables::cloth::solvers::strain_based_dynamics::kernels
