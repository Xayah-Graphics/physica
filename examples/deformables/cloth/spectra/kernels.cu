#include "kernels.h"
#include <cuda/launch>
#include <math.h>

namespace physica::examples::cloth::spectra_cuda {
    namespace {
        constexpr std::uint32_t block_size = 256u;

        __global__ void write_surface_kernel(const std::uint32_t rows, const std::uint32_t columns, const float* position_x, const float* position_y, const float* position_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* normals) {
            const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
            if (vertex >= rows * columns) return;
            const std::uint32_t row = vertex / columns;
            const std::uint32_t column = vertex % columns;
            positions[vertex] = {.x = position_x[vertex], .y = position_y[vertex], .z = position_z[vertex]};

            const std::uint32_t left = row * columns + (column == 0u ? 0u : column - 1u);
            const std::uint32_t right = row * columns + (column + 1u < columns ? column + 1u : columns - 1u);
            const std::uint32_t top = (row == 0u ? 0u : row - 1u) * columns + column;
            const std::uint32_t bottom = (row + 1u < rows ? row + 1u : rows - 1u) * columns + column;
            const float tangent_x_x = position_x[right] - position_x[left];
            const float tangent_x_y = position_y[right] - position_y[left];
            const float tangent_x_z = position_z[right] - position_z[left];
            const float tangent_y_x = position_x[bottom] - position_x[top];
            const float tangent_y_y = position_y[bottom] - position_y[top];
            const float tangent_y_z = position_z[bottom] - position_z[top];
            float normal_x = tangent_x_y * tangent_y_z - tangent_x_z * tangent_y_y;
            float normal_y = tangent_x_z * tangent_y_x - tangent_x_x * tangent_y_z;
            float normal_z = tangent_x_x * tangent_y_y - tangent_x_y * tangent_y_x;
            const float inverse_length = rsqrtf(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
            normal_x *= inverse_length;
            normal_y *= inverse_length;
            normal_z *= inverse_length;
            normals[vertex] = {.x = normal_x, .y = normal_y, .z = normal_z};
        }
    } // namespace

    void write_surface(const ::cuda::stream_ref stream, const std::uint32_t rows, const std::uint32_t columns, const float* position_x, const float* position_y, const float* position_z, spectra::sdk::Float3* positions, spectra::sdk::Float3* normals) {
        ::cuda::launch(stream, ::cuda::distribute<block_size>(rows * columns), write_surface_kernel, rows, columns, position_x, position_y, position_z, positions, normals);
    }
} // namespace physica::examples::cloth::spectra_cuda
