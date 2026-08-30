#ifndef PHYSICA_EXAMPLES_FLUIDS_GAS_SMOKE_SIMULATION_KERNELS_H
#define PHYSICA_EXAMPLES_FLUIDS_GAS_SMOKE_SIMULATION_KERNELS_H

#include <cstdint>
#include <math/math.h>
#include <physica/cuda_stream.h>

namespace physica::examples::smoke::simulation_cuda {
    struct Grid final {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        float cell_size;
        float time_step;
    };

    void write_control(::cuda::stream_ref stream, Grid grid, std::uint64_t step, float pulse_period, Vector3<float> left_center, Vector3<float> right_center, float source_radius, float density_source_rate, float temperature_source_rate, Vector3<float> left_acceleration, Vector3<float> right_acceleration, float* density_source, float* temperature_source, float* acceleration_x, float* acceleration_y, float* acceleration_z);
} // namespace physica::examples::smoke::simulation_cuda

#endif
