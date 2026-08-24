#ifndef PHYSICA_FLUIDS_LIQUID_PARTICLE_DOMAIN_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_PARTICLE_DOMAIN_KERNELS_H

#include "device.h"
#include <physica/cuda_stream.h>
#include <cstddef>

namespace physica::fluids::liquid::particle::cuda_detail {
    void accumulate(::cuda::stream_ref stream, const double* source, double* destination, std::size_t count);
    void accumulate(::cuda::stream_ref stream, ConstVectorView<double> source, VectorView<double> destination, std::size_t count);
} // namespace physica::fluids::liquid::particle::cuda_detail

#endif
