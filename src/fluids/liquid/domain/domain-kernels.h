#ifndef PHYSICA_FLUIDS_LIQUID_DOMAIN_KERNELS_H
#define PHYSICA_FLUIDS_LIQUID_DOMAIN_KERNELS_H

#include "../detail/cuda/types.h"
#include <cstddef>
#include <physica/cuda_stream.h>

namespace physica::fluids::liquid::cuda_detail {
    void accumulate(::cuda::stream_ref stream, const double* source, double* destination, std::size_t count);
    void accumulate(::cuda::stream_ref stream, ConstVectorView<double> source, VectorView<double> destination, std::size_t count);
} // namespace physica::fluids::liquid::cuda_detail

#endif
