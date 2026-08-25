#ifndef PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_DOMAIN_KERNELS_H
#define PHYSICA_FLUIDS_GAS_KEYFRAME_SMOKE_DOMAIN_KERNELS_H

#include <physica/cuda_stream.h>
#include <cstddef>

namespace physica::fluids::gas::keyframe_smoke::cuda_detail {
    void accumulate(::cuda::stream_ref stream, const double* source, double* destination, std::size_t count);
}

#endif
