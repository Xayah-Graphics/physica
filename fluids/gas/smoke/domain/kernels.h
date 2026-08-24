#ifndef PHYSICA_FLUIDS_GAS_SMOKE_DOMAIN_KERNELS_H
#define PHYSICA_FLUIDS_GAS_SMOKE_DOMAIN_KERNELS_H

#include <cstdint>
#include <physica/cuda_stream.h>

namespace physica::fluids::gas::smoke::cuda_detail {
    void accumulate(::cuda::stream_ref stream, const double* source, double* destination, std::uint64_t count);
} // namespace physica::fluids::gas::smoke::cuda_detail

#endif
