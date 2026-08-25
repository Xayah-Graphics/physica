#ifndef PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_DOMAIN_KERNELS_H
#define PHYSICA_FLUIDS_GAS_ADJOINT_CONTROL_DOMAIN_KERNELS_H

#include <physica/cuda_stream.h>
#include <cstddef>

namespace physica::fluids::gas::adjoint_control::cuda_detail {
    void accumulate(::cuda::stream_ref stream, const double* source, double* destination, std::size_t count);
}

#endif
