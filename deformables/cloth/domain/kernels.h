#ifndef PHYSICA_DEFORMABLES_CLOTH_DOMAIN_KERNELS_H
#define PHYSICA_DEFORMABLES_CLOTH_DOMAIN_KERNELS_H

#include "device.h"
#include <cstddef>
#include <physica/cuda_stream.h>

namespace physica::deformables::cloth::cuda_detail {
    void accumulate(::cuda::stream_ref stream, ConstFieldView<double> source, FieldView<double> destination, std::size_t count);
} // namespace physica::deformables::cloth::cuda_detail

#endif
