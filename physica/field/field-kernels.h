#ifndef PHYSICA_FIELD_FIELD_KERNELS_H
#define PHYSICA_FIELD_FIELD_KERNELS_H

#include <physica/field/device.cuh>
#include <cstddef>
#include <physica/cuda_stream.h>

namespace physica::field::kernels {
    void accumulate(::cuda::stream_ref stream, const double* source, double* destination, std::size_t count);
    void accumulate(::cuda::stream_ref stream, VectorView<const double> source, VectorView<double> destination, std::size_t count);
}

#endif
