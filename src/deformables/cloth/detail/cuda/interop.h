#ifndef PHYSICA_DEFORMABLES_CLOTH_DETAIL_CUDA_INTEROP_H
#define PHYSICA_DEFORMABLES_CLOTH_DETAIL_CUDA_INTEROP_H

#include "types.h"

namespace physica::deformables::cloth::cuda_detail {
    template <class Scalar, class Field>
    ConstFieldView<Scalar> field(const Field& value) {
        return {.x = value.x.data(), .y = value.y.data(), .z = value.z.data()};
    }

    template <class Scalar, class Field>
    FieldView<Scalar> field(Field& value) {
        return {.x = value.x.data(), .y = value.y.data(), .z = value.z.data()};
    }
} // namespace physica::deformables::cloth::cuda_detail

#endif
