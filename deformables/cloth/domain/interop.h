#ifndef PHYSICA_DEFORMABLES_CLOTH_DOMAIN_INTEROP_H
#define PHYSICA_DEFORMABLES_CLOTH_DOMAIN_INTEROP_H

#include "device.h"

namespace physica::deformables::cloth::cuda_detail {
    template<class Field>
    ConstFieldView<float> field(const Field& value) {
        return {.x = value.x.values.data(), .y = value.y.values.data(), .z = value.z.values.data()};
    }

    template<class Field>
    FieldView<float> field(Field& value) {
        return {.x = value.x.values.data(), .y = value.y.values.data(), .z = value.z.values.data()};
    }

    template<class Field>
    ConstFieldView<double> adjoint_field(const Field& value) {
        return {.x = value.x.values.data(), .y = value.y.values.data(), .z = value.z.values.data()};
    }

    template<class Field>
    FieldView<double> adjoint_field(Field& value) {
        return {.x = value.x.values.data(), .y = value.y.values.data(), .z = value.z.values.data()};
    }
} // namespace physica::deformables::cloth::cuda_detail

#endif
