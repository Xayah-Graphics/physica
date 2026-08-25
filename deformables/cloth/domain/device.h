#ifndef PHYSICA_DEFORMABLES_CLOTH_DOMAIN_DEVICE_H
#define PHYSICA_DEFORMABLES_CLOTH_DOMAIN_DEVICE_H

namespace physica::deformables::cloth::cuda_detail {
    template <class Scalar>
    struct Vector final {
        Scalar x;
        Scalar y;
        Scalar z;
    };

    template <class Scalar>
    struct ConstFieldView;

    template <class Scalar>
    struct FieldView final {
        Scalar* x;
        Scalar* y;
        Scalar* z;

        operator ConstFieldView<Scalar>() const {
            return {x, y, z};
        }
    };

    template <class Scalar>
    struct ConstFieldView final {
        const Scalar* x;
        const Scalar* y;
        const Scalar* z;
    };
} // namespace physica::deformables::cloth::cuda_detail

#endif
