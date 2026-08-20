#ifndef PHYSICA_FLUIDS_GAS_SMOKE_DOMAIN_DEVICE_H
#define PHYSICA_FLUIDS_GAS_SMOKE_DOMAIN_DEVICE_H

#include <cstdint>

namespace physica::fluids::gas::smoke::cuda_detail {
    struct Vector final {
        float x;
        float y;
        float z;
    };

    struct Grid final {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        float cell_size;
        float time_step;
    };

    struct ScalarBoundaryData final {
        std::uint32_t modes[6];
        float values[6];
    };

    struct VelocityBoundaryData final {
        std::uint32_t modes[6];
        float values[18];
    };

    struct ConstScalarView;

    struct ScalarView final {
        float* values;

        operator ConstScalarView() const;
    };

    struct ConstScalarView final {
        const float* values;
    };

    inline ScalarView::operator ConstScalarView() const {
        return {values};
    }

    struct ConstCenteredVectorView;

    struct CenteredVectorView final {
        float* x;
        float* y;
        float* z;

        operator ConstCenteredVectorView() const;
    };

    struct ConstCenteredVectorView final {
        const float* x;
        const float* y;
        const float* z;
    };

    inline CenteredVectorView::operator ConstCenteredVectorView() const {
        return {x, y, z};
    }

    struct ConstStaggeredVectorView;

    struct StaggeredVectorView final {
        float* x;
        float* y;
        float* z;

        operator ConstStaggeredVectorView() const;
    };

    struct ConstStaggeredVectorView final {
        const float* x;
        const float* y;
        const float* z;
    };

    inline StaggeredVectorView::operator ConstStaggeredVectorView() const {
        return {x, y, z};
    }

    struct ConstScalarAdjointView;

    struct ScalarAdjointView final {
        double* values;

        operator ConstScalarAdjointView() const;
    };

    struct ConstScalarAdjointView final {
        const double* values;
    };

    inline ScalarAdjointView::operator ConstScalarAdjointView() const {
        return {values};
    }

    struct ConstCenteredVectorAdjointView;

    struct CenteredVectorAdjointView final {
        double* x;
        double* y;
        double* z;

        operator ConstCenteredVectorAdjointView() const;
    };

    struct ConstCenteredVectorAdjointView final {
        const double* x;
        const double* y;
        const double* z;
    };

    inline CenteredVectorAdjointView::operator ConstCenteredVectorAdjointView() const {
        return {x, y, z};
    }

    struct ConstStaggeredVectorAdjointView;

    struct StaggeredVectorAdjointView final {
        double* x;
        double* y;
        double* z;

        operator ConstStaggeredVectorAdjointView() const;
    };

    struct ConstStaggeredVectorAdjointView final {
        const double* x;
        const double* y;
        const double* z;
    };

    inline StaggeredVectorAdjointView::operator ConstStaggeredVectorAdjointView() const {
        return {x, y, z};
    }
} // namespace physica::fluids::gas::smoke::cuda_detail

#endif
