module;

#include <physica/cuda.h>

export module physica.fluids.grid.poisson;

import std;
export import physica.fluids.grid;

export namespace physica::fluids::grid {
    struct PoissonSystem final {
        ScalarField<float> diagonal;
        ScalarField<float> rhs;
        ScalarField<float> pressure;
    };
} // namespace physica::fluids::grid
