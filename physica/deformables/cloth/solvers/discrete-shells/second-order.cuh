#ifndef PHYSICA_DEFORMABLES_CLOTH_SOLVERS_DISCRETE_SHELLS_SECOND_ORDER_CUH
#define PHYSICA_DEFORMABLES_CLOTH_SOLVERS_DISCRETE_SHELLS_SECOND_ORDER_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace physica::deformables::cloth::solvers::discrete_shells {
    template <std::uint32_t Dimension>
    struct SecondOrder final {
        static constexpr std::uint32_t hessian_size = Dimension * (Dimension + 1u) / 2u;

        float value{};
        float gradient[Dimension]{};
        float upper_hessian[hessian_size]{};

        [[nodiscard]] __device__ static constexpr std::uint32_t hessian_index(const std::uint32_t first, const std::uint32_t second) {
            const std::uint32_t row    = first < second ? first : second;
            const std::uint32_t column = first < second ? second : first;
            return row * Dimension - row * (row + 1u) / 2u + column;
        }

        [[nodiscard]] __device__ static SecondOrder constant(const float value) {
            SecondOrder result{};
            result.value = value;
            return result;
        }

        [[nodiscard]] __device__ static SecondOrder variable(const float value, const std::uint32_t index) {
            SecondOrder result{};
            result.value           = value;
            result.gradient[index] = 1.0F;
            return result;
        }

        [[nodiscard]] __device__ float hessian(const std::uint32_t first, const std::uint32_t second) const {
            return upper_hessian[hessian_index(first, second)];
        }
    };

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> operator+(const SecondOrder<Dimension>& first, const SecondOrder<Dimension>& second) {
        SecondOrder<Dimension> result{};
        result.value = first.value + second.value;
        for (std::uint32_t entry = 0u; entry < Dimension; ++entry) result.gradient[entry] = first.gradient[entry] + second.gradient[entry];
        for (std::uint32_t entry = 0u; entry < SecondOrder<Dimension>::hessian_size; ++entry) result.upper_hessian[entry] = first.upper_hessian[entry] + second.upper_hessian[entry];
        return result;
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> operator-(const SecondOrder<Dimension>& first, const SecondOrder<Dimension>& second) {
        SecondOrder<Dimension> result{};
        result.value = first.value - second.value;
        for (std::uint32_t entry = 0u; entry < Dimension; ++entry) result.gradient[entry] = first.gradient[entry] - second.gradient[entry];
        for (std::uint32_t entry = 0u; entry < SecondOrder<Dimension>::hessian_size; ++entry) result.upper_hessian[entry] = first.upper_hessian[entry] - second.upper_hessian[entry];
        return result;
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> operator-(const SecondOrder<Dimension>& operand) {
        SecondOrder<Dimension> result{};
        result.value = -operand.value;
        for (std::uint32_t entry = 0u; entry < Dimension; ++entry) result.gradient[entry] = -operand.gradient[entry];
        for (std::uint32_t entry = 0u; entry < SecondOrder<Dimension>::hessian_size; ++entry) result.upper_hessian[entry] = -operand.upper_hessian[entry];
        return result;
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> operator*(const SecondOrder<Dimension>& first, const SecondOrder<Dimension>& second) {
        SecondOrder<Dimension> result{};
        result.value = first.value * second.value;
        for (std::uint32_t entry = 0u; entry < Dimension; ++entry) result.gradient[entry] = first.gradient[entry] * second.value + first.value * second.gradient[entry];
        for (std::uint32_t row = 0u; row < Dimension; ++row) {
            for (std::uint32_t column = row; column < Dimension; ++column) {
                const std::uint32_t entry = SecondOrder<Dimension>::hessian_index(row, column);
                result.upper_hessian[entry] = first.upper_hessian[entry] * second.value + first.value * second.upper_hessian[entry] + first.gradient[row] * second.gradient[column] + second.gradient[row] * first.gradient[column];
            }
        }
        return result;
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> compose(const SecondOrder<Dimension>& operand, const float value, const float first_derivative, const float second_derivative) {
        SecondOrder<Dimension> result{};
        result.value = value;
        for (std::uint32_t entry = 0u; entry < Dimension; ++entry) result.gradient[entry] = first_derivative * operand.gradient[entry];
        for (std::uint32_t row = 0u; row < Dimension; ++row) {
            for (std::uint32_t column = row; column < Dimension; ++column) {
                const std::uint32_t entry = SecondOrder<Dimension>::hessian_index(row, column);
                result.upper_hessian[entry] = first_derivative * operand.upper_hessian[entry] + second_derivative * operand.gradient[row] * operand.gradient[column];
            }
        }
        return result;
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> inverse(const SecondOrder<Dimension>& operand) {
        const float inverse_value = 1.0F / operand.value;
        return compose(operand, inverse_value, -inverse_value * inverse_value, 2.0F * inverse_value * inverse_value * inverse_value);
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> operator/(const SecondOrder<Dimension>& numerator, const SecondOrder<Dimension>& denominator) {
        return numerator * inverse(denominator);
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> square_root(const SecondOrder<Dimension>& operand) {
        const float value = sqrtf(operand.value);
        return compose(operand, value, 0.5F / value, -0.25F / (operand.value * value));
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> atan2(const SecondOrder<Dimension>& y, const SecondOrder<Dimension>& x) {
        SecondOrder<Dimension> result{};
        const float squared_radius = x.value * x.value + y.value * y.value;
        const float fourth_radius  = squared_radius * squared_radius;
        const float derivative_y   = x.value / squared_radius;
        const float derivative_x   = -y.value / squared_radius;
        const float derivative_yy  = -2.0F * x.value * y.value / fourth_radius;
        const float derivative_xx  = 2.0F * x.value * y.value / fourth_radius;
        const float derivative_yx  = (y.value * y.value - x.value * x.value) / fourth_radius;
        result.value = atan2f(y.value, x.value);
        for (std::uint32_t entry = 0u; entry < Dimension; ++entry) result.gradient[entry] = derivative_y * y.gradient[entry] + derivative_x * x.gradient[entry];
        for (std::uint32_t row = 0u; row < Dimension; ++row) {
            for (std::uint32_t column = row; column < Dimension; ++column) {
                const std::uint32_t entry = SecondOrder<Dimension>::hessian_index(row, column);
                result.upper_hessian[entry] = derivative_y * y.upper_hessian[entry] + derivative_x * x.upper_hessian[entry] + derivative_yy * y.gradient[row] * y.gradient[column] + derivative_xx * x.gradient[row] * x.gradient[column] + derivative_yx * (y.gradient[row] * x.gradient[column] + x.gradient[row] * y.gradient[column]);
            }
        }
        return result;
    }

    template <std::uint32_t Dimension>
    struct SecondVector3 final {
        SecondOrder<Dimension> x;
        SecondOrder<Dimension> y;
        SecondOrder<Dimension> z;
    };

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondVector3<Dimension> operator-(const SecondVector3<Dimension>& first, const SecondVector3<Dimension>& second) {
        return {.x = first.x - second.x, .y = first.y - second.y, .z = first.z - second.z};
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondVector3<Dimension> operator/(const SecondVector3<Dimension>& numerator, const SecondOrder<Dimension>& denominator) {
        return {.x = numerator.x / denominator, .y = numerator.y / denominator, .z = numerator.z / denominator};
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> dot(const SecondVector3<Dimension>& first, const SecondVector3<Dimension>& second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondVector3<Dimension> cross(const SecondVector3<Dimension>& first, const SecondVector3<Dimension>& second) {
        return {
            .x = first.y * second.z - first.z * second.y,
            .y = first.z * second.x - first.x * second.z,
            .z = first.x * second.y - first.y * second.x,
        };
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondOrder<Dimension> length(const SecondVector3<Dimension>& vector) {
        return square_root(dot(vector, vector));
    }

    template <std::uint32_t Dimension>
    [[nodiscard]] __device__ SecondVector3<Dimension> normalized(const SecondVector3<Dimension>& vector) {
        return vector / length(vector);
    }
} // namespace physica::deformables::cloth::solvers::discrete_shells

#endif
