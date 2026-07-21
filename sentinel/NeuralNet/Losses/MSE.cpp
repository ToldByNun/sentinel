#include "MSE.hpp"

Matrix MSE::compute(const Matrix& prediction, const Matrix& target) {
    return Matrix::subtract(prediction, target);
}
