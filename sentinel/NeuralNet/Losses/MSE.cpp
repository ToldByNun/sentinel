#include "MSE.hpp"

float MSE::loss(const Matrix& prediction, const Matrix& target) {
    Matrix difference = Matrix::subtract(prediction, target);

    const size_t rows = difference.rows;
    const size_t columns = difference.cols;
    const float totalElements = static_cast<float>(rows * columns);

    float sum = 0.0f;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            const float value = difference.at(row, column);
            sum += value * value;
        }
    }

    return sum / totalElements;
}

Matrix MSE::gradient(const Matrix& prediction, const Matrix& target) {
    Matrix difference = Matrix::subtract(prediction, target);

    const size_t rows = difference.rows;
    const size_t columns = difference.cols;
    const float totalElements = static_cast<float>(rows * columns);

    return Matrix::scale(difference, 2.0f / totalElements);
}
