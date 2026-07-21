#include "MSE.hpp"

float MSE::loss(const Matrix& prediction, const Matrix& target) {
    Matrix diff = Matrix::subtract(prediction, target);

    size_t rows = diff.data.size();
    size_t cols = rows > 0 ? diff.data[0].size() : 0;
    float totalElements = static_cast<float>(rows * cols);

    float sum = 0.0f;
    for (const auto& row : diff.data) {
        for (float value : row) {
            sum += value * value;
        }
    }

    return sum / totalElements;
}

Matrix MSE::gradient(const Matrix& prediction, const Matrix& target) {
    Matrix diff = Matrix::subtract(prediction, target);

    size_t rows = diff.data.size();
    size_t cols = rows > 0 ? diff.data[0].size() : 0;
    float totalElements = static_cast<float>(rows * cols);

    return Matrix::scale(diff, 2.0f / totalElements);
}
