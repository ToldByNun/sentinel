#include "MSE.hpp"

float MSE::loss(const Matrix& prediction, const Matrix& target) {
    Matrix difference = Matrix::subtract(prediction, target);

    size_t rows = difference.data.size();
    size_t columns = rows > 0 ? difference.data[0].size() : 0;
    float totalElements = static_cast<float>(rows * columns);

    float sum = 0.0f;
    for (const auto& row : difference.data) {
        for (float value : row) sum += value * value;
    }

    return sum / totalElements;
}

Matrix MSE::gradient(const Matrix& prediction, const Matrix& target) {
    Matrix difference = Matrix::subtract(prediction, target);

    size_t rows = difference.data.size();
    size_t columns = rows > 0 ? difference.data[0].size() : 0;
    float totalElements = static_cast<float>(rows * columns);

    return Matrix::scale(difference, 2.0f / totalElements);
}
