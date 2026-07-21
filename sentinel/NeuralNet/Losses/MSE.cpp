#include "MSE.hpp"

float MSE::loss(const Matrix& prediction, const Matrix& target) {
    Matrix diff = Matrix::subtract(prediction, target);

    size_t rows = diff.data.size();
    size_t cols = rows > 0 ? diff.data[0].size() : 0;
    float total_elements = static_cast<float>(rows * cols);

    float sum = 0.0f;
    for (const auto& row : diff.data)
        for (float v : row)
            sum += v * v;

    return sum / total_elements;
}

Matrix MSE::gradient(const Matrix& prediction, const Matrix& target) {
    Matrix diff = Matrix::subtract(prediction, target);

    size_t rows = diff.data.size();
    size_t cols = rows > 0 ? diff.data[0].size() : 0;
    float total_elements = static_cast<float>(rows * cols);

    return Matrix::scale(diff, 2.0f / total_elements);
}