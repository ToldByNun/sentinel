#include "CrossEntropy.hpp"

#include <cmath>
#include <stdexcept>

float CrossEntropy::loss(const Matrix& probabilities, const Matrix& target) {
    if (probabilities.rows != target.rows || probabilities.empty() || probabilities.cols != target.cols) throw std::invalid_argument("CrossEntropy::loss shape mismatch");

    float total = 0.0f;

    for (size_t row = 0; row < probabilities.rows; ++row) {
        for (size_t column = 0; column < probabilities.cols; ++column) {
            const float rawProbability = probabilities.at(row, column);
            const float probability = (rawProbability > 1e-7f) ? rawProbability : 1e-7f;
            total += -target.at(row, column) * std::log(probability);
        }
    }

    const size_t columnCount = probabilities.cols;
    return total / static_cast<float>(columnCount);
}

Matrix CrossEntropy::gradient(const Matrix& probabilities, const Matrix& target) {
    if (probabilities.rows != target.rows || probabilities.empty() || probabilities.cols != target.cols)
        throw std::invalid_argument("CrossEntropy::gradient shape mismatch");

    // matches loss averaged over columns: dL/d(logits) = (softmax - target) / columnCount
    const float inverseColumnCount = 1.0f / static_cast<float>(probabilities.cols);
    return Matrix::scale(Matrix::subtract(probabilities, target), inverseColumnCount);
}
