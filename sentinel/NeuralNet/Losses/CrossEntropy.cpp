#include "CrossEntropy.hpp"

#include <cmath>
#include <stdexcept>

float CrossEntropy::loss(const Matrix& probabilities, const Matrix& target) {
    if (probabilities.data.size() != target.data.size() || probabilities.data.empty() || probabilities.data[0].size() != target.data[0].size()) throw std::invalid_argument("CrossEntropy::loss shape mismatch");

    float total = 0.0f;

    for (size_t row = 0; row < probabilities.data.size(); ++row) {
        for (size_t column = 0; column < probabilities.data[row].size(); ++column) {
            const float rawProbability = probabilities.data[row][column];
            const float probability = (rawProbability > 1e-7f) ? rawProbability : 1e-7f;
            total += -target.data[row][column] * std::log(probability);
        }
    }

    const size_t columnCount = probabilities.data[0].size();
    return total / static_cast<float>(columnCount);
}

Matrix CrossEntropy::gradient(const Matrix& probabilities, const Matrix& target) {
    if (probabilities.data.size() != target.data.size() || probabilities.data.empty() || probabilities.data[0].size() != target.data[0].size())
        throw std::invalid_argument("CrossEntropy::gradient shape mismatch");

    // dL/d(logits) = softmax(logits) - target
    return Matrix::subtract(probabilities, target);
}
