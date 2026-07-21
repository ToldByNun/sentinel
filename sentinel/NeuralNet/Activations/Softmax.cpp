#include "Softmax.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

Matrix Softmax::apply(const Matrix& logits) {
    if (logits.data.empty() || logits.data[0].empty()) throw std::invalid_argument("Softmax::apply expects a non-empty matrix");

    const size_t classCount = logits.data.size();
    const size_t columnCount = logits.data[0].size();

    Matrix probabilities;
    probabilities.data = std::vector<std::vector<float>>(
        classCount,
        std::vector<float>(columnCount, 0.0f)
    );

    for (size_t column = 0; column < columnCount; ++column) {
        float maxLogit = logits.data[0][column];
        for (size_t classIndex = 1; classIndex < classCount; ++classIndex) {
            if (logits.data[classIndex][column] > maxLogit)
                maxLogit = logits.data[classIndex][column];
        }

        float exponentialSum = 0.0f;
        for (size_t classIndex = 0; classIndex < classCount; ++classIndex) {
            const float value = std::exp(logits.data[classIndex][column] - maxLogit);
            probabilities.data[classIndex][column] = value;
            exponentialSum += value;
        }

        for (size_t classIndex = 0; classIndex < classCount; ++classIndex)
            probabilities.data[classIndex][column] /= exponentialSum;
    }

    return probabilities;
}
