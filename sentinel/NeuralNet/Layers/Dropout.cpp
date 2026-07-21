#include "Dropout.hpp"

#include <stdexcept>

Dropout::Dropout(float dropRate, unsigned seed) : dropRate(dropRate), seed(seed) {
    if (dropRate < 0.0f || dropRate >= 1.0f) throw std::invalid_argument("Dropout dropRate must be in [0, 1)");
}

unsigned Dropout::advanceSeed(unsigned seed) {
    return seed * 1664525u + 1013904223u;
}

Matrix Dropout::forward(const Matrix& input) {
    if (input.data.empty() || input.data[0].empty()) throw std::invalid_argument("Dropout::forward empty input");

    if (!this->training || this->dropRate == 0.0f) {
        this->lastMask = Matrix();
        return input;
    }

    const float keepProbability = 1.0f - this->dropRate;
    const float scale = 1.0f / keepProbability;

    this->lastMask.data = std::vector<std::vector<float>>(
        input.data.size(),
        std::vector<float>(input.data[0].size(), 0.0f)
    );

    Matrix result = input;
    for (size_t row = 0; row < input.data.size(); ++row) {
        for (size_t column = 0; column < input.data[row].size(); ++column) {
            this->seed = Dropout::advanceSeed(this->seed);
            const float unit = static_cast<float>(this->seed % 10001u) / 10000.0f;
            if (unit < this->dropRate) {
                this->lastMask.data[row][column] = 0.0f;
                result.data[row][column] = 0.0f;
                continue;
            }

            this->lastMask.data[row][column] = scale;
            result.data[row][column] = input.data[row][column] * scale;
        }
    }

    return result;
}

Matrix Dropout::backward(const Matrix& outputGradient) const {
    if (!this->training || this->dropRate == 0.0f) return outputGradient;
    if (this->lastMask.data.empty()) throw std::invalid_argument("Dropout::backward called without a training forward");
    if (outputGradient.data.size() != this->lastMask.data.size() || outputGradient.data[0].size() != this->lastMask.data[0].size())
        throw std::invalid_argument("Dropout::backward shape mismatch");

    return Matrix::multiplyElementwise(outputGradient, this->lastMask);
}
