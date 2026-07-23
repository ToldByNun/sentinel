#include "SiLU.hpp"

#include <cmath>
#include <stdexcept>

static float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

void SiLU::applyInto(const Matrix& input, Matrix& out) {
    if (input.empty()) throw std::invalid_argument("SiLU::applyInto expects a non-empty matrix");
    out.ensureSize(input.rows, input.cols);
    for (size_t index = 0; index < input.data.size(); ++index) {
        const float value = input.data[index];
        out.data[index] = value * sigmoid(value);
    }
}

void SiLU::derivativeInto(const Matrix& input, Matrix& out) {
    if (input.empty()) throw std::invalid_argument("SiLU::derivativeInto expects a non-empty matrix");
    out.ensureSize(input.rows, input.cols);
    for (size_t index = 0; index < input.data.size(); ++index) {
        const float value = input.data[index];
        const float sigma = sigmoid(value);
        out.data[index] = sigma * (1.0f + value * (1.0f - sigma));
    }
}
