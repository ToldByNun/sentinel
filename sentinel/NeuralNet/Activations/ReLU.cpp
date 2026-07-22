#include "ReLU.hpp"

#include <stdexcept>

void ReLU::applyInto(const Matrix& input, Matrix& out) {
    if (input.empty()) throw std::invalid_argument("ReLU::applyInto expects a non-empty matrix");
    out.ensureSize(input.rows, input.cols);
    for (size_t index = 0; index < input.data.size(); ++index)
        out.data[index] = input.data[index] > 0.0f ? input.data[index] : 0.0f;
}

void ReLU::derivativeInto(const Matrix& input, Matrix& out) {
    if (input.empty()) throw std::invalid_argument("ReLU::derivativeInto expects a non-empty matrix");
    out.ensureSize(input.rows, input.cols);
    for (size_t index = 0; index < input.data.size(); ++index)
        out.data[index] = input.data[index] > 0.0f ? 1.0f : 0.0f;
}

Matrix ReLU::apply(Matrix matrix) {
    ReLU::applyInto(matrix, matrix);
    return matrix;
}

Matrix ReLU::derivative(Matrix matrix) {
    ReLU::derivativeInto(matrix, matrix);
    return matrix;
}
