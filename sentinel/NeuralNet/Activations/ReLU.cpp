#include "ReLU.hpp"

Matrix ReLU::apply(Matrix matrix) {
    for (size_t row = 0; row < matrix.rows; row++) {
        for (size_t column = 0; column < matrix.cols; column++)
            matrix.at(row, column) = matrix.at(row, column) > 0.0f ? matrix.at(row, column) : 0.0f;
    }

    return matrix;
}

Matrix ReLU::derivative(Matrix matrix) {
    for (size_t row = 0; row < matrix.rows; row++) {
        for (size_t column = 0; column < matrix.cols; column++)
            matrix.at(row, column) = matrix.at(row, column) > 0.0f ? 1.0f : 0.0f;
    }

    return matrix;
}
