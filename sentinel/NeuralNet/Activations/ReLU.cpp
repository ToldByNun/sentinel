#include "ReLU.hpp"

Matrix ReLU::apply(Matrix matrix) {
    for (size_t row = 0; row < matrix.data.size(); row++) {
        for (size_t col = 0; col < matrix.data[row].size(); col++) {
            matrix.data[row][col] = matrix.data[row][col] > 0.0f ? matrix.data[row][col] : 0.0f;
        }
    }
    return matrix;
}

Matrix ReLU::derivative(Matrix matrix) {
    for (size_t row = 0; row < matrix.data.size(); row++) {
        for (size_t col = 0; col < matrix.data[row].size(); col++) {
            matrix.data[row][col] = matrix.data[row][col] > 0.0f ? 1.0f : 0.0f;
        }
    }
    return matrix;
}
