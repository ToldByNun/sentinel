#include "ReLU.hpp"

Matrix ReLU::apply(Matrix matrix) {
    for (size_t row = 0; row < matrix.data.size(); row++) {
        for (size_t column = 0; column < matrix.data[row].size(); column++)
            matrix.data[row][column] = matrix.data[row][column] > 0.0f ? matrix.data[row][column] : 0.0f;
    }

    return matrix;
}

Matrix ReLU::derivative(Matrix matrix) {
    for (size_t row = 0; row < matrix.data.size(); row++) {
        for (size_t column = 0; column < matrix.data[row].size(); column++)
            matrix.data[row][column] = matrix.data[row][column] > 0.0f ? 1.0f : 0.0f;
    }

    return matrix;
}
