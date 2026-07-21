#include "Matrix.hpp"

#include <utility>

Matrix::Matrix(std::vector<std::vector<float>> values) : data(std::move(values)) {}

Matrix Matrix::multiply(const Matrix& left, const Matrix& right) {
    size_t rowsLeft = left.data.size();
    size_t colsLeft = left.data[0].size();
    size_t colsRight = right.data[0].size();

    Matrix result;
    result.data = std::vector<std::vector<float>>(rowsLeft, std::vector<float>(colsRight, 0.0f));

    for (size_t row = 0; row < rowsLeft; row++) {
        for (size_t col = 0; col < colsRight; col++) {
            for (size_t shared = 0; shared < colsLeft; shared++) {
                result.data[row][col] += left.data[row][shared] * right.data[shared][col];
            }
        }
    }
    return result;
}

Matrix Matrix::transpose(const Matrix& matrix) {
    size_t rows = matrix.data.size();
    size_t cols = matrix.data[0].size();

    Matrix result;
    result.data = std::vector<std::vector<float>>(cols, std::vector<float>(rows, 0.0f));

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            result.data[col][row] = matrix.data[row][col];
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    for (size_t row = 0; row < left.data.size(); row++) {
        for (size_t col = 0; col < left.data[row].size(); col++) {
            result.data[row][col] += right.data[row][col];
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    for (size_t row = 0; row < left.data.size(); row++) {
        for (size_t col = 0; col < left.data[row].size(); col++) {
            result.data[row][col] -= right.data[row][col];
        }
    }
    return result;
}

Matrix Matrix::scale(const Matrix& matrix, float scalar) {
    Matrix result = matrix;
    for (size_t row = 0; row < matrix.data.size(); row++) {
        for (size_t col = 0; col < matrix.data[row].size(); col++) {
            result.data[row][col] *= scalar;
        }
    }
    return result;
}

Matrix Matrix::multiplyElementwise(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    for (size_t row = 0; row < left.data.size(); ++row) {
        for (size_t col = 0; col < left.data[row].size(); ++col) {
            result.data[row][col] = left.data[row][col] * right.data[row][col];
        }
    }
    return result;
}
