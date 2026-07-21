#include "Matrix.hpp"

#include <utility>

Matrix::Matrix(std::vector<std::vector<float>> values) : data(std::move(values)) {}

Matrix Matrix::multiply(const Matrix& left, const Matrix& right) {
    size_t rowsLeft = left.data.size();
    size_t columnsLeft = left.data[0].size();
    size_t columnsRight = right.data[0].size();

    Matrix result;
    result.data = std::vector<std::vector<float>>(rowsLeft, std::vector<float>(columnsRight, 0.0f));

    for (size_t row = 0; row < rowsLeft; row++) {
        for (size_t column = 0; column < columnsRight; column++) {
            for (size_t shared = 0; shared < columnsLeft; shared++)
                result.data[row][column] += left.data[row][shared] * right.data[shared][column];
        }
    }
    return result;
}

Matrix Matrix::transpose(const Matrix& matrix) {
    size_t rows = matrix.data.size();
    size_t columns = matrix.data[0].size();

    Matrix result;
    result.data = std::vector<std::vector<float>>(columns, std::vector<float>(rows, 0.0f));

    for (size_t row = 0; row < rows; row++) {
        for (size_t column = 0; column < columns; column++)
            result.data[column][row] = matrix.data[row][column];
    }
    return result;
}

Matrix Matrix::add(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    for (size_t row = 0; row < left.data.size(); row++) {
        for (size_t column = 0; column < left.data[row].size(); column++)
            result.data[row][column] += right.data[row][column];
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    for (size_t row = 0; row < left.data.size(); row++) {
        for (size_t column = 0; column < left.data[row].size(); column++)
            result.data[row][column] -= right.data[row][column];
    }
    return result;
}

Matrix Matrix::scale(const Matrix& matrix, float scalar) {
    Matrix result = matrix;
    for (size_t row = 0; row < matrix.data.size(); row++) {
        for (size_t column = 0; column < matrix.data[row].size(); column++)
            result.data[row][column] *= scalar;
    }
    return result;
}

Matrix Matrix::multiplyElementwise(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    for (size_t row = 0; row < left.data.size(); ++row) {
        for (size_t column = 0; column < left.data[row].size(); ++column)
            result.data[row][column] = left.data[row][column] * right.data[row][column];
    }
    return result;
}
