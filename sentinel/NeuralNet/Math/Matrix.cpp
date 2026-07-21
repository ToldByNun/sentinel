#include "Matrix.hpp"

#include <utility>

Matrix::Matrix(std::vector<std::vector<float>> values) : data(std::move(values)) {}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    size_t rowsA = a.data.size();
    size_t colsA = a.data[0].size();
    size_t colsB = b.data[0].size();

    Matrix result;
    result.data = std::vector<std::vector<float>>(rowsA, std::vector<float>(colsB, 0.0f));

    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            for (size_t k = 0; k < colsA; k++) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::transpose(const Matrix& a) {
    size_t rows = a.data.size();
    size_t cols = a.data[0].size();

    Matrix result;
    result.data = std::vector<std::vector<float>>(cols, std::vector<float>(rows, 0.0f));

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[j][i] = a.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            result.data[i][j] += b.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            result.data[i][j] -= b.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::scale(const Matrix& a, float s) {
    Matrix result = a;
    for (size_t i = 0; i < a.data.size(); i++) {
        for (size_t j = 0; j < a.data[i].size(); j++) {
            result.data[i][j] *= s;
        }
    }
    return result;
}

Matrix Matrix::multiplyElementwise(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    for (size_t i = 0; i < a.data.size(); ++i) {
        for (size_t j = 0; j < a.data[i].size(); ++j) {
            result.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }
    return result;
}
