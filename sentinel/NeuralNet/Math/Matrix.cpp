#include "Matrix.hpp"

#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

Matrix::Matrix(size_t rowCount, size_t columnCount, float fillValue) {
    this->resize(rowCount, columnCount, fillValue);
}

float& Matrix::at(size_t row, size_t column) {
    return this->data[row * this->cols + column];
}

const float& Matrix::at(size_t row, size_t column) const {
    return this->data[row * this->cols + column];
}

bool Matrix::empty() const {
    return this->rows == 0 || this->cols == 0;
}

void Matrix::resize(size_t rowCount, size_t columnCount, float fillValue) {
    this->rows = rowCount;
    this->cols = columnCount;
    this->data.assign(rowCount * columnCount, fillValue);
}

void Matrix::fill(float value) {
    for (size_t index = 0; index < this->data.size(); ++index)
        this->data[index] = value;
}

void Matrix::gemm(const Matrix& left, const Matrix& right, Matrix& out, bool transposeLeft, bool transposeRight) {
    if (left.empty() || right.empty()) throw std::invalid_argument("Matrix::gemm expects non-empty inputs");

    const size_t leftRows = transposeLeft ? left.cols : left.rows;
    const size_t leftCols = transposeLeft ? left.rows : left.cols;
    const size_t rightRows = transposeRight ? right.cols : right.rows;
    const size_t rightCols = transposeRight ? right.rows : right.cols;

    if (leftCols != rightRows) throw std::invalid_argument("Matrix::gemm shape mismatch");

    out.resize(leftRows, rightCols, 0.0f);

    const size_t work = leftRows * leftCols * rightCols;
#if defined(_OPENMP)
    if (work >= 65536 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < static_cast<int>(leftRows); ++row) {
            for (size_t shared = 0; shared < leftCols; ++shared) {
                const float leftValue = transposeLeft
                    ? left.at(shared, static_cast<size_t>(row))
                    : left.at(static_cast<size_t>(row), shared);
                for (size_t column = 0; column < rightCols; ++column) {
                    const float rightValue = transposeRight
                        ? right.at(column, shared)
                        : right.at(shared, column);
                    out.at(static_cast<size_t>(row), column) += leftValue * rightValue;
                }
            }
        }
        return;
    }
#endif

    for (size_t row = 0; row < leftRows; ++row) {
        for (size_t shared = 0; shared < leftCols; ++shared) {
            const float leftValue = transposeLeft ? left.at(shared, row) : left.at(row, shared);
            for (size_t column = 0; column < rightCols; ++column) {
                const float rightValue = transposeRight ? right.at(column, shared) : right.at(shared, column);
                out.at(row, column) += leftValue * rightValue;
            }
        }
    }
}

Matrix Matrix::multiply(const Matrix& left, const Matrix& right) {
    return Matrix::multiply(left, right, false, false);
}

Matrix Matrix::multiply(const Matrix& left, const Matrix& right, bool transposeLeft, bool transposeRight) {
    Matrix result;
    Matrix::gemm(left, right, result, transposeLeft, transposeRight);
    return result;
}

Matrix Matrix::transpose(const Matrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("Matrix::transpose expects a non-empty matrix");

    Matrix result(matrix.cols, matrix.rows, 0.0f);

#if defined(_OPENMP)
    if (matrix.rows * matrix.cols >= 4096 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < static_cast<int>(matrix.rows); ++row) {
            for (size_t column = 0; column < matrix.cols; ++column)
                result.at(column, static_cast<size_t>(row)) = matrix.at(static_cast<size_t>(row), column);
        }
        return result;
    }
#endif

    for (size_t row = 0; row < matrix.rows; ++row) {
        for (size_t column = 0; column < matrix.cols; ++column)
            result.at(column, row) = matrix.at(row, column);
    }
    return result;
}

Matrix Matrix::add(const Matrix& left, const Matrix& right) {
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("Matrix::add shape mismatch");

    Matrix result = left;
    const size_t elementCount = left.data.size();

#if defined(_OPENMP)
    if (elementCount >= 4096 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < static_cast<int>(elementCount); ++index)
            result.data[static_cast<size_t>(index)] += right.data[static_cast<size_t>(index)];
        return result;
    }
#endif

    for (size_t index = 0; index < elementCount; ++index)
        result.data[index] += right.data[index];
    return result;
}

Matrix Matrix::subtract(const Matrix& left, const Matrix& right) {
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("Matrix::subtract shape mismatch");

    Matrix result = left;
    const size_t elementCount = left.data.size();

#if defined(_OPENMP)
    if (elementCount >= 4096 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < static_cast<int>(elementCount); ++index)
            result.data[static_cast<size_t>(index)] -= right.data[static_cast<size_t>(index)];
        return result;
    }
#endif

    for (size_t index = 0; index < elementCount; ++index)
        result.data[index] -= right.data[index];
    return result;
}

Matrix Matrix::scale(const Matrix& matrix, float scalar) {
    Matrix result = matrix;
    Matrix::scaleInPlace(result, scalar);
    return result;
}

Matrix Matrix::multiplyElementwise(const Matrix& left, const Matrix& right) {
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("Matrix::multiplyElementwise shape mismatch");

    Matrix result = left;
    const size_t elementCount = left.data.size();

#if defined(_OPENMP)
    if (elementCount >= 4096 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < static_cast<int>(elementCount); ++index)
            result.data[static_cast<size_t>(index)] =
                left.data[static_cast<size_t>(index)] * right.data[static_cast<size_t>(index)];
        return result;
    }
#endif

    for (size_t index = 0; index < elementCount; ++index)
        result.data[index] = left.data[index] * right.data[index];
    return result;
}

void Matrix::addInPlace(Matrix& total, const Matrix& delta) {
    if (total.rows != delta.rows || total.cols != delta.cols) throw std::invalid_argument("Matrix::addInPlace shape mismatch");
    for (size_t index = 0; index < total.data.size(); ++index)
        total.data[index] += delta.data[index];
}

void Matrix::scaleInPlace(Matrix& matrix, float scalar) {
    for (size_t index = 0; index < matrix.data.size(); ++index)
        matrix.data[index] *= scalar;
}

Matrix Matrix::zerosLike(const Matrix& matrix) {
    return Matrix(matrix.rows, matrix.cols, 0.0f);
}

void Matrix::zeroInPlace(Matrix& matrix) {
    matrix.fill(0.0f);
}
