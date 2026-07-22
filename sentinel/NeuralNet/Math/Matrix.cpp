#include "Matrix.hpp"

#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

Matrix::Matrix(std::vector<std::vector<float>> values) : data(std::move(values)) {}

Matrix Matrix::multiply(const Matrix& left, const Matrix& right) {
    const size_t rowsLeft = left.data.size();
    const size_t columnsLeft = left.data[0].size();
    const size_t columnsRight = right.data[0].size();

    Matrix result;
    result.data = std::vector<std::vector<float>>(rowsLeft, std::vector<float>(columnsRight, 0.0f));

    const size_t work = rowsLeft * columnsLeft * columnsRight;
#if defined(_OPENMP)
    if (work >= 65536 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < static_cast<int>(rowsLeft); ++row) {
            for (size_t shared = 0; shared < columnsLeft; ++shared) {
                const float leftValue = left.data[static_cast<size_t>(row)][shared];
                for (size_t column = 0; column < columnsRight; ++column)
                    result.data[static_cast<size_t>(row)][column] += leftValue * right.data[shared][column];
            }
        }
        return result;
    }
#endif

    for (size_t row = 0; row < rowsLeft; ++row) {
        for (size_t shared = 0; shared < columnsLeft; ++shared) {
            const float leftValue = left.data[row][shared];
            for (size_t column = 0; column < columnsRight; ++column)
                result.data[row][column] += leftValue * right.data[shared][column];
        }
    }
    return result;
}

Matrix Matrix::transpose(const Matrix& matrix) {
    const size_t rows = matrix.data.size();
    const size_t columns = matrix.data[0].size();

    Matrix result;
    result.data = std::vector<std::vector<float>>(columns, std::vector<float>(rows, 0.0f));

#if defined(_OPENMP)
    if (rows * columns >= 4096) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < static_cast<int>(rows); ++row) {
            for (size_t column = 0; column < columns; ++column)
                result.data[column][static_cast<size_t>(row)] = matrix.data[static_cast<size_t>(row)][column];
        }
        return result;
    }
#endif

    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column)
            result.data[column][row] = matrix.data[row][column];
    }
    return result;
}

Matrix Matrix::add(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    const size_t rows = left.data.size();
    const size_t columns = left.data[0].size();

#if defined(_OPENMP)
    if (rows * columns >= 4096) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < static_cast<int>(rows); ++row) {
            for (size_t column = 0; column < columns; ++column)
                result.data[static_cast<size_t>(row)][column] += right.data[static_cast<size_t>(row)][column];
        }
        return result;
    }
#endif

    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column)
            result.data[row][column] += right.data[row][column];
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    const size_t rows = left.data.size();
    const size_t columns = left.data[0].size();

#if defined(_OPENMP)
    if (rows * columns >= 4096) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < static_cast<int>(rows); ++row) {
            for (size_t column = 0; column < columns; ++column)
                result.data[static_cast<size_t>(row)][column] -= right.data[static_cast<size_t>(row)][column];
        }
        return result;
    }
#endif

    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column)
            result.data[row][column] -= right.data[row][column];
    }
    return result;
}

Matrix Matrix::scale(const Matrix& matrix, float scalar) {
    Matrix result = matrix;
    const size_t rows = matrix.data.size();
    const size_t columns = matrix.data[0].size();

#if defined(_OPENMP)
    if (rows * columns >= 4096) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < static_cast<int>(rows); ++row) {
            for (size_t column = 0; column < columns; ++column)
                result.data[static_cast<size_t>(row)][column] *= scalar;
        }
        return result;
    }
#endif

    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column)
            result.data[row][column] *= scalar;
    }
    return result;
}

Matrix Matrix::multiplyElementwise(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    const size_t rows = left.data.size();
    const size_t columns = left.data[0].size();

#if defined(_OPENMP)
    if (rows * columns >= 4096) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < static_cast<int>(rows); ++row) {
            for (size_t column = 0; column < columns; ++column)
                result.data[static_cast<size_t>(row)][column] =
                    left.data[static_cast<size_t>(row)][column] * right.data[static_cast<size_t>(row)][column];
        }
        return result;
    }
#endif

    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column)
            result.data[row][column] = left.data[row][column] * right.data[row][column];
    }
    return result;
}

void Matrix::addInPlace(Matrix& total, const Matrix& delta) {
    for (size_t row = 0; row < total.data.size(); ++row) {
        for (size_t column = 0; column < total.data[row].size(); ++column)
            total.data[row][column] += delta.data[row][column];
    }
}

void Matrix::scaleInPlace(Matrix& matrix, float scalar) {
    for (size_t row = 0; row < matrix.data.size(); ++row) {
        for (size_t column = 0; column < matrix.data[row].size(); ++column)
            matrix.data[row][column] *= scalar;
    }
}

Matrix Matrix::zerosLike(const Matrix& matrix) {
    Matrix result = matrix;
    for (size_t row = 0; row < result.data.size(); ++row) {
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] = 0.0f;
    }
    return result;
}
