#include "Matrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

static constexpr size_t gemmBlockSize = 32;

/// <summary>pack transposed src into contiguous outRows x outCols row major</summary>
static void packTransposed(const Matrix& src, size_t outRows, size_t outCols, std::vector<float>& packed) {
    if (src.rows != outCols || src.cols != outRows) throw std::invalid_argument("packTransposed shape mismatch");
    packed.resize(outRows * outCols);
    for (size_t row = 0; row < outRows; ++row) {
        for (size_t column = 0; column < outCols; ++column)
            packed[row * outCols + column] = src.data[column * src.cols + row];
    }
}

/// <summary>blocked C = A * B for contiguous row major panels</summary>
static void blockedGemmNN(const float* left, size_t leftLeadingDimension, const float* right, size_t rightLeadingDimension, float* out, size_t outLeadingDimension, size_t rowCount, size_t columnCount, size_t sharedCount) {
    for (size_t rowBlock = 0; rowBlock < rowCount; rowBlock += gemmBlockSize) {
        const size_t rowEnd = (std::min)(rowBlock + gemmBlockSize, rowCount);
        for (size_t columnBlock = 0; columnBlock < columnCount; columnBlock += gemmBlockSize) {
            const size_t columnEnd = (std::min)(columnBlock + gemmBlockSize, columnCount);
            for (size_t sharedBlock = 0; sharedBlock < sharedCount; sharedBlock += gemmBlockSize) {
                const size_t sharedEnd = (std::min)(sharedBlock + gemmBlockSize, sharedCount);
                for (size_t row = rowBlock; row < rowEnd; ++row) {
                    for (size_t shared = sharedBlock; shared < sharedEnd; ++shared) {
                        const float leftValue = left[row * leftLeadingDimension + shared];
                        float* outRow = out + row * outLeadingDimension;
                        const float* rightRow = right + shared * rightLeadingDimension;
                        for (size_t column = columnBlock; column < columnEnd; ++column)
                            outRow[column] += leftValue * rightRow[column];
                    }
                }
            }
        }
    }
}

/// <summary>OpenMP over row blocks of blockedGemmNN</summary>
static void blockedGemmNNParallel(const float* left, size_t leftLeadingDimension, const float* right, size_t rightLeadingDimension, float* out, size_t outLeadingDimension, size_t rowCount, size_t columnCount, size_t sharedCount) {
#if defined(_OPENMP)
    const int rowBlockCount = static_cast<int>((rowCount + gemmBlockSize - 1) / gemmBlockSize);
    #pragma omp parallel for schedule(static)
    for (int rowBlockIndex = 0; rowBlockIndex < rowBlockCount; ++rowBlockIndex) {
        const size_t rowBlock = static_cast<size_t>(rowBlockIndex) * gemmBlockSize;
        const size_t rowEnd = (std::min)(rowBlock + gemmBlockSize, rowCount);
        for (size_t columnBlock = 0; columnBlock < columnCount; columnBlock += gemmBlockSize) {
            const size_t columnEnd = (std::min)(columnBlock + gemmBlockSize, columnCount);
            for (size_t sharedBlock = 0; sharedBlock < sharedCount; sharedBlock += gemmBlockSize) {
                const size_t sharedEnd = (std::min)(sharedBlock + gemmBlockSize, sharedCount);
                for (size_t row = rowBlock; row < rowEnd; ++row) {
                    for (size_t shared = sharedBlock; shared < sharedEnd; ++shared) {
                        const float leftValue = left[row * leftLeadingDimension + shared];
                        float* outRow = out + row * outLeadingDimension;
                        const float* rightRow = right + shared * rightLeadingDimension;
                        for (size_t column = columnBlock; column < columnEnd; ++column)
                            outRow[column] += leftValue * rightRow[column];
                    }
                }
            }
        }
    }
#else
    blockedGemmNN(left, leftLeadingDimension, right, rightLeadingDimension, out, outLeadingDimension, rowCount, columnCount, sharedCount);
#endif
}

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

void Matrix::ensureSize(size_t rowCount, size_t columnCount) {
    if (this->rows == rowCount && this->cols == columnCount) return;
    this->resize(rowCount, columnCount, 0.0f);
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

    if (out.rows != leftRows || out.cols != rightCols)
        out.resize(leftRows, rightCols, 0.0f);
    else
        out.fill(0.0f);

    thread_local std::vector<float> packedLeft;
    thread_local std::vector<float> packedRight;

    const float* leftPointer = left.data.data();
    size_t leftLeadingDimension = left.cols;
    if (transposeLeft) {
        packTransposed(left, leftRows, leftCols, packedLeft);
        leftPointer = packedLeft.data();
        leftLeadingDimension = leftCols;
    }

    const float* rightPointer = right.data.data();
    size_t rightLeadingDimension = right.cols;
    if (transposeRight) {
        packTransposed(right, rightRows, rightCols, packedRight);
        rightPointer = packedRight.data();
        rightLeadingDimension = rightCols;
    }

    const size_t work = leftRows * leftCols * rightCols;
#if defined(_OPENMP)
    if (work >= 65536 && !omp_in_parallel()) {
        blockedGemmNNParallel(leftPointer, leftLeadingDimension, rightPointer, rightLeadingDimension, out.data.data(), out.cols, leftRows, rightCols, leftCols);
        return;
    }
#endif

    blockedGemmNN(leftPointer, leftLeadingDimension, rightPointer, rightLeadingDimension, out.data.data(), out.cols, leftRows, rightCols, leftCols);
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

void Matrix::multiplyElementwiseInPlace(Matrix& total, const Matrix& other) {
    if (total.rows != other.rows || total.cols != other.cols) throw std::invalid_argument("Matrix::multiplyElementwiseInPlace shape mismatch");
    for (size_t index = 0; index < total.data.size(); ++index)
        total.data[index] *= other.data[index];
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
