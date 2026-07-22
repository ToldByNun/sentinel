#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <vector>

/// <summary>contiguous row major float matrix data[row * cols + col]</summary>
class Matrix {
public:
    size_t rows = 0;
    size_t cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(size_t rowCount, size_t columnCount, float fillValue = 0.0f);

    /// <summary>element access row major</summary>
    float& at(size_t row, size_t column);

    /// <summary>const element access</summary>
    const float& at(size_t row, size_t column) const;

    /// <summary>true if rows or cols is zero</summary>
    bool empty() const;

    /// <summary>resize and optionally fill</summary>
    void resize(size_t rowCount, size_t columnCount, float fillValue = 0.0f);

    /// <summary>set every element</summary>
    void fill(float value);

    /// <summary>C = op(A) * op(B) writes into out resizing as needed</summary>
    static void gemm(const Matrix& left, const Matrix& right, Matrix& out, bool transposeLeft = false, bool transposeRight = false);

    /// <summary>matrix product left * right</summary>
    static Matrix multiply(const Matrix& left, const Matrix& right);

    /// <summary>matrix product op(left) * op(right)</summary>
    static Matrix multiply(const Matrix& left, const Matrix& right, bool transposeLeft, bool transposeRight);

    /// <summary>swap rows/cols</summary>
    static Matrix transpose(const Matrix& matrix);

    /// <summary>element wise add same shape required</summary>
    static Matrix add(const Matrix& left, const Matrix& right);

    /// <summary>element wise subtract same shape required</summary>
    static Matrix subtract(const Matrix& left, const Matrix& right);

    /// <summary>multiply every element by scalar</summary>
    static Matrix scale(const Matrix& matrix, float scalar);

    /// <summary>element wise multiply same shape required</summary>
    static Matrix multiplyElementwise(const Matrix& left, const Matrix& right);

    /// <summary>total += delta in place</summary>
    static void addInPlace(Matrix& total, const Matrix& delta);

    /// <summary>matrix *= scalar in place</summary>
    static void scaleInPlace(Matrix& matrix, float scalar);

    /// <summary>same shape filled with zeros</summary>
    static Matrix zerosLike(const Matrix& matrix);

    /// <summary>zero every element in place</summary>
    static void zeroInPlace(Matrix& matrix);
};

#endif // MATRIX_HPP
