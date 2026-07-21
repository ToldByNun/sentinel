#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

/// <summary>row major float matrix: data[row][col]</summary>
class Matrix {
public:
    std::vector<std::vector<float>> data;

    Matrix() = default;
    explicit Matrix(std::vector<std::vector<float>> values);

    /// <summary>matrix product left * right</summary>
    static Matrix multiply(const Matrix& left, const Matrix& right);

    /// <summary>swap rows/cols</summary>
    static Matrix transpose(const Matrix& matrix);

    /// <summary>element-wise add. same shape required</summary>
    static Matrix add(const Matrix& left, const Matrix& right);

    /// <summary>element wise subtract. same shape required</summary>
    static Matrix subtract(const Matrix& left, const Matrix& right);

    /// <summary>multiply every element by scalar</summary>
    static Matrix scale(const Matrix& matrix, float scalar);

    /// <summary>element wise multiply. same shape required</summary>
    static Matrix multiplyElementwise(const Matrix& left, const Matrix& right);
};

#endif // MATRIX_HPP
